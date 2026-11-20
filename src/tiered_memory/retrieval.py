"""Cross-tier recall with Generative-Agents-style scoring and budget-aware packing.

Candidates come from all three tiers:

* **semantic**: facts that are true now (plus superseded ones when history is requested);
* **episodic**: for each episode, the sentences that best match the question (a snippet), scored by the best
  sentence's similarity. Sentences that produced facts are left to the semantic tier, which holds their current
  version, and values that are no longer true anywhere are masked as ``[superseded]``;
* **working**: the scratchpad of the open session(s).

Each candidate gets

    score = (w_rel · relevance + w_imp · importance + w_rec · recency) × penalty

where relevance (cosine similarity to the question) and recency (exponential decay since last access, per-tier
half-life) are min-max normalised across the candidate pool, as in the Generative Agents paper. The penalty
down-weights snippets that had outdated values masked and, in history mode, superseded facts.

Packing walks the ranking and adds every item that still fits the token budget, skipping near-duplicates, so one long
item cannot crowd out several short useful ones. Recalled facts and episodes are reinforced: their access count and
last-access time go up and their TTL is pushed out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .config import DAY, HOUR, MemoryConfig
from .consolidation import Consolidator
from .embeddings import Embedder
from .models import Fact, Provenance, RecallItem, RecallResult, ScoreBreakdown, Tier
from .predicates import normalise_text
from .scoring import recency
from .semantic import SemanticMemory
from .store import SQLiteStore
from .tokens import TokenCounter

WORKING_HALF_LIFE = 1.0 * HOUR
SUPERSEDED = "[superseded]"


@dataclass
class _Candidate:
    id: str
    tier: Tier
    text: str
    vector: np.ndarray
    relevance: float
    importance: float
    recency: float
    penalty: float
    provenance: Provenance


def outdated_value_pattern(facts: list[Fact]) -> re.Pattern[str] | None:
    """Regex matching values that are no longer true for any slot (superseded or retracted, not current again).

    Supersession propagates to episodic memory through it: "It's been raining in Turin all day" produced no fact,
    but once the user has moved away from Turin, "Turin" is masked in recalled episode text.
    """
    current = {normalise_text(f.object) for f in facts if f.status == "current"}
    values = {
        f.object
        for f in facts
        if f.status != "current" and len(f.object) >= 3 and normalise_text(f.object) not in current
    }
    if not values:
        return None
    alternatives = "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.I)


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)


class Retriever:
    """Recall for one namespace."""

    def __init__(
        self,
        *,
        store: SQLiteStore,
        embedder: Embedder,
        counter: TokenCounter,
        config: MemoryConfig,
        namespace: str,
        semantic: SemanticMemory,
        consolidator: Consolidator,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.counter = counter
        self.config = config
        self.namespace = namespace
        self.semantic = semantic
        self.consolidator = consolidator

    # -------------------------------------------------------------- candidates
    def _facts(self, facts: list[Fact], q: np.ndarray, now: float, include_history: bool) -> list[_Candidate]:
        selected = [f for f in facts if include_history or f.status == "current"]
        if not selected:
            return []
        ids, matrix = self.store.fact_vectors(self.namespace)
        index = {fid: i for i, fid in enumerate(ids)}
        sims = matrix @ q
        half_life = self.config.fact_half_life_days * DAY
        return [
            _Candidate(
                id=fact.id,
                tier="semantic",
                text=fact.text,
                vector=matrix[index[fact.id]],
                relevance=float(sims[index[fact.id]]),
                importance=fact.importance,
                recency=recency(now, fact.last_accessed, half_life),
                penalty=1.0 if fact.status == "current" else 1.0 - self.config.stale_penalty,
                provenance=Provenance(
                    tier="semantic",
                    session_id=fact.source_session,
                    source_episode=fact.source_episode,
                    created_at=fact.created_at,
                    valid_from=fact.valid_from,
                    valid_to=fact.valid_to,
                    status=fact.status,
                    confidence=fact.confidence,
                    mentions=fact.mentions,
                    access_count=fact.access_count,
                    note=f"{fact.subject} · {fact.predicate} · {fact.object}",
                ),
            )
            for fact in selected
        ]

    def _episodes(self, facts: list[Fact], q: np.ndarray, now: float, include_history: bool) -> list[_Candidate]:
        episodes = self.store.episodes(self.namespace)
        if not episodes:
            return []
        current = {f.id for f in facts if f.status == "current"}
        mask = None if include_history else outdated_value_pattern(facts)
        vectors = self.store.episode_vectors(self.namespace)
        half_life = self.config.episode_half_life_days * DAY
        out = []
        for episode in episodes:
            units = vectors[episode.id]
            sims = units @ q
            texts = [s.text for s in episode.sentences]
            outdated = {f for s in episode.sentences for f in s.fact_ids} - current
            # Fact-bearing sentences are represented by the semantic tier (current) or redacted (outdated).
            eligible = [i for i, s in enumerate(episode.sentences) if include_history or not s.fact_ids]
            if episode.abstract and (include_history or not outdated):
                texts.append(episode.abstract)
                eligible.append(len(texts) - 1)
            if not eligible:
                continue
            top = sorted(eligible, key=lambda i: -sims[i])[: self.config.snippet_sentences]
            text = " ".join(texts[i] for i in sorted(top))
            masked = 0
            if mask is not None:
                text, masked = mask.subn(SUPERSEDED, text)
            note = f"{len(top)} of {len(episode.sentences)} kept sentences"
            if masked:
                note += f", {masked} outdated value(s) masked"
            out.append(
                _Candidate(
                    id=episode.id,
                    tier="episodic",
                    text=text,
                    vector=units[top[0]],
                    relevance=float(sims[top[0]]),
                    importance=episode.importance,
                    recency=recency(now, episode.last_accessed, half_life),
                    penalty=1.0 - self.config.stale_penalty if masked else 1.0,
                    provenance=Provenance(
                        tier="episodic",
                        session_id=episode.session_id,
                        source_episode=episode.id,
                        created_at=episode.end_ts,
                        access_count=episode.access_count,
                        note=note,
                    ),
                )
            )
        return out

    def _working(self, q: np.ndarray, now: float, session_id: str | None) -> list[_Candidate]:
        if session_id is not None:
            items = self.store.working_items(self.namespace, session_id)
        else:
            items = [
                item
                for sid, _ in self.store.open_sessions(self.namespace)
                for item in self.store.working_items(self.namespace, sid)
            ]
        if not items:
            return []
        vectors = self.embedder.embed([i.text for i in items])
        sims = vectors @ q
        return [
            _Candidate(
                id=f"w-{item.id}",
                tier="working",
                text=item.text if item.kind != "turn" else f"{item.role}: {item.text}",
                vector=vectors[n],
                relevance=float(sims[n]),
                importance=item.importance,
                recency=recency(now, item.ts, WORKING_HALF_LIFE),
                penalty=1.0,
                provenance=Provenance(tier="working", session_id=item.session_id, created_at=item.ts, note=item.kind),
            )
            for n, item in enumerate(items)
        ]

    # ------------------------------------------------------------------ recall
    def recall(
        self,
        query: str,
        *,
        now: float,
        budget_tokens: int = 256,
        k: int | None = None,
        include_history: bool = False,
        session_id: str | None = None,
        reinforce: bool = True,
    ) -> RecallResult:
        q = self.embedder.embed([query])[0]
        facts = self.store.facts(self.namespace)
        candidates = (
            self._facts(facts, q, now, include_history)
            + self._episodes(facts, q, now, include_history)
            + self._working(q, now, session_id)
        )
        if not candidates:
            return RecallResult(query=query, budget_tokens=budget_tokens, used_tokens=0, items=[], considered=0)
        raw_relevance = np.clip(np.array([c.relevance for c in candidates]), 0.0, 1.0)
        rel = _minmax(raw_relevance)
        rec = _minmax(np.array([c.recency for c in candidates]))
        imp = np.array([c.importance for c in candidates])
        penalty = np.array([c.penalty for c in candidates])
        w = self.config.weights
        totals = (w.relevance * rel + w.importance * imp + w.recency * rec) * penalty

        items: list[RecallItem] = []
        chosen_vectors: list[np.ndarray] = []
        chosen_texts: set[str] = set()
        used = skipped = 0
        for i in np.argsort(-totals, kind="stable"):
            if k is not None and len(items) >= k:
                break
            cand = candidates[int(i)]
            norm = normalise_text(cand.text)
            if norm in chosen_texts or any(
                float(v @ cand.vector) >= self.config.dedupe_similarity for v in chosen_vectors
            ):
                continue
            tokens = self.counter(cand.text)
            if used + tokens > budget_tokens:
                skipped += 1
                continue
            used += tokens
            chosen_texts.add(norm)
            chosen_vectors.append(cand.vector)
            items.append(
                RecallItem(
                    id=cand.id,
                    tier=cand.tier,
                    text=cand.text,
                    tokens=tokens,
                    score=ScoreBreakdown(
                        relevance=round(float(rel[i]), 4),
                        relevance_raw=round(float(raw_relevance[i]), 4),
                        recency=round(float(rec[i]), 4),
                        importance=round(float(imp[i]), 4),
                        penalty=round(float(penalty[i]), 4),
                        total=round(float(totals[i]), 4),
                    ),
                    provenance=cand.provenance,
                )
            )
        if reinforce:
            self._reinforce(items, now)
        return RecallResult(
            query=query,
            budget_tokens=budget_tokens,
            used_tokens=used,
            items=items,
            considered=len(candidates),
            skipped_for_budget=skipped,
        )

    def _reinforce(self, items: list[RecallItem], now: float) -> None:
        for item in items:
            if item.tier == "semantic":
                fact = self.store.get_fact(item.id)
                if fact is not None:
                    expires = self.semantic.fact_expiry(now, fact.importance, fact.access_count + 1, fact.mentions)
                    self.store.touch(self.namespace, "semantic", item.id, ts=now, expires_at=expires)
            elif item.tier == "episodic":
                episode = self.store.get_episode(item.id)
                if episode is not None:
                    expires = self.consolidator.episode_expiry(now, episode.importance, episode.access_count + 1)
                    self.store.touch(self.namespace, "episodic", item.id, ts=now, expires_at=expires)
