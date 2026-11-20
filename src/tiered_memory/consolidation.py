"""Consolidation pipeline: working memory -> episodic memory -> semantic memory.

1. The flushed turns are split into sentences and scored for importance.
2. The compressor keeps salient sentences verbatim and fills a token target with TextRank picks (or an LLM summary).
3. Facts are extracted from the kept user sentences and merged into semantic memory; every fact is linked to the
   sentence that produced it, and every sentence to its facts, so supersession can later hide stale sentences.
4. The episode is stored with an embedding, an importance and a TTL.

Episode ids are content hashes, so consolidating the same segment twice is a no-op.
"""

from __future__ import annotations

from .compression import Compressor, split_sentences
from .config import DAY, MemoryConfig
from .embeddings import Embedder
from .extraction import FactExtractor
from .importance import ImportanceScorer
from .models import ConsolidationReport, Episode, EpisodeReason, EpisodeSentence, Role, WorkingItem
from .scoring import expiry
from .semantic import SemanticMemory
from .store import SQLiteStore, content_hash
from .tokens import TokenCounter

FACT_SOURCES = ("user", "note", "goal")
NOTE_IMPORTANCE = 0.9  # explicit notes are protected from TTL expiry whatever the scorer says
_LINKED = ("added", "reinforced", "merged", "superseded", "historical")


class Consolidator:
    """Runs the pipeline for one namespace."""

    def __init__(
        self,
        *,
        store: SQLiteStore,
        embedder: Embedder,
        counter: TokenCounter,
        config: MemoryConfig,
        namespace: str,
        importance: ImportanceScorer,
        compressor: Compressor,
        extractor: FactExtractor,
        semantic: SemanticMemory,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.counter = counter
        self.config = config
        self.namespace = namespace
        self.importance = importance
        self.compressor = compressor
        self.extractor = extractor
        self.semantic = semantic

    def episode_expiry(self, last_accessed: float, importance: float, access_count: int = 0) -> float | None:
        return expiry(
            last_accessed=last_accessed,
            ttl_seconds=self.config.episode_ttl_days * DAY,
            importance=importance,
            access_count=access_count,
            importance_gain=self.config.ttl_importance_gain,
            protect_importance=self.config.protect_importance,
        )

    def _sentences(self, items: list[WorkingItem]) -> list[EpisodeSentence]:
        records: list[tuple[str, float, Role]] = []
        for item in items:
            role: Role = item.kind if item.kind in ("goal", "note") else item.role
            records += [(text, item.ts, role) for text in split_sentences(item.text)]
        scores = self.importance.score_many([(text, role) for text, _, role in records])
        return [
            EpisodeSentence(
                text=text,
                ts=ts,
                role=role,
                importance=max(score.value, NOTE_IMPORTANCE) if role == "note" else score.value,
                salient=score.value >= self.config.salient_threshold and role in FACT_SOURCES,
            )
            for (text, ts, role), score in zip(records, scores, strict=True)
        ]

    def consolidate(
        self, session_id: str, items: list[WorkingItem], *, reason: EpisodeReason, now: float
    ) -> ConsolidationReport:
        """Turn working-memory items into one episode plus facts. Returns what was created or changed."""
        report = ConsolidationReport(session_id=session_id, reason=reason)
        sentences = self._sentences(items)
        if not sentences:
            return report
        report.sentences_in = len(sentences)
        report.raw_tokens = sum(self.counter(s.text) for s in sentences)
        if max(s.importance for s in sentences) < self.config.discard_importance:
            self.store.log_event(self.namespace, now, "segment_discarded", session=session_id, tokens=report.raw_tokens)
            return report

        raw_text = " ".join(s.text for s in sentences)
        episode_id = "e-" + content_hash(
            self.namespace, session_id, f"{sentences[0].ts:.3f}", f"{sentences[-1].ts:.3f}", raw_text
        )[:12]
        if self.store.episode_exists(episode_id) or self.store.is_tombstoned(self.namespace, item_id=episode_id):
            return report

        # Extract first: a sentence that yields a fact is kept verbatim, so every fact has a quotable source.
        sources = [s for s in sentences if s.role in FACT_SOURCES]
        extracted = self.extractor.extract([(s.text, s.ts) for s in sources])
        for sentence, candidates in zip(sources, extracted, strict=True):
            if candidates:
                sentence.salient = True
        result = self.compressor.compress(sentences)
        kept = [s for s, keep in zip(sentences, result.keep, strict=True) if keep]
        for sentence, candidates in zip(sources, extracted, strict=True):
            for candidate in candidates:
                change = self.semantic.apply(
                    candidate,
                    now=now,
                    importance=sentence.importance,
                    source_episode=episode_id,
                    source_session=session_id,
                )
                if change.fact_id and change.action in _LINKED and change.fact_id not in sentence.fact_ids:
                    sentence.fact_ids.append(change.fact_id)
                    sentence.spans[change.fact_id] = candidate.span or candidate.object
                report.facts.append(change)

        importance = max(s.importance for s in kept)
        episode = Episode(
            id=episode_id,
            namespace=self.namespace,
            session_id=session_id,
            reason=reason,
            start_ts=sentences[0].ts,
            end_ts=sentences[-1].ts,
            sentences=kept,
            abstract=result.abstract,
            importance=importance,
            raw_tokens=report.raw_tokens,
            tokens=0,
            created_at=now,
            last_accessed=now,
            expires_at=self.episode_expiry(now, importance),
        )
        episode.tokens = self.counter(episode.render())
        units = [s.text for s in kept] + ([result.abstract] if result.abstract else [])
        self.store.insert_episode(episode, self.embedder.embed(units))
        self.store.log_event(
            self.namespace,
            now,
            "episode_created",
            id=episode_id,
            reason=reason,
            raw_tokens=report.raw_tokens,
            tokens=episode.tokens,
            facts=len(report.facts),
        )
        report.episode_id = episode_id
        report.kept_tokens = episode.tokens
        report.sentences_kept = len(kept)
        return report
