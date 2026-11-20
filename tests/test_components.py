from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tiered_memory.compression import ExtractiveCompressor, LLMCompressor, split_sentences, textrank
from tiered_memory.embeddings import FastEmbedEmbedder, HashingEmbedder, get_embedder
from tiered_memory.importance import HeuristicImportance, LLMImportance
from tiered_memory.models import EpisodeSentence, fmt_day
from tiered_memory.predicates import head_noun, is_single_valued, normalise_predicate, render_fact
from tiered_memory.tokens import ApproxTokenCounter, get_token_counter

from .test_extraction import ScriptedLLM

# ------------------------------------------------------------------ tokens


def test_approx_token_counter_basics() -> None:
    count = ApproxTokenCounter()
    assert count("") == 0
    assert count("Hello, world!") == 4
    assert count("internationalisation") > count("nation")
    assert count("2026") == 2


@settings(max_examples=50)
@given(st.text(max_size=200), st.text(max_size=200))
def test_token_count_is_additive_over_whitespace_concatenation(a: str, b: str) -> None:
    count = ApproxTokenCounter()
    assert count(a + " " + b) == count(a) + count(b)


def test_unknown_token_counter_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_token_counter("words")


# -------------------------------------------------------------- embeddings


def test_hashing_embedder_is_deterministic_normalised_and_lexical() -> None:
    emb = HashingEmbedder()
    vectors = emb.embed(["I live in Turin", "I live in Turin", "Turin is where I live", "quantum chromodynamics", ""])
    assert vectors.shape == (5, 384) and vectors.dtype == np.float32
    assert np.allclose(vectors[0], vectors[1])
    assert abs(np.linalg.norm(vectors[0]) - 1) < 1e-5 and np.linalg.norm(vectors[4]) == 0
    assert vectors[0] @ vectors[2] > vectors[0] @ vectors[3]
    assert np.allclose(HashingEmbedder().embed(["I live in Turin"]), vectors[:1])  # stable across instances


def test_get_embedder_rejects_unknown_kind() -> None:
    assert get_embedder("hashing").name == "hashing-384"
    with pytest.raises(ValueError):
        get_embedder("word2vec")


@pytest.mark.models
def test_fastembed_model_when_available() -> None:
    pytest.importorskip("fastembed")
    cache = os.environ.get("FASTEMBED_CACHE_PATH")
    if not cache or not any("bge-small-en-v1.5" in p.name for p in Path(cache).glob("*")):
        pytest.skip("BAAI/bge-small-en-v1.5 is not cached locally (set FASTEMBED_CACHE_PATH); no download in tests")
    emb = FastEmbedEmbedder()
    a, b, c = emb.embed(["Where does the user live?", "The user lives in Milan.", "The match ended 2-1."])
    assert emb.dim == 384 and a @ b > a @ c


# -------------------------------------------------------------- importance


def test_importance_rewards_what_should_be_remembered() -> None:
    h = HeuristicImportance()
    explicit = h.score("Remember that I'm allergic to peanuts.")
    deadline = h.score("The quarterly report deadline is 20 March.")
    chatter = h.score("The weather has been grey all week.")
    thanks = h.score("Ok, thanks!")
    assert explicit.value >= 0.9 and "explicit" in explicit.reasons
    assert deadline.value > chatter.value and "deadline" in deadline.reasons
    assert thanks.value < 0.2 and chatter.value < 0.2
    assert h.score("Remember that I'm allergic to peanuts.", "assistant").value < explicit.value
    assert h.score("Buy milk", "note").value >= 0.9


@settings(max_examples=50)
@given(st.text(max_size=300), st.sampled_from(["user", "assistant", "goal", "note"]))
def test_importance_is_always_in_unit_interval(text: str, role: str) -> None:
    value = HeuristicImportance().score(text, role).value
    assert 0.0 <= value <= 1.0


def test_llm_importance_rescales_and_falls_back() -> None:
    scores = LLMImportance(ScriptedLLM(['{"ratings": [10, 1]}'])).score_many([("a", "user"), ("b", "user")])
    assert [s.value for s in scores] == [1.0, 0.0]
    wrong_length = LLMImportance(ScriptedLLM(['{"ratings": [5]}'])).score_many([("Ok", "user"), ("Hi", "user")])
    assert all("llm" not in s.reasons for s in wrong_length)
    assert LLMImportance(ScriptedLLM([])).score("Remember my PIN").value >= 0.7


# ------------------------------------------------------------- compression


def sentences(*items: tuple[str, float]) -> list[EpisodeSentence]:
    return [EpisodeSentence(text=t, ts=i, role="user", importance=imp, salient=imp >= 0.45)
            for i, (t, imp) in enumerate(items)]


def test_split_sentences() -> None:
    assert split_sentences("Hi there. I live in Turin! Do you? 3 cats.") == [
        "Hi there.", "I live in Turin!", "Do you?", "3 cats."]
    assert split_sentences("  ") == []


def test_textrank_prefers_central_sentences() -> None:
    scores = textrank(["Turin has great coffee.", "Coffee in Turin is cheap.", "Turin coffee bars open early.",
                       "My cat sleeps."])
    assert abs(scores.sum() - 1) < 1e-9 and int(np.argmin(scores)) == 3
    assert textrank([]).size == 0 and textrank(["one"]).tolist() == [1.0]


def test_extractive_compressor_keeps_salient_and_respects_target() -> None:
    comp = ExtractiveCompressor(ratio=0.3, max_tokens=20, min_tokens=5)
    items = sentences(("Remember that I'm allergic to peanuts.", 0.95), ("We talked about films for a while.", 0.2),
                      ("Films are long these days and films are fun.", 0.2), ("Ok.", 0.0))
    result = comp.compress(items)
    assert result.keep[0] and not result.keep[3]
    kept_tokens = sum(
        ApproxTokenCounter()(s.text) for s, k in zip(items, result.keep, strict=True) if k and not s.salient
    )
    assert kept_tokens <= comp.target_tokens(sum(ApproxTokenCounter()(s.text) for s in items))


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.text(min_size=1, max_size=60), st.floats(0, 1)), min_size=1, max_size=12))
def test_compression_always_keeps_something_and_every_salient_sentence(items: list[tuple[str, float]]) -> None:
    batch = sentences(*items)
    result = ExtractiveCompressor().compress(batch)
    assert len(result.keep) == len(batch) and any(result.keep)
    assert all(k for s, k in zip(batch, result.keep, strict=True) if s.salient)


def test_llm_compressor_adds_abstract_and_falls_back() -> None:
    items = sentences(("I live in Turin.", 0.5), ("Nice weather.", 0.1))
    result = LLMCompressor(ScriptedLLM(['{"summary": "Sara lives in Turin."}'])).compress(items)
    assert result.abstract == "Sara lives in Turin." and result.keep == [True, False]
    assert LLMCompressor(ScriptedLLM([])).compress(items).abstract is None


# --------------------------------------------------------------- predicates


def test_predicate_normalisation_and_cardinality() -> None:
    assert normalise_predicate("Resides in") == "lives_in"
    assert normalise_predicate("favorite:Food") == "favourite:food"
    assert is_single_valued("lives_in") and is_single_valued("deadline:atlas") and not is_single_valued("likes")
    assert head_noun("window seats") == "seat" and head_noun("") == "general"
    assert render_fact("user", "manager", "Giulia") == "The user's manager is Giulia."
    assert render_fact("user", "job_title", "engineer") == "The user works as an engineer."
    assert render_fact("Marco", "owns", "a boat") == "Marco owns a boat."
    assert fmt_day(None) == "-"


def test_config_selects_the_token_counter() -> None:
    from tiered_memory import MemoryConfig, MemorySystem

    assert MemorySystem(offline=True).counter.name == "approx"
    with pytest.raises(ValueError):
        MemorySystem(offline=True, config=MemoryConfig(token_counter="words"))
