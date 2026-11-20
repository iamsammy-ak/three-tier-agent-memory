from __future__ import annotations

import pytest
from pydantic import BaseModel

from tiered_memory import MemorySystem
from tiered_memory.compression import ExtractiveCompressor, LLMCompressor
from tiered_memory.extraction import LLMExtractor, RuleBasedExtractor
from tiered_memory.importance import HeuristicImportance, LLMImportance
from tiered_memory.llm import LLMError, LLMSettings, OpenRouterLLM, extract_json, get_llm, load_settings

from .test_extraction import ScriptedLLM


class Answer(BaseModel):
    value: int


def test_extract_json_handles_fences_prose_and_reasoning() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('<think>hmm {"no": 0}</think> Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("[1, 2]")


def test_complete_json_repairs_once() -> None:
    llm = ScriptedLLM(['{"value": "not a number"}', '{"value": 7}'])
    assert llm.complete_json([{"role": "user", "content": "q"}], Answer).value == 7
    assert llm.calls == 2


def test_complete_json_gives_up_after_the_repair() -> None:
    with pytest.raises(LLMError):
        ScriptedLLM(["nope", "still nope"]).complete_json([{"role": "user", "content": "q"}], Answer)


def test_no_key_means_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_settings().api_key is None
    assert get_llm() is None
    memory = MemorySystem(offline=False)
    assert memory.mode == "offline"
    assert isinstance(memory.extractor, RuleBasedExtractor)
    assert isinstance(memory.compressor, ExtractiveCompressor)
    assert isinstance(memory.importance, HeuristicImportance)
    with pytest.raises(LLMError):
        OpenRouterLLM(LLMSettings(api_key=None, models=("m",)))


def test_models_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODELS", "a/b:free, c/d:free ,")
    assert load_settings().models == ("a/b:free", "c/d:free")


def test_offline_flag_wins_even_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-for-test")
    assert get_llm(offline=True) is None


def test_an_injected_llm_drives_every_component() -> None:
    memory = MemorySystem(llm=ScriptedLLM([]))
    assert memory.mode == "llm"
    assert isinstance(memory.extractor, LLMExtractor)
    assert isinstance(memory.compressor, LLMCompressor)
    assert isinstance(memory.importance, LLMImportance)
    # The scripted model has no replies, so every component falls back to its offline stand-in.
    report = memory.remember("I live in Turin.")
    assert [f.object for f in memory.facts()] == ["Turin"] and report.episode_id
