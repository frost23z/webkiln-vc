import pytest
from webkiln_agents import scripts
from webkiln_agents.core.config import Settings
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS
from webkiln_agents.memory.service import SUPERSEDE_ABOVE
from webkiln_agents.scripts import PAIRS, Pair, PairResult, compare_pairs
from webkiln_agents.testing import FakeEmbeddings


async def test_every_pair_gets_a_cosine_and_the_size_of_the_embedding() -> None:
    results, vector = await compare_pairs(FakeEmbeddings())

    assert [r.pair for r in results] == list(PAIRS)
    assert all(0.0 <= r.similarity <= 1.0 + 1e-9 for r in results)
    assert len(vector) == EMBEDDING_DIMS


async def test_identical_texts_have_a_cosine_of_one_and_replace() -> None:
    [result], _ = await compare_pairs(FakeEmbeddings(), [Pair("same words", "same words", True)])

    assert result.similarity == pytest.approx(1.0)
    assert result.supersedes
    assert result.as_expected


async def test_texts_with_no_word_in_common_are_kept_apart() -> None:
    pair = Pair("green gold serif", "booking button header", same=False)
    [result], _ = await compare_pairs(FakeEmbeddings(), [pair])

    assert not result.supersedes
    assert result.as_expected


def test_a_pair_that_lands_on_the_wrong_side_of_the_threshold_is_flagged() -> None:
    pair = Pair("a", "b", same=True)

    assert not PairResult(pair, SUPERSEDE_ABOVE - 0.01).as_expected
    assert PairResult(pair, SUPERSEDE_ABOVE).as_expected


def test_the_pairs_cover_both_cases() -> None:
    assert any(p.same for p in PAIRS)
    assert any(not p.same for p in PAIRS)


def test_it_stops_with_a_clear_message_when_there_is_no_google_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scripts, "get_settings", lambda: Settings(_env_file=None))  # pyrefly: ignore[bad-argument-type]
    monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)

    with pytest.raises(SystemExit, match="GOOGLE_API_KEYS"):
        scripts.check_memory()
