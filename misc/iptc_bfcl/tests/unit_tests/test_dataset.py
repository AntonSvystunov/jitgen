import json

import httpx
import pytest
from conftest import FIXTURES, load_entry

from iptc_bfcl.dataset import (
    BFCL_REVISION,
    CATEGORIES,
    answer_path,
    ensure_downloaded,
    load_entries,
    question_path,
    select_entries,
)


def test_questions_are_joined_with_their_answers():
    [entry] = load_entries("parallel", FIXTURES)

    assert entry.id == "parallel_0"
    assert entry.category == "parallel"
    assert entry.system == ""
    assert "Taylor Swift" in entry.user
    assert [f["name"] for f in entry.functions] == ["spotify.play"]
    assert len(entry.ground_truth) == 2


def test_an_entry_system_message_is_kept_apart_from_the_request():
    entry = load_entry("live_simple_58-27-0")

    assert entry.system
    assert entry.system not in entry.user


def test_every_category_has_a_fixture():
    assert {
        category for category in CATEGORIES if load_entries(category, FIXTURES)
    } == set(CATEGORIES)


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _question(entry_id):
    return {
        "id": entry_id,
        "question": [[{"role": "user", "content": "hi"}]],
        "function": [],
    }


def test_a_mistyped_answer_id_falls_back_to_the_same_line(tmp_path):
    _write(tmp_path / question_path("simple"), [_question("a"), _question("b-79")])
    _write(
        tmp_path / answer_path("simple"),
        [{"id": "a", "ground_truth": [1]}, {"id": "b-279", "ground_truth": [2]}],
    )

    entries = load_entries("simple", tmp_path)

    assert [(e.id, e.ground_truth) for e in entries] == [("a", (1,)), ("b-79", (2,))]


def test_a_question_without_any_answer_is_rejected(tmp_path):
    _write(tmp_path / question_path("simple"), [_question("a"), _question("b")])
    # The line-0 answer belongs to question `b`, so `a` can't fall back to it.
    _write(tmp_path / answer_path("simple"), [{"id": "b", "ground_truth": []}])

    with pytest.raises(ValueError, match="a: no possible answer"):
        load_entries("simple", tmp_path)


def test_multi_turn_questions_are_rejected(tmp_path):
    question = _question("a")
    question["question"].append([{"role": "user", "content": "more"}])
    _write(tmp_path / question_path("simple"), [question])
    _write(tmp_path / answer_path("simple"), [{"id": "a", "ground_truth": []}])

    with pytest.raises(ValueError, match="expected one turn"):
        load_entries("simple", tmp_path)


def test_the_sample_is_seeded_and_ids_override_it():
    entries = [
        load_entry(entry_id) for entry_id in ("simple_0", "simple_89")
    ] * 3  # six entries of one category

    first = select_entries(entries, limit=2, sample_seed=7)
    second = select_entries(entries, limit=2, sample_seed=7)

    assert first == second
    assert len(first) == 2
    assert select_entries(entries, limit=0, sample_seed=7) == entries
    assert [
        e.id for e in select_entries(entries, limit=2, sample_seed=7, ids={"simple_89"})
    ] == ["simple_89"] * 3


def test_downloads_come_from_the_pinned_revision_once(tmp_path):
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=b'{"id": "x"}\n')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ensure_downloaded(["simple"], tmp_path, client=client)
    ensure_downloaded(["simple"], tmp_path, client=client)

    assert len(requested) == 2
    assert all(f"/resolve/{BFCL_REVISION}/" in url for url in requested)
    assert (tmp_path / answer_path("simple")).read_text() == '{"id": "x"}\n'
