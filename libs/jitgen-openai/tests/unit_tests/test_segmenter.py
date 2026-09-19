import json

from jitgen.base import CodeSegment

from jitgen_openai import OpenAIToolCallSegmenter


def _feed_all(
    segmenter: OpenAIToolCallSegmenter, chunks: list[str]
) -> list[CodeSegment]:
    segments: list[CodeSegment] = []
    for chunk in chunks:
        segments.extend(segmenter.feed(chunk))
    return segments


def _text(segments: list[CodeSegment]) -> str:
    return "".join(segment.text for segment in segments)


def test_property_in_middle_of_object_split_into_arbitrary_chunks():
    payload = json.dumps({"language": "python", "code": "print(1)"})
    chunks = [payload[:5], payload[5:19], payload[19:31], payload[31:]]

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(segmenter, chunks)

    assert _text(segments) == "print(1)"
    assert segments[-1].end_of_block is True
    assert sum(s.end_of_block for s in segments) == 1


def test_property_is_only_field():
    payload = json.dumps({"code": "x = 1\ny = 2"})

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(segmenter, [payload])

    assert _text(segments) == "x = 1\ny = 2"
    assert segments[-1].end_of_block is True


def test_one_character_at_a_time_feeding():
    payload = json.dumps({"language": "python", "code": "print('hi')\nprint(2)"})
    expected = "print('hi')\nprint(2)"

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(segmenter, list(payload))

    assert _text(segments) == expected
    end_of_block_flags = [s.end_of_block for s in segments]
    assert end_of_block_flags.count(True) == 1
    assert end_of_block_flags[-1] is True


def test_escaped_quote_and_backslash_split_across_chunks():
    # Raw JSON text for {"code": "print(\"a\\b\")"}, decoding to print("a\b")
    payload = '{"code": "print(\\"a\\\\b\\")"}'
    expected = 'print("a\\b")'

    for split in range(len(payload)):
        segmenter = OpenAIToolCallSegmenter(property_name="code")
        segments = _feed_all(segmenter, [payload[:split], payload[split:]])
        assert _text(segments) == expected, f"failed at split={split}"


def test_newline_escape_split_across_chunk_boundary():
    payload = '{"code": "a\\nb"}'
    split = payload.index("\\n") + 1  # land right after the backslash

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(segmenter, [payload[:split], payload[split:]])

    assert _text(segments) == "a\nb"


def test_unicode_escape_split_across_several_chunks():
    payload = '{"code": "caf\\u00e9"}'
    idx = payload.index("\\u00e9")

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(
        segmenter,
        [
            payload[: idx + 1],  # ends right after '\'
            payload[idx + 1 : idx + 3],  # "u0"
            payload[idx + 3 : idx + 5],  # "0e"
            payload[idx + 5 :],  # "9..."
        ],
    )

    assert _text(segments) == "café"


def test_surrogate_pair_split_across_chunks():
    # U+1F600 GRINNING FACE, encoded as the surrogate pair 😀.
    payload = '{"code": "\\ud83d\\ude00"}'
    idx = payload.index("\\ud83d")

    for split in (idx, idx + 3, idx + 6, idx + 9, idx + 12):
        segmenter = OpenAIToolCallSegmenter(property_name="code")
        segments = _feed_all(segmenter, [payload[:split], payload[split:]])
        assert _text(segments) == "\U0001f600", f"failed at split={split}"


def test_target_key_text_inside_an_earlier_value_is_not_mistaken_for_a_key():
    payload = '{"note": "use the \\"code\\" key", "code": "1+1"}'

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(segmenter, [payload])

    assert _text(segments) == "1+1"
    assert segments[-1].end_of_block is True


def test_nested_struct_value_with_brace_like_string_content_is_skipped():
    payload = json.dumps({"meta": {"note": "a {b} c"}, "code": "42"})

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(segmenter, [payload])

    assert _text(segments) == "42"


def test_scalar_and_array_values_before_target_are_skipped():
    payload = json.dumps({"count": 3, "tags": ["a", "b"], "code": "1"})

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(segmenter, [payload])

    assert _text(segments) == "1"


def test_property_not_found_emits_nothing():
    payload = json.dumps({"language": "python", "notes": "n/a"})

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = _feed_all(segmenter, [payload])

    assert segments == []
    assert segmenter.inside_block is False


def test_inside_block_true_only_while_scanning_target_value():
    payload = json.dumps({"language": "python", "code": "1+1"})
    idx = payload.index('"code"')

    segmenter = OpenAIToolCallSegmenter(property_name="code")

    segmenter.feed(payload[:idx])
    assert segmenter.inside_block is False

    segmenter.feed(payload[idx : idx + len('"code": "')])
    assert segmenter.inside_block is True

    segmenter.feed(payload[idx + len('"code": "') :])
    assert segmenter.inside_block is False


def test_truncated_value_already_streamed_progressively_finalize_is_a_noop():
    # Every fully-decoded character is emitted the moment it's known — see
    # `feed()` — so by the time the stream ends there is nothing left for
    # `finalize()` to add; the truncated code already reached the caller.
    payload = '{"code": "print(1'  # never closes

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = segmenter.feed(payload)
    assert _text(segments) == "print(1"
    assert segmenter.inside_block is True

    assert segmenter.finalize() == []


def test_finalize_flushes_a_lone_unpaired_high_surrogate():
    # The escape is complete (`\ud83d` fully parsed) but the value never
    # closes, so the decoded high surrogate is still held back waiting for
    # a low surrogate that never arrives; `finalize()` flushes it best-effort.
    payload = '{"code": "abc\\ud83d'  # never closes

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = segmenter.feed(payload)
    assert _text(segments) == "abc"
    assert segmenter.inside_block is True

    segments = segmenter.finalize()

    assert len(segments) == 1
    assert segments[0].text == "\ud83d"
    assert segments[0].end_of_block is True
    assert segmenter.finalize() == []  # nothing left to flush twice


def test_finalize_returns_empty_when_not_inside_a_value():
    payload = json.dumps({"code": "1"})

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segmenter.feed(payload)

    assert segmenter.finalize() == []


def test_reset_allows_reuse_for_a_new_object():
    segmenter = OpenAIToolCallSegmenter(property_name="code")

    first = _feed_all(segmenter, [json.dumps({"code": "1"})])
    assert _text(first) == "1"

    segmenter.reset()

    second = _feed_all(segmenter, [json.dumps({"language": "python", "code": "2"})])
    assert _text(second) == "2"


def test_two_objects_concatenated_in_one_feed_call():
    payload = '{"code": "1"}{"code":"2"}'

    segmenter = OpenAIToolCallSegmenter(property_name="code")
    segments = segmenter.feed(payload)

    assert [s.text for s in segments] == ["1", "2"]
    assert all(s.end_of_block for s in segments)
