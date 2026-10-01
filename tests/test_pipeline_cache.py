from indicmeet import pipeline


def _cached(cache, source, settings, calls):
    return pipeline._cached_json(
        "test-stage", cache, "fixture", False,
        lambda: calls.append(1) or {"runs": len(calls)},
        input_files=[source], settings=settings, cache_version=1,
    )


def test_stage_cache_invalidates_when_input_content_changes(tmp_path):
    source = tmp_path / "input.bin"
    cache = tmp_path / "cache.json"
    calls = []
    source.write_bytes(b"first")
    assert _cached(cache, source, {"model": "a"}, calls) == {"runs": 1}
    assert _cached(cache, source, {"model": "a"}, calls) == {"runs": 1}

    source.write_bytes(b"changed")
    assert _cached(cache, source, {"model": "a"}, calls) == {"runs": 2}


def test_stage_cache_invalidates_when_settings_change(tmp_path):
    source = tmp_path / "input.bin"
    cache = tmp_path / "cache.json"
    calls = []
    source.write_bytes(b"same input")
    assert _cached(cache, source, {"model": "a"}, calls) == {"runs": 1}
    assert _cached(cache, source, {"model": "b"}, calls) == {"runs": 2}


def test_stage_cache_version_can_force_manual_invalidation(tmp_path):
    source = tmp_path / "input.bin"
    cache = tmp_path / "cache.json"
    calls = []
    source.write_bytes(b"same input")
    assert _cached(cache, source, {"model": "a"}, calls) == {"runs": 1}
    pipeline._cached_json(
        "test-stage", cache, "fixture", False,
        lambda: calls.append(1) or {"runs": len(calls)},
        input_files=[source], settings={"model": "a"}, cache_version=2,
    )
    assert calls == [1, 1]
