"""LoadRankerConfigUseCase: config parsing, missing file, malformed groups."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_mem.application.load_ranker_config import LoadRankerConfigUseCase


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "ranker_config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_missing_file_returns_empty(tmp_path: Path):
    uc = LoadRankerConfigUseCase(tmp_path / "nonexistent.json")
    assert uc.execute() == {}


def test_valid_config_maps_all_members(tmp_path: Path):
    path = _write(tmp_path, {
        "groups": [
            {
                "name": "work-microservices",
                "collections": ["repo.payment-svc", "repo.order-svc", "repo.gateway"],
            }
        ]
    })
    result = LoadRankerConfigUseCase(path).execute()

    assert set(result.keys()) == {"repo.payment-svc", "repo.order-svc", "repo.gateway"}
    for col, scope in result.items():
        assert scope.mode == "hybrid"
        assert scope.group == "work-microservices"
        assert set(scope.member_collections) == {"repo.payment-svc", "repo.order-svc", "repo.gateway"}


def test_multiple_groups_produce_separate_scopes(tmp_path: Path):
    path = _write(tmp_path, {
        "groups": [
            {"name": "grp-a", "collections": ["col.a1", "col.a2"]},
            {"name": "grp-b", "collections": ["col.b1"]},
        ]
    })
    result = LoadRankerConfigUseCase(path).execute()

    assert result["col.a1"].group == "grp-a"
    assert result["col.b1"].group == "grp-b"


def test_malformed_group_missing_name_is_skipped(tmp_path: Path, capsys):
    path = _write(tmp_path, {
        "groups": [
            {"collections": ["col.a"]},
            {"name": "good", "collections": ["col.b"]},
        ]
    })
    result = LoadRankerConfigUseCase(path).execute()

    assert "col.b" in result
    assert "col.a" not in result
    assert "skipping" in capsys.readouterr().err


def test_malformed_group_empty_collections_is_skipped(tmp_path: Path, capsys):
    path = _write(tmp_path, {
        "groups": [
            {"name": "bad", "collections": []},
            {"name": "good", "collections": ["col.x"]},
        ]
    })
    result = LoadRankerConfigUseCase(path).execute()

    assert "col.x" in result
    assert capsys.readouterr().err != ""


def test_duplicate_collection_last_group_wins(tmp_path: Path, capsys):
    path = _write(tmp_path, {
        "groups": [
            {"name": "first", "collections": ["shared"]},
            {"name": "second", "collections": ["shared"]},
        ]
    })
    result = LoadRankerConfigUseCase(path).execute()

    assert result["shared"].group == "second"
    assert "multiple groups" in capsys.readouterr().err


def test_invalid_json_returns_empty(tmp_path: Path, capsys):
    p = tmp_path / "ranker_config.json"
    p.write_text("{ not valid json }", encoding="utf-8")
    result = LoadRankerConfigUseCase(p).execute()

    assert result == {}
    assert capsys.readouterr().err != ""
