"""Shared pytest fixtures for ai-mem tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
from ai_mem.infrastructure.ranker_storage import RankerStorage


@pytest.fixture
def tmp_repo(tmp_path: Path) -> ChromaMemoryRepository:
    """A fresh ChromaMemoryRepository backed by a temp directory."""
    return ChromaMemoryRepository(tmp_path / "chroma")


@pytest.fixture
def track_access(tmp_repo) -> TrackAccessUseCase:
    return TrackAccessUseCase(tmp_repo)


@pytest.fixture
def tmp_storage(tmp_path: Path) -> RankerStorage:
    return RankerStorage(tmp_path / "rankers")
