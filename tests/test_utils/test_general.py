"""Tests for general utilities."""

import time
from pathlib import Path

import pytest
from pytest_check import check

from ai_rucksack.utils.general import get_parent_directory_path, measure_execution_time


def test_get_parent_directory_path() -> None:
    """Test get_parent_directory_path."""
    with check:
        # Check that the root test directory can be found from the current file's path
        tests_dir = get_parent_directory_path(Path(__file__), "tests")
        if tests_dir is None:
            raise FileNotFoundError("tests directory not found.")
        assert tests_dir.name == "tests"
        assert tests_dir.parent.name == "ai-rucksack"
    with check:
        # Check that a non-existent directory returns None
        assert get_parent_directory_path(Path(__file__), "a-path-to-nowhere") is None


def test_measure_execution_time() -> None:
    """Test that measure_execution_time returns (roughly) the correct time."""
    with measure_execution_time() as get_time:
        # Sleep for 1 second
        time.sleep(1.0)

    assert pytest.approx(get_time(), 0.1) == 1.0
