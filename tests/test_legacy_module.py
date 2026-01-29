import pytest
from examples.legacy_module import process_data, average


def test_process_data_basic():
    assert process_data([1, 2, 3]) == [2, 4, 6]


def test_process_data_with_none():
    assert process_data([1, None, 3]) == [2, 6]


def test_average_basic():
    assert average([2, 4, 6]) == 4


def test_average_empty():
    assert average([]) is None
