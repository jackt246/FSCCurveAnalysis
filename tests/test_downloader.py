"""Unit tests for the standalone helpers in fsc_analysis.downloader."""

from __future__ import annotations

import numpy as np

from fsc_analysis.downloader import is_valid_number_array


class TestIsValidNumberArray:
    def test_none_is_invalid(self):
        assert is_valid_number_array(None) is False

    def test_plain_numeric_list_is_valid(self):
        assert is_valid_number_array([1.0, 2.0, 3.0]) is True

    def test_numpy_array_is_valid(self):
        assert is_valid_number_array(np.array([0.1, 0.2])) is True

    def test_scalar_number_is_valid(self):
        assert is_valid_number_array(4.2) is True

    def test_infinity_is_invalid(self):
        assert is_valid_number_array([1.0, np.inf]) is False

    def test_nan_is_invalid(self):
        assert is_valid_number_array([1.0, np.nan]) is False

    def test_non_numeric_strings_are_invalid(self):
        assert is_valid_number_array(["a", "b"]) is False

    def test_empty_array_is_valid(self):
        # np.all over an empty array is vacuously True.
        assert is_valid_number_array([]) is True
