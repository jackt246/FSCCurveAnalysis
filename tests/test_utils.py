"""Unit tests for the pure preprocessing and classification helpers in fsc_analysis.utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

from fsc_analysis.utils import (
    FSCModels,
    anchor_curve,
    classify_fsc_curve,
    draw_typicality_bar,
    fetch_fsc_curve,
    find_crossing_point,
    normalize_curve,
    resample_curve,
    set_seeds,
)


class TestResampleCurve:
    def test_returns_requested_length(self):
        out = resample_curve(np.array([0.0, 0.5, 1.0]), target_length=50)
        assert out.shape == (50,)

    def test_default_length_is_100(self):
        assert resample_curve(np.array([1.0, 2.0])).shape == (100,)

    def test_preserves_endpoints(self):
        curve = np.array([0.2, 0.5, 0.9])
        out = resample_curve(curve, target_length=11)
        assert out[0] == pytest.approx(0.2)
        assert out[-1] == pytest.approx(0.9)

    def test_linear_curve_stays_linear(self):
        curve = np.linspace(0.0, 1.0, 5)
        out = resample_curve(curve, target_length=9)
        assert np.allclose(out, np.linspace(0.0, 1.0, 9))

    def test_constant_curve_stays_constant(self):
        out = resample_curve(np.array([0.7, 0.7, 0.7]), target_length=20)
        assert np.allclose(out, 0.7)

    def test_nans_are_ignored(self):
        out = resample_curve(np.array([0.0, np.nan, 1.0]), target_length=10)
        assert not np.isnan(out).any()
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)

    def test_single_valid_point_fills_constant(self):
        out = resample_curve(np.array([np.nan, 0.42, np.nan]), target_length=8)
        assert np.allclose(out, 0.42)

    def test_all_nan_returns_all_nan(self):
        out = resample_curve(np.array([np.nan, np.nan]), target_length=5)
        assert out.shape == (5,)
        assert np.isnan(out).all()


class TestFindCrossingPoint:
    def test_finds_fractional_crossing(self):
        # Drops from 0.2 (index 1) to 0.1 (index 2); crosses 0.143 between them.
        crossing = find_crossing_point(np.array([1.0, 0.2, 0.1, 0.0]))
        expected = 1 + (0.2 - 0.143) / (0.2 - 0.1)
        assert crossing == pytest.approx(expected)

    def test_returns_none_when_never_crosses_down(self):
        assert find_crossing_point(np.array([1.0, 0.9, 0.8])) is None

    def test_returns_none_when_always_below(self):
        assert find_crossing_point(np.array([0.0, 0.05, 0.1])) is None

    def test_only_first_crossing_is_returned(self):
        # Crosses below at index 2, recovers, crosses again at index 4.
        crossing = find_crossing_point(np.array([1.0, 1.0, 0.0, 1.0, 0.0]))
        assert 1 < crossing < 2

    def test_custom_threshold(self):
        crossing = find_crossing_point(np.array([1.0, 0.6, 0.4]), threshold=0.5)
        assert crossing == pytest.approx(1 + (0.6 - 0.5) / (0.6 - 0.4))


class TestAnchorCurve:
    def test_output_length(self):
        curve = np.linspace(1.0, 0.0, 11)
        assert anchor_curve(curve, output_length=100).shape == (100,)

    def test_crossing_lands_on_target_index(self):
        # A descending line definitely crosses 0.143; after anchoring the
        # 0.143 point must sit at target_idx.
        curve = np.linspace(1.0, 0.0, 11)
        anchored = anchor_curve(curve, target_idx=50, output_length=100)
        assert anchored[50] == pytest.approx(0.143, abs=1e-6)

    def test_falls_back_to_resample_without_crossing(self):
        # Never drops below 0.143 -> no crossing -> plain resample of given length.
        curve = np.array([1.0, 0.9, 0.8, 0.7])
        anchored = anchor_curve(curve, output_length=30)
        assert anchored.shape == (30,)
        assert anchored[0] == pytest.approx(1.0)


class _FakeEncoder:
    def __init__(self, embedding):
        self._embedding = np.asarray(embedding, dtype=float)

    def predict(self, x, verbose=0):
        return self._embedding.reshape(1, -1)


class _FakeKMeans:
    def __init__(self, cluster_id: int, centers):
        self._cluster_id = cluster_id
        self.cluster_centers_ = np.asarray(centers, dtype=float)

    def predict(self, x):
        return np.array([self._cluster_id])


def _make_models(cluster_id, embedding, centers, thresholds, data_min=0.0, data_max=1.0):
    return FSCModels(
        encoder=_FakeEncoder(embedding),
        kmeans=_FakeKMeans(cluster_id, centers),
        data_min=data_min,
        data_max=data_max,
        cluster_thresholds=pd.Series(thresholds),
    )


class TestNormalizeCurve:
    def test_scales_to_unit_range(self):
        out = normalize_curve(np.array([0.0, 5.0, 10.0]), data_min=0.0, data_max=10.0)
        assert out == pytest.approx([0.0, 0.5, 1.0])

    def test_constant_range_returns_input(self):
        curve = np.array([3.0, 3.0, 3.0])
        out = normalize_curve(curve, data_min=3.0, data_max=3.0)
        assert out == pytest.approx([3.0, 3.0, 3.0])


class TestClassifyFSCCurve:
    def test_on_centroid_is_fully_typical(self):
        models = _make_models(
            cluster_id=0, embedding=[0.0, 0.0], centers=[[0.0, 0.0], [5.0, 5.0]],
            thresholds={0: 1.0},
        )
        cluster_id, distance, typicality = classify_fsc_curve(np.zeros(100), models)
        assert cluster_id == 0
        assert distance == pytest.approx(0.0)
        assert typicality == pytest.approx(1.0)

    def test_half_threshold_distance_scores_half(self):
        models = _make_models(
            cluster_id=0, embedding=[1.0, 0.0], centers=[[0.0, 0.0]],
            thresholds={0: 2.0},
        )
        _, distance, typicality = classify_fsc_curve(np.zeros(100), models)
        assert distance == pytest.approx(1.0)
        assert typicality == pytest.approx(0.5)

    def test_beyond_threshold_clamps_to_zero(self):
        models = _make_models(
            cluster_id=0, embedding=[3.0, 0.0], centers=[[0.0, 0.0]],
            thresholds={0: 2.0},
        )
        _, _, typicality = classify_fsc_curve(np.zeros(100), models)
        assert typicality == pytest.approx(0.0)

    def test_return_types_are_native_python(self):
        models = _make_models(
            cluster_id=0, embedding=[0.0, 0.0], centers=[[0.0, 0.0]],
            thresholds={0: 1.0},
        )
        cluster_id, distance, typicality = classify_fsc_curve(np.zeros(100), models)
        assert isinstance(cluster_id, int)
        assert isinstance(distance, float)
        assert isinstance(typicality, float)

    def test_missing_threshold_defaults_to_zero(self):
        models = _make_models(
            cluster_id=5, embedding=[0.0, 0.0], centers=[[0.0, 0.0]] * 6,
            thresholds={0: 1.0},
        )
        cluster_id, _, typicality = classify_fsc_curve(np.zeros(100), models)
        assert cluster_id == 5
        assert typicality == 0.0


class TestSetSeeds:
    def test_numpy_reproducible(self):
        set_seeds(123)
        first = np.random.rand(5)
        set_seeds(123)
        second = np.random.rand(5)
        assert first == pytest.approx(second)


class _FakeResponse:
    def __init__(self, json_data=None, http_error: Exception | None = None):
        self._json_data = json_data
        self._http_error = http_error

    def raise_for_status(self):
        if self._http_error is not None:
            raise self._http_error

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


class TestFetchFSCCurve:
    def test_returns_curve_on_success(self, monkeypatch):
        payload = {"EMD-1234": {"fsc": {"curves": {"fsc": [1.0, 0.5, 0.1]}}}}
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(payload))
        assert fetch_fsc_curve("EMD-1234") == [1.0, 0.5, 0.1]

    def test_http_error_propagates(self, monkeypatch):
        err = requests.HTTPError("404")
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(http_error=err))
        with pytest.raises(requests.HTTPError):
            fetch_fsc_curve("EMD-0000")

    def test_unparseable_json_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(json_data=None))
        with pytest.raises(ValueError):
            fetch_fsc_curve("EMD-1234")

    def test_missing_fsc_data_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse({"EMD-1234": {}}))
        with pytest.raises(ValueError):
            fetch_fsc_curve("EMD-1234")


class TestDrawTypicalityBar:
    def test_prints_scale_labels(self, capsys):
        draw_typicality_bar(0.5)
        out = capsys.readouterr().out
        assert "Least Typical" in out
        assert "Most Typical" in out
        assert "O" in out

    def test_marker_position_tracks_percentile(self, capsys):
        draw_typicality_bar(0.0, width=10)
        low = capsys.readouterr().out.index("O")
        draw_typicality_bar(0.9, width=10)
        high = capsys.readouterr().out.index("O")
        assert high > low
