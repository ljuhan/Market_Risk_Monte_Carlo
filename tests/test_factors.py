"""Deterministic tests for PCA factor analysis."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.risk_engine.factors import run_pca


ASSET_RETURNS = pd.DataFrame(
    {
        "SPY": [0.01, 0.03, -0.02, 0.04, -0.01, 0.02],
        "TLT": [-0.02, 0.01, 0.03, -0.01, 0.02, -0.03],
        "GLD": [0.00, 0.02, -0.01, 0.03, -0.02, 0.01],
    },
    index=pd.date_range("2020-01-01", periods=6, freq="D"),
)


def test_run_pca_explained_variance_matches_independent_pca() -> None:
    scaled = StandardScaler().fit_transform(ASSET_RETURNS)
    expected = PCA().fit(scaled).explained_variance_ratio_

    result = run_pca(ASSET_RETURNS)

    np.testing.assert_allclose(
        result["explained_variance_ratio"].to_numpy(), expected
    )


def test_run_pca_preserves_labels_and_shapes() -> None:
    result = run_pca(ASSET_RETURNS)
    explained = result["explained_variance_ratio"]
    loadings = result["loadings"]
    scores = result["scores"]

    assert list(explained.index) == ["PC1", "PC2", "PC3"]
    assert list(loadings.index) == list(ASSET_RETURNS.columns)
    assert list(loadings.columns) == ["PC1", "PC2", "PC3"]
    assert list(scores.index) == list(ASSET_RETURNS.index)
    assert list(scores.columns) == ["PC1", "PC2", "PC3"]
    assert loadings.shape == (3, 3)
    assert scores.shape == (6, 3)
    assert explained.shape == (3,)


def test_run_pca_scores_match_independent_transformation() -> None:
    scaled = StandardScaler().fit_transform(ASSET_RETURNS)
    expected_scores = PCA().fit_transform(scaled)

    result = run_pca(ASSET_RETURNS)

    np.testing.assert_allclose(result["scores"].to_numpy(), expected_scores)


def test_run_pca_is_reproducible_and_allows_component_sign_ambiguity() -> None:
    first = run_pca(ASSET_RETURNS)
    second = run_pca(ASSET_RETURNS)

    np.testing.assert_allclose(
        first["explained_variance_ratio"].to_numpy(),
        second["explained_variance_ratio"].to_numpy(),
    )
    np.testing.assert_allclose(
        np.abs(first["loadings"].to_numpy()),
        np.abs(second["loadings"].to_numpy()),
    )
    np.testing.assert_allclose(
        np.abs(first["scores"].to_numpy()),
        np.abs(second["scores"].to_numpy()),
    )


@pytest.mark.parametrize(
    "asset_returns",
    [
        pd.DataFrame(),
        pd.DataFrame({"SPY": [0.01]}),
        pd.DataFrame({"SPY": [0.01, np.nan], "TLT": [0.02, 0.03]}),
        pd.DataFrame({"SPY": [0.01, 0.02], "TLT": [0.02, "bad"]}),
        pd.DataFrame({"SPY": [0.01, 0.01], "TLT": [0.02, 0.03]}),
    ],
)
def test_run_pca_rejects_invalid_or_insufficient_inputs(
    asset_returns: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        run_pca(asset_returns)

