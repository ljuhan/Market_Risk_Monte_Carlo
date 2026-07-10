"""PCA factor analysis for asset return data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _validate_asset_returns(asset_returns: pd.DataFrame) -> None:
    """Validate a non-empty numeric return matrix for PCA."""
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame")
    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty")
    if asset_returns.shape[0] < 2:
        raise ValueError("asset_returns must contain at least two observations")
    if asset_returns.shape[1] == 0:
        raise ValueError("asset_returns must contain at least one asset")
    if not all(is_numeric_dtype(dtype) for dtype in asset_returns.dtypes):
        raise ValueError("asset_returns must contain only numeric columns")

    values = asset_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("asset_returns must contain only finite values")
    if np.any(values.var(axis=0) == 0.0):
        raise ValueError("asset_returns must not contain zero-variance columns")


def run_pca(
    asset_returns: pd.DataFrame,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Standardize returns and run the notebook's unconstrained PCA workflow."""
    _validate_asset_returns(asset_returns)

    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(asset_returns)
    pca = PCA()
    pca.fit(returns_scaled)

    component_labels = [f"PC{i + 1}" for i in range(pca.n_components_)]
    explained_variance_ratio = pd.Series(
        pca.explained_variance_ratio_,
        index=component_labels,
        name="explained_variance_ratio",
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=asset_returns.columns,
        columns=component_labels,
    )
    scores = pd.DataFrame(
        pca.transform(returns_scaled),
        index=asset_returns.index,
        columns=component_labels,
    )

    return {
        "explained_variance_ratio": explained_variance_ratio,
        "loadings": loadings,
        "scores": scores,
    }
