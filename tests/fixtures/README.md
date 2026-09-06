# Frozen numerical baseline

The immutable price fixture is in `../../data/snapshot_2026-08-31/`. It includes the raw vendor response, complete-case adjusted-close CSV, and metadata with source, acquisition time, exact dates, cleaning and SHA-256 hashes. No network access is needed to load it or run tests.

`expected_summary.json` records results of the corrected analysis on that fixture. It is a regression baseline, not independent proof of the formulas. Independent tests separately reconstruct portfolio returns and prior-window quantiles, compare PCA with correlation-matrix eigenvalues, and compute likelihood ratios and p-values without calling the production test functions. Kupiec is also checked against the published MathWorks example.

Do not automatically regenerate this file in CI. A new data/config version or methodology change needs review, a new numerical baseline, and updated narrative. This fixture supersedes the earlier README's unverified 231/4,370 snapshot.
