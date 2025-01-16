# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - YYYY-MM-DD
### Added
- Core functionality for `RDDCounts` class, including RDD counts generation and filtering.
- PCA analysis function: `perform_pca_RDD_counts` with optional CLR transformation.
- Multiple visualization methods via `Matplotlib` and `Plotly` backends.
- Utility functions for data transformation (`RDD_counts_to_wide`, `calculate_proportions`).
- Unit tests for major functionalities.
- Automated CI with GitHub Actions for testing and linting.
- `CONTRIBUTING.md` with detailed contribution guidelines.
- Modern packaging files (`setup.cfg` and `pyproject.toml`).

---

## Guidelines for Updating the Changelog

- **Follow Semantic Versioning**:
  - `MAJOR` version for breaking changes.
  - `MINOR` version for backward-compatible new features.
  - `PATCH` version for bug fixes.
- **Structure**:
  - Use categories like `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, and `Security`.
  - Keep entries concise and clear.