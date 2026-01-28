# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-01-28
### Added
- Google Colab setup instructions in `README.md`.
- `download_data.py` script for automated dataset retrieval using `gdown` (supports CLI arguments for folder links).
- `gdown` dependency in `requirements.txt`.

### Changed
- Updated `todolist.md` to reflect environment setup progress.
- Configured `train.py` to use ResNet50 FPN V2 with LIGHTNING mode (256px, subsampling).
- Implemented robust label mapping to resolve CUDA device-side assert errors.
- Optimized `train.py` for maximum speed: added 256px resizing and training subsampling.
