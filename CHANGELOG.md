# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-01-28
### Added
- Google Colab setup instructions in `README.md`.
- `download_data.py` script for automated dataset retrieval using `gdown` (supports CLI arguments for folder links).
- `gdown` dependency in `requirements.txt`.

### Changed
- Updated `todolist.md` to reflect environment setup progress.
- Configured `train.py` to use ResNet50 FPN V2 by default.
- Reverted to default image sizes (removed forced 640px resize).
- Added Accuracy and Mean Dice metrics calculation per epoch.
