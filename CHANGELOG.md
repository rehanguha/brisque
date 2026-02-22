# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-22

### Changed
- **Flexible OpenCV dependencies** - OpenCV is no longer a hard dependency
  - Users can now choose their preferred OpenCV variant:
    - `pip install brisque[opencv-python]` - Desktop applications with GUI
    - `pip install brisque[opencv-python-headless]` - Servers/Docker (recommended)
    - `pip install brisque[opencv-contrib-python]` - Desktop with extra modules
    - `pip install brisque[opencv-contrib-python-headless]` - Servers with extra modules
  - Users with existing OpenCV installations can install brisque without conflicts
  - Helpful error message when OpenCV is missing with installation instructions

### Removed
- **requirements.txt** - Replaced by `requirements-dev.txt` for development only
  - Users should rely on `setup.py` for package installation
  - `requirements-dev.txt` contains only development dependencies (pytest, black, etc.)

### Fixed
- Improved build configuration in `pyproject.toml` (removed runtime dependencies from build-system)
- Updated minimum Python version requirement to 3.7+
- Updated classifiers to reflect current Python version support (3.7-3.12)

### Migration Guide
- **From v0.1.x**: No code changes required. Your existing code will work.
  - If upgrading: `pip install --upgrade brisque` (your OpenCV will be detected)
  - For new installs: `pip install brisque[opencv-python-headless]` (or your preferred variant)

## [0.1.0] - 2026-02-22

### Added
- **BRISQUETrainer class** - Complete training workflow for custom BRISQUE models
  - `add_image(image, score)` - Add single image with quality score
  - `add_dataset(image_dir, scores_file)` - Load images and scores from CSV
  - `train(svm_c, svm_gamma, svm_epsilon, kernel)` - Train SVM regression model
  - `save_model(svm_path, norm_path)` - Save trained model for later use
  - `evaluate(images, scores)` - Evaluate model performance (RMSE, PLCC, SROCC)
  - `clear()` - Reset trainer state
- **Custom model support** - BRISQUE class now accepts custom trained models
  - `model_path` parameter in BRISQUE constructor for custom model files
- **Comprehensive test suite** - 56 tests covering all functionality
- **Documentation** - Docstrings for all public methods and classes

### Fixed
- **scipy compatibility** - Added workaround for libsvm's use of deprecated `scipy.ndarray`
  - Works with scipy 1.8+ which removed `scipy.ndarray`
- **Feature extraction** - Fixed array conversion issues in trainer module
- **Image handling** - Better support for various image formats:
  - RGBA images (alpha channel removed)
  - Grayscale images
  - Single-channel images with 3D shape (H, W, 1)
  - Float images normalized to [0, 1]

### Changed
- Version bumped from 0.0.18 to 0.1.0
- Improved error messages for invalid inputs

### Dependencies
- No changes to dependencies - uses same libsvm, scipy, numpy, opencv-python, scikit-image

## [0.0.18] - Previous Release

### Note
- Original BRISQUE implementation with pre-trained model
- Basic quality scoring functionality