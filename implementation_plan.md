# Implementation Plan

[Overview]
Make OpenCV dependency optional and flexible, allowing users to choose their preferred OpenCV variant while preserving all v0.1.0 features and maintaining backward compatibility.

This change addresses the need for flexibility in OpenCV variants (opencv-python, opencv-python-headless, opencv-contrib-python, etc.) which are mutually exclusive packages. Users in server/headless environments need opencv-python-headless, while desktop users may prefer opencv-python or opencv-contrib-python. The current hard-coded dependency on opencv-python causes conflicts when users have a different OpenCV variant installed. This implementation will make OpenCV an optional extra with clear error messages, following modern Python packaging best practices.

[Types]
No type system changes required.

All existing type hints and interfaces remain unchanged. The BRISQUE and BRISQUETrainer classes maintain their existing signatures.

[Files]
Four files will be modified, one created, and one deleted.

### Modified Files:

**1. `setup.py`** - Update dependency configuration
- Remove `opencv-python` from `install_requires`
- Add `EXTRAS_REQUIRE` dictionary with all OpenCV variants
- Pass `extras_require` to `setuptools.setup()`

**2. `brisque/brisque.py`** - Add graceful import handling
- Wrap `import cv2` in try/except block
- Provide helpful error message with installation instructions if cv2 is missing

**3. `brisque/__init__.py`** - Version bump
- Change `__version__` from `"0.1.0"` to `"0.2.0"`

**4. `pyproject.toml`** - Update build system
- Remove `opencv-python` from `requires` list (build system should not have runtime deps)

**5. `README.md`** - Update installation documentation
- Add prominent v0.2.0 release notes section at top of file with detailed change description
- Document the OpenCV dependency changes clearly
- Update installation section to explain OpenCV extras
- Add examples for each installation scenario
- Add migration guide for existing users

**README v0.2.0 Section Content:**
```markdown
## 🆕 What's New in v0.2.0

### Flexible OpenCV Dependencies

Starting with v0.2.0, BRISQUE no longer forces a specific OpenCV variant. This allows you to use whichever OpenCV package best suits your environment.

| OpenCV Variant | Best For |
|----------------|----------|
| `opencv-python` | Desktop applications with GUI support |
| `opencv-python-headless` | Servers, Docker, CI/CD (no GUI) |
| `opencv-contrib-python` | Desktop with extra OpenCV modules |
| `opencv-contrib-python-headless` | Servers with extra modules |

**Why this change?**
- OpenCV variants are mutually exclusive - installing one removes others
- Previous versions required `opencv-python`, causing conflicts for users with different variants
- Server/headless environments don't need GUI dependencies from `opencv-python`

### Installation Options

**Option 1: Install with OpenCV (Recommended for new users)**
```bash
pip install brisque[opencv-python]              # Desktop applications
pip install brisque[opencv-python-headless]     # Servers / Docker (recommended)
pip install brisque[opencv-contrib-python]      # With extra OpenCV modules
pip install brisque[opencv-contrib-python-headless]  # Servers with extra modules
```

**Option 2: Use existing OpenCV installation**
```bash
# If you already have OpenCV installed
pip install opencv-python-headless   # or any variant you prefer
pip install brisque                  # will use your existing OpenCV
```

**Option 3: Install OpenCV separately (for Docker/corporate environments)**
```bash
# In your Dockerfile or controlled environment
pip install opencv-python-headless
pip install brisque  # Uses the pre-installed OpenCV
```

> ⚠️ **Note**: BRISQUE requires OpenCV to function. If you install brisque without OpenCV, you'll get a helpful error message with installation instructions.

### Migration from v0.1.x

If you're upgrading from v0.1.x or earlier:

**Scenario A: Standard desktop usage (no changes needed)**
```bash
# Your existing OpenCV will be detected and used
pip install --upgrade brisque
```

**Scenario B: Clean install or server environment**
```bash
# Uninstall old version
pip uninstall brisque

# Install with your preferred OpenCV variant
pip install brisque[opencv-python-headless]  # Recommended for servers
```

**Scenario C: Docker/CI environments**
```dockerfile
# In your Dockerfile
RUN pip install opencv-python-headless
RUN pip install brisque
```

**What changed:**
- v0.1.x: `pip install brisque` automatically installed `opencv-python`
- v0.2.0: OpenCV is no longer auto-installed - you choose your variant

**Code compatibility:** No changes needed! Your existing code continues to work:
```python
from brisque import BRISQUE  # Still works exactly the same
obj = BRISQUE(url=False)
score = obj.score(image)
```
```

**6. `CHANGELOG.md`** - Document changes
- Add v0.2.0 section documenting the dependency changes

### Created Files:

**1. `requirements-dev.txt`** - Development dependencies
- Include only development/testing dependencies (pytest, pytest-cov, etc.)
- Remove transitive dependencies
- Use flexible version specifiers

### Deleted Files:

**1. `requirements.txt`** - Remove in favor of setup.py
- Replaced by requirements-dev.txt for development
- Users will rely on setup.py for installation

[Functions]
No new functions required; one function modification.

### Modified Functions:

**1. Module-level imports in `brisque/brisque.py`**
- Current: `import cv2` at top of file
- New: Wrap in try/except with helpful error message
```python
try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is required for BRISQUE. Install it with one of:\n"
        "  pip install brisque[opencv-python]\n"
        "  pip install brisque[opencv-python-headless]\n"
        "  pip install brisque[opencv-contrib-python]\n"
        "  pip install brisque[opencv-contrib-python-headless]"
    ) from e
```

[Classes]
No class modifications required.

The BRISQUE and BRISQUETrainer classes remain unchanged. All existing functionality is preserved.

[Dependencies]
Dependency structure changes to make OpenCV optional.

### Current Dependencies (install_requires):
- numpy
- scikit-image
- scipy
- opencv-python (hard requirement)
- libsvm-official
- requests

### New Dependencies Structure:

**install_requires:**
- numpy>=1.20
- scikit-image>=0.19
- scipy>=1.7
- libsvm-official>=3.32
- requests>=2.25

**extras_require:**
```python
{
    "opencv-python": ["opencv-python>=4.5"],
    "opencv-python-headless": ["opencv-python-headless>=4.5"],
    "opencv-contrib-python": ["opencv-contrib-python>=4.5"],
    "opencv-contrib-python-headless": ["opencv-contrib-python-headless>=4.5"],
}
```

### Development Dependencies (requirements-dev.txt):
- pytest>=8.0
- pytest-cov>=4.0
- pillow>=10.0

[Testing]
Testing approach ensures backward compatibility and new functionality.

### Test Requirements:
1. Existing test suite must pass without modification
2. Test that ImportError is raised with helpful message when cv2 is missing
3. Test that installation with each OpenCV variant works

### Validation Strategy:
1. Run existing test suite: `pytest brisque/tests/ -v`
2. Verify installation from setup.py: `pip install -e .[opencv-python-headless]`
3. Test import error handling by mocking missing cv2

### Test File Modifications:
No test file modifications required. The existing test suite will continue to work as long as an OpenCV variant is installed in the test environment.

[Implementation Order]
Implementation sequence to minimize conflicts and ensure successful integration.

1. **Update `brisque/brisque.py`** - Add graceful cv2 import handling with error message
2. **Update `brisque/__init__.py`** - Bump version to 0.2.0
3. **Update `setup.py`** - Reconfigure dependencies with extras_require
4. **Update `pyproject.toml`** - Remove opencv-python from build-system requires
5. **Create `requirements-dev.txt`** - Add development dependencies file
6. **Delete `requirements.txt`** - Remove in favor of setup.py and requirements-dev.txt
7. **Update `README.md`** - Update installation instructions with OpenCV variant examples
8. **Update `CHANGELOG.md`** - Document v0.2.0 changes
9. **Run tests** - Verify all existing tests pass with new configuration