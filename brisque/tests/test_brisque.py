from brisque import BRISQUE
import numpy as np
from PIL import Image
import sys
from unittest.mock import patch
import importlib
import pytest


# def test_validate_url_score():
#     URL = "https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"
#     obj = BRISQUE(url=True)
#     assert type(round(obj.score(URL),1)) == type(round(11.11, 1))

def test_validate_local_image():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    ndarray = np.asarray(img)
    obj = BRISQUE(url=False)
    assert type(round(obj.score(ndarray),1)) == type(round(11.11,1))


class TestOpenCVImportError:
    """Test that missing OpenCV raises helpful error message."""

    def test_missing_opencv_raises_import_error(self):
        """Test that ImportError is raised when cv2 is not available."""
        # Store the original cv2 module
        original_cv2 = sys.modules.get('cv2')
        
        try:
            # Remove cv2 from available modules to simulate missing OpenCV
            sys.modules['cv2'] = None
            
            # Re-import the brisque module to trigger the ImportError check
            import brisque.brisque as brisque_module
            with pytest.raises(ImportError) as exc_info:
                importlib.reload(brisque_module)
            
            error_msg = str(exc_info.value)
            assert "OpenCV is required for BRISQUE" in error_msg
            assert "pip install brisque[opencv-python]" in error_msg
            assert "opencv-python-headless" in error_msg
        finally:
            # Restore the original cv2 module
            if original_cv2 is not None:
                sys.modules['cv2'] = original_cv2
            else:
                sys.modules.pop('cv2', None)
            
            # Reload the module to restore normal functionality
            importlib.reload(brisque_module)

    def test_error_message_contains_all_variants(self):
        """Test that error message mentions all OpenCV variants."""
        original_cv2 = sys.modules.get('cv2')
        
        try:
            sys.modules['cv2'] = None
            
            import brisque.brisque as brisque_module
            with pytest.raises(ImportError) as exc_info:
                importlib.reload(brisque_module)
            
            error_msg = str(exc_info.value)
            assert "opencv-python" in error_msg
            assert "opencv-python-headless" in error_msg
            assert "opencv-contrib-python" in error_msg
            assert "opencv-contrib-python-headless" in error_msg
        finally:
            if original_cv2 is not None:
                sys.modules['cv2'] = original_cv2
            else:
                sys.modules.pop('cv2', None)
            
            importlib.reload(brisque_module)

    def test_error_message_contains_install_instructions(self):
        """Test that error message contains clear installation instructions."""
        original_cv2 = sys.modules.get('cv2')
        
        try:
            sys.modules['cv2'] = None
            
            import brisque.brisque as brisque_module
            with pytest.raises(ImportError) as exc_info:
                importlib.reload(brisque_module)
            
            error_msg = str(exc_info.value)
            # Check for installation instructions
            assert "pip install brisque[" in error_msg
            assert "Or install OpenCV separately" in error_msg
        finally:
            if original_cv2 is not None:
                sys.modules['cv2'] = original_cv2
            else:
                sys.modules.pop('cv2', None)
            
            importlib.reload(brisque_module)
