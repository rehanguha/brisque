from brisque import BRISQUE
import numpy as np
import pytest
from PIL import Image

def test_validate_url_score():
    URL = "https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"
    obj = BRISQUE(url=True)
    assert round(obj.score(URL),3) == round(71.73427549219397, 3)

def test_validate_local_image():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    ndarray = np.asarray(img)
    obj = BRISQUE(url=False)
    assert round(obj.score(ndarray),3) == round(35.427014669206955,3)
    



    
