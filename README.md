# Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)

BRISQUE is a no-reference image quality score.

A good place to know how BRISQUE works : [LearnOpenCV](https://learnopencv.com/image-quality-assessment-brisque/)


## Installation

```bash
# You will need to specify which version of OpenCV you intend to use with BRISQUE:
#   * opencv-python
#   * opencv-python-headless
#   * opencv-contrib-python
#   * opencv-contrib-python-headless
# You can do this with `pip install brisque[<YOUR CHOSEN VERSION HERE>]`, e.g.
pip install [opencv-python-headless]
```

## Usage

1. Trying to perform Image Quality Assessment on **local images**
```python
from brisque.brisque import BRISQUE

obj = BRISQUE(url=False)
obj.score("<Ndarray of the Image>")
```

2. Trying to perform Image Quality Assessment on **web images**
```python
from brisque.brisque import BRISQUE

obj = BRISQUE(url=True)
obj.score("<URL for the Image>")
```

### Example

#### Local Image

- Input
```python
from brisque.brisque import BRISQUE
import numpy as np
from PIL import Image

img_path = "brisque/tests/sample-image.jpg"
img = Image.open(img_path)
ndarray = np.asarray(img)

obj = BRISQUE(url=False)
obj.score(img=ndarray)
```
- Output
```
34.84883848208594
```

#### URL

- Input
```python
from brisque.brisque import BRISQUE

URL = "https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"

obj = BRISQUE(url=True)
obj.score(URL)
```
- Output
```
71.73427549219988
```
