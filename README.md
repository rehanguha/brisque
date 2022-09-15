# Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) 

BRISQUE is a no-reference image quality score.

A good place to know how BRISQUE works : [LearnOpenCV](https://learnopencv.com/image-quality-assessment-brisque/)


## Installation

```bash
pip install brisque
```

## Usage

1. Trying to perform Image Quality Assessment on **local images** 
```python
from brisque import BRISQUE

obj = BRISQUE("<Location of the Image>", url=False)
obj.score()
```

2. Trying to perform Image Quality Assessment on **web images** 
```python
from brisque import BRISQUE

obj = BRISQUE("<URL for the Image>", url=True)
obj.score()
```

### Example

- Input
```python
from brisque import BRISQUE

obj = BRISQUE("https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png", 
        url=True)
obj.score()
```
- Output
```
74.41910327611319
```
