# Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11104461.svg)](https://doi.org/10.5281/zenodo.11104461)


BRISQUE is a no-reference image quality score.

A good place to know how BRISQUE works : [LearnOpenCV](https://learnopencv.com/image-quality-assessment-brisque/)


## Installation

```bash
pip install brisque
```

## Usage

### 1. Image Quality Assessment on Local Images

```python
from brisque import BRISQUE

obj = BRISQUE(url=False)
obj.score("<Ndarray of the Image>")
```

### 2. Image Quality Assessment on Web Images

```python
from brisque import BRISQUE

obj = BRISQUE(url=True)
obj.score("<URL for the Image>")
```

### 3. Using Custom Trained Models

```python
from brisque import BRISQUE

obj = BRISQUE(url=False, model_path={
    "svm": "path/to/custom_svm.txt",
    "normalize": "path/to/custom_normalize.pickle"
})
obj.score(image)
```

## Examples

### Local Image

```python
from brisque import BRISQUE
import numpy as np
from PIL import Image

img_path = "brisque/tests/sample-image.jpg"
img = Image.open(img_path)
ndarray = np.asarray(img)

obj = BRISQUE(url=False)
obj.score(img=ndarray)
```

Output:
```
34.84883848208594
```

### URL

```python
from brisque import BRISQUE

URL = "https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"

obj = BRISQUE(url=True)
obj.score(URL)
```

Output:
```
71.73427549219988
```

---

## Training Custom Models

The `BRISQUETrainer` class allows you to train custom BRISQUE models using your own image quality datasets. This is useful when you need a model tailored to specific image types or distortion patterns.

### When to Train a Custom Model

- Domain-specific images (medical, satellite, underwater, etc.)
- Specific distortion types not well-represented in the default model
- Custom quality scales or scoring methodologies
- Research purposes requiring reproducible quality assessment

---

## Step-by-Step Guide: Training a Custom BRISQUE Model

### Step 1: Install the Package

```bash
pip install brisque
```

### Step 2: Prepare Your Dataset

Organize your images and quality scores. You'll need:
- A directory containing your images
- A CSV file with image filenames and their quality scores

**Dataset structure:**
```
my_dataset/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   ├── img_003.jpg
│   └── ...
└── scores.csv
```

**CSV format (scores.csv):**
```csv
image,score
img_001.jpg,23.5
img_002.jpg,45.2
img_003.jpg,31.8
img_004.jpg,12.1
...
```

> **Note on Quality Scores**: Lower scores should indicate better quality. Use consistent scoring methodology across all images. Common scales include:
> - MOS (Mean Opinion Score): 1-5 scale
> - DMOS (Differential MOS): 0-100 scale
> - Custom scales specific to your application

### Step 3: Create a Training Script

Create a file called `train_brisque.py`:

```python
from brisque import BRISQUETrainer, BRISQUE
import numpy as np

# Initialize the trainer
trainer = BRISQUETrainer()

# Load your dataset
trainer.add_dataset(
    image_dir="my_dataset/images",
    scores_file="my_dataset/scores.csv",
    image_column="image",    # column name for image filenames
    score_column="score"     # column name for quality scores
)

print(f"Loaded {len(trainer.features)} images")

# Train the model with custom SVM parameters
trainer.train(
    svm_c=10.0,           # Regularization (higher = less tolerance for errors)
    svm_gamma=0.1,        # RBF kernel coefficient
    svm_epsilon=0.1,      # Epsilon-tube for SVR
    kernel="rbf"          # Kernel type: "rbf", "linear", "poly", or "sigmoid"
)

# Save the trained model
trainer.save_model(
    svm_path="custom_model/svm.txt",
    norm_path="custom_model/normalize.pickle"
)

print("Model saved successfully!")
```

### Step 4: Run the Training

```bash
python train_brisque.py
```

### Step 5: Evaluate the Model (Optional)

Evaluate your model on a separate test set:

```python
from brisque import BRISQUETrainer
import numpy as np
from PIL import Image

# After training, evaluate on test data
test_images = [np.array(Image.open(f"test_images/img_{i}.jpg")) for i in range(10)]
test_scores = [25.3, 42.1, 18.9, ...]  # ground truth scores

metrics = trainer.evaluate(test_images, test_scores)

print(f"RMSE:  {metrics['rmse']:.4f}")   # Root Mean Square Error (lower is better)
print(f"PLCC:  {metrics['plcc']:.4f}")   # Pearson correlation (closer to 1 is better)
print(f"SROCC: {metrics['srocc']:.4f}")  # Spearman correlation (closer to 1 is better)
```

### Step 6: Use Your Trained Model

```python
from brisque import BRISQUE
import numpy as np
from PIL import Image

# Load your custom model
obj = BRISQUE(url=False, model_path={
    "svm": "custom_model/svm.txt",
    "normalize": "custom_model/normalize.pickle"
})

# Score a new image
image = np.array(Image.open("new_image.jpg"))
quality_score = obj.score(image)

print(f"Quality Score: {quality_score}")
```

### Step 7: Integrate into Your Pipeline

```python
from brisque import BRISQUE
import numpy as np

# Initialize once at application startup
brisque = BRISQUE(url=False, model_path={
    "svm": "custom_model/svm.txt",
    "normalize": "custom_model/normalize.pickle"
})

# Use throughout your application
def assess_image_quality(image_array: np.ndarray) -> float:
    """Assess image quality using custom BRISQUE model."""
    return brisque.score(image_array)

# Example usage
for image_path in image_paths:
    img = np.array(Image.open(image_path))
    score = assess_image_quality(img)
    print(f"{image_path}: {score:.2f}")
```

---

## Alternative: Training with Individual Images

If you prefer to add images individually instead of from a CSV:

```python
from brisque import BRISQUETrainer
import numpy as np
from PIL import Image

trainer = BRISQUETrainer()

# Add images one by one
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
scores = [23.5, 45.2, 31.8]

for img_path, score in zip(images, scores):
    img = np.array(Image.open(img_path))
    trainer.add_image(img, score)

# Train and save
trainer.train()
trainer.save_model("custom_svm.txt", "custom_normalize.pickle")
```

---

## Training Tips

### Data Requirements
| Dataset Size | Expected Performance |
|--------------|---------------------|
| 2-10 images | Minimal - for testing only |
| 50-100 images | Basic functionality |
| 200+ images | Good performance |
| 500+ images | Optimal performance |

### SVM Parameter Tuning

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `svm_c` | Regularization strength | 0.1 - 100 |
| `svm_gamma` | RBF kernel width | 0.001 - 1 |
| `svm_epsilon` | Tolerance for errors | 0.01 - 0.5 |
| `kernel` | Kernel function | "rbf" (default), "linear" |

### Best Practices

1. **Normalize scores**: Ensure all scores use the same scale
2. **Balance quality levels**: Include images across the quality spectrum
3. **Use representative images**: Training images should match your target domain
4. **Validate on held-out data**: Keep 20% of data for testing
5. **Experiment with parameters**: Try different `svm_c` and `svm_gamma` values

---

## API Reference

### BRISQUE Class

```python
BRISQUE(url=False, model_path=None)
```

**Parameters:**
- `url` (bool): If True, accept URLs in `score()`. If False, accept numpy arrays. Default: False.
- `model_path` (dict, optional): Custom model paths with keys:
  - `"svm"`: Path to SVM model file
  - `"normalize"`: Path to normalization parameters file

**Methods:**
- `score(img)`: Compute quality score for an image

### BRISQUETrainer Class

```python
BRISQUETrainer()
```

**Methods:**
- `add_image(image, score)`: Add single image with quality score
- `add_dataset(image_dir, scores_file, image_column="image", score_column="score")`: Load from directory and CSV
- `train(svm_c=1.0, svm_gamma=0.1, svm_epsilon=0.1, kernel="rbf")`: Train the model
- `save_model(svm_path, norm_path)`: Save trained model
- `evaluate(images, scores)`: Evaluate model, returns dict with RMSE, PLCC, SROCC
- `clear()`: Clear all data and model

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest brisque/tests/ -v

# Run specific test file
pytest brisque/tests/test_trainer.py -v

# Run with coverage
pytest brisque/tests/ -v --cov=brisque
```

---

## How BRISQUE Works

BRISQUE extracts 36 features from each image:
- 18 features from the original scale (MSCN coefficients)
- 18 features from the half-scale (downsampled image)

Features are based on:
- Mean Subtracted Contrast Normalized (MSCN) coefficients
- Pairwise products (horizontal, vertical, diagonal)
- Asymmetric Generalized Gaussian Distribution fitting

The SVM regression model maps these features to quality scores.

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{brisque2024,
  author = {Guha, Rehan},
  title = {BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator},
  year = {2024},
  doi = {10.5281/zenodo.11104461}
}
```

## License

Apache-2.0