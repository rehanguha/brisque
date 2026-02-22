"""
Comprehensive pytest tests for BRISQUETrainer.

Tests cover:
- Initialization
- Feature extraction
- Training workflow
- Model saving/loading
- Evaluation metrics
- Error handling
- Edge cases
"""

import os
import tempfile
import pickle
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from brisque import BRISQUE, BRISQUETrainer


# ==================== Fixtures ====================

@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    np.random.seed(42)
    # Create a 64x64 RGB image with random values
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def sample_rgba_image():
    """Create a sample RGBA image with alpha channel."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)


@pytest.fixture
def sample_images_batch():
    """Create a batch of sample images for training."""
    np.random.seed(42)
    images = []
    for _ in range(5):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        images.append(img)
    return images


@pytest.fixture
def sample_scores():
    """Sample quality scores for testing."""
    return [30.0, 45.0, 25.0, 50.0, 35.0]


@pytest.fixture
def trainer():
    """Create a fresh trainer instance."""
    return BRISQUETrainer()


@pytest.fixture
def trained_trainer(trainer, sample_images_batch, sample_scores):
    """Create a trained trainer instance."""
    for img, score in zip(sample_images_batch, sample_scores):
        trainer.add_image(img, score)
    trainer.train()
    return trainer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ==================== Initialization Tests ====================

class TestBRISQUETrainerInit:
    """Test BRISQUETrainer initialization."""
    
    def test_init_creates_empty_features_list(self, trainer):
        """Test that initialization creates an empty features list."""
        assert trainer.features == []
        
    def test_init_creates_empty_scores_list(self, trainer):
        """Test that initialization creates an empty scores list."""
        assert trainer.scores == []
        
    def test_init_model_is_none(self, trainer):
        """Test that model is None after initialization."""
        assert trainer._model is None
        
    def test_init_scaler_params_is_none(self, trainer):
        """Test that scaler params is None after initialization."""
        assert trainer._scaler_params is None


# ==================== Feature Extraction Tests ====================

class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_extract_features_returns_array(self, trainer, sample_image):
        """Test that feature extraction returns a numpy array."""
        features = trainer._extract_features(sample_image)
        assert isinstance(features, np.ndarray)
        
    def test_extract_features_correct_shape(self, trainer, sample_image):
        """Test that extracted features have correct shape (36,)."""
        features = trainer._extract_features(sample_image)
        assert features.shape == (36,)
        
    def test_extract_features_grayscale(self, trainer, sample_grayscale_image):
        """Test feature extraction on grayscale image."""
        features = trainer._extract_features(sample_grayscale_image)
        assert features.shape == (36,)
        
    def test_extract_features_rgba(self, trainer, sample_rgba_image):
        """Test that alpha channel is handled correctly."""
        features = trainer._extract_features(sample_rgba_image)
        assert features.shape == (36,)
        
    def test_extract_features_normalized_image(self, trainer):
        """Test feature extraction on already normalized image [0, 1]."""
        np.random.seed(42)
        image = np.random.rand(64, 64, 3)
        features = trainer._extract_features(image)
        assert features.shape == (36,)
        
    def test_extract_features_float_image(self, trainer):
        """Test feature extraction on float image [0, 255]."""
        np.random.seed(42)
        image = np.random.rand(64, 64, 3) * 255
        features = trainer._extract_features(image)
        assert features.shape == (36,)
        
    def test_extract_features_returns_float64(self, trainer, sample_image):
        """Test that features are float64 type."""
        features = trainer._extract_features(sample_image)
        assert features.dtype == np.float64


# ==================== Add Image Tests ====================

class TestAddImage:
    """Test adding images to trainer."""
    
    def test_add_image_increases_count(self, trainer, sample_image):
        """Test that adding image increases the count."""
        initial_count = len(trainer.features)
        trainer.add_image(sample_image, score=30.0)
        assert len(trainer.features) == initial_count + 1
        assert len(trainer.scores) == initial_count + 1
        
    def test_add_image_stores_correct_score(self, trainer, sample_image):
        """Test that score is stored correctly."""
        trainer.add_image(sample_image, score=42.5)
        assert trainer.scores[-1] == 42.5
        
    def test_add_image_stores_correct_features(self, trainer, sample_image):
        """Test that features are stored correctly."""
        features = trainer._extract_features(sample_image)
        trainer.add_image(sample_image, score=30.0)
        np.testing.assert_array_almost_equal(trainer.features[-1], features)
        
    def test_add_multiple_images(self, trainer, sample_images_batch, sample_scores):
        """Test adding multiple images."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        assert len(trainer.features) == len(sample_images_batch)
        assert len(trainer.scores) == len(sample_scores)
        
    def test_add_image_invalid_type_raises(self, trainer):
        """Test that invalid image type raises ValueError."""
        with pytest.raises(ValueError, match="Image must be a numpy array"):
            trainer.add_image("not_an_array", score=30.0)
            
    def test_add_image_invalid_score_raises(self, trainer, sample_image):
        """Test that invalid score type raises ValueError."""
        with pytest.raises(ValueError, match="Score must be a numeric value"):
            trainer.add_image(sample_image, score="not_a_number")
            
    def test_add_image_integer_score_works(self, trainer, sample_image):
        """Test that integer scores work correctly."""
        trainer.add_image(sample_image, score=30)  # Integer
        assert trainer.scores[-1] == 30.0  # Should be converted to float


# ==================== Training Tests ====================

class TestTraining:
    """Test training functionality."""
    
    def test_train_with_sufficient_data(self, trainer, sample_images_batch, sample_scores):
        """Test training with sufficient data."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        trainer.train()
        assert trainer._model is not None
        assert trainer._scaler_params is not None
        
    def test_train_with_insufficient_data_raises(self, trainer, sample_image):
        """Test that training with insufficient data raises ValueError."""
        trainer.add_image(sample_image, score=30.0)
        with pytest.raises(ValueError, match="Need at least 2 images"):
            trainer.train()
            
    def test_train_empty_data_raises(self, trainer):
        """Test that training with no data raises ValueError."""
        with pytest.raises(ValueError, match="Need at least 2 images"):
            trainer.train()
            
    def test_train_creates_scaler_params(self, trainer, sample_images_batch, sample_scores):
        """Test that training creates scaler params."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        trainer.train()
        assert 'min_' in trainer._scaler_params
        assert 'max_' in trainer._scaler_params
        
    def test_train_scaler_params_correct_length(self, trainer, sample_images_batch, sample_scores):
        """Test that scaler params have correct length (36 features)."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        trainer.train()
        assert len(trainer._scaler_params['min_']) == 36
        assert len(trainer._scaler_params['max_']) == 36
        
    def test_train_with_custom_svm_params(self, trainer, sample_images_batch, sample_scores):
        """Test training with custom SVM parameters."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        trainer.train(svm_c=10.0, svm_gamma=0.01, kernel='rbf')
        assert trainer._model is not None
        
    def test_train_linear_kernel(self, trainer, sample_images_batch, sample_scores):
        """Test training with linear kernel."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        trainer.train(kernel='linear')
        assert trainer._model is not None
        
    def test_train_invalid_kernel_raises(self, trainer, sample_images_batch, sample_scores):
        """Test that invalid kernel raises ValueError."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        with pytest.raises(ValueError, match="Invalid kernel"):
            trainer.train(kernel='invalid_kernel')


# ==================== Model Save/Load Tests ====================

class TestModelSaveLoad:
    """Test model saving and loading."""
    
    def test_save_model_creates_files(self, trained_trainer, temp_dir):
        """Test that save_model creates the expected files."""
        svm_path = os.path.join(temp_dir, "model.txt")
        norm_path = os.path.join(temp_dir, "norm.pkl")
        
        trained_trainer.save_model(svm_path, norm_path)
        
        assert os.path.exists(svm_path)
        assert os.path.exists(norm_path)
        
    def test_save_model_without_training_raises(self, trainer, temp_dir):
        """Test that saving without training raises RuntimeError."""
        svm_path = os.path.join(temp_dir, "model.txt")
        norm_path = os.path.join(temp_dir, "norm.pkl")
        
        with pytest.raises(RuntimeError, match="No trained model to save"):
            trainer.save_model(svm_path, norm_path)
            
    def test_load_custom_model_in_brisque(self, trained_trainer, temp_dir, sample_image):
        """Test that saved model can be loaded in BRISQUE."""
        svm_path = os.path.join(temp_dir, "model.txt")
        norm_path = os.path.join(temp_dir, "norm.pkl")
        
        trained_trainer.save_model(svm_path, norm_path)
        
        # Load with BRISQUE
        obj = BRISQUE(url=False, model_path={"svm": svm_path, "normalize": norm_path})
        score = obj.score(sample_image)
        
        assert isinstance(score, float)
        
    def test_saved_model_format_compatibility(self, trained_trainer, temp_dir):
        """Test that saved model format is compatible with BRISQUE."""
        svm_path = os.path.join(temp_dir, "model.txt")
        norm_path = os.path.join(temp_dir, "norm.pkl")
        
        trained_trainer.save_model(svm_path, norm_path)
        
        # Check normalization file format
        with open(norm_path, 'rb') as f:
            scaler_params = pickle.load(f)
            
        assert 'min_' in scaler_params
        assert 'max_' in scaler_params


# ==================== Evaluation Tests ====================

class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluate_returns_dict(self, trained_trainer, sample_images_batch, sample_scores):
        """Test that evaluate returns a dictionary."""
        metrics = trained_trainer.evaluate(sample_images_batch, sample_scores)
        assert isinstance(metrics, dict)
        
    def test_evaluate_returns_rmse(self, trained_trainer, sample_images_batch, sample_scores):
        """Test that evaluate returns RMSE."""
        metrics = trained_trainer.evaluate(sample_images_batch, sample_scores)
        assert 'rmse' in metrics
        assert isinstance(metrics['rmse'], float)
        
    def test_evaluate_returns_plcc(self, trained_trainer, sample_images_batch, sample_scores):
        """Test that evaluate returns PLCC."""
        metrics = trained_trainer.evaluate(sample_images_batch, sample_scores)
        assert 'plcc' in metrics
        assert isinstance(metrics['plcc'], float)
        
    def test_evaluate_returns_srocc(self, trained_trainer, sample_images_batch, sample_scores):
        """Test that evaluate returns SROCC."""
        metrics = trained_trainer.evaluate(sample_images_batch, sample_scores)
        assert 'srocc' in metrics
        assert isinstance(metrics['srocc'], float)
        
    def test_evaluate_without_training_raises(self, trainer, sample_image):
        """Test that evaluating without training raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No trained model"):
            trainer.evaluate([sample_image], [30.0])


# ==================== Clear Tests ====================

class TestClear:
    """Test clear functionality."""
    
    def test_clear_resets_features(self, trainer, sample_images_batch, sample_scores):
        """Test that clear resets features."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        trainer.clear()
        assert trainer.features == []
        
    def test_clear_resets_scores(self, trainer, sample_images_batch, sample_scores):
        """Test that clear resets scores."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        trainer.clear()
        assert trainer.scores == []
        
    def test_clear_resets_model(self, trainer, sample_images_batch, sample_scores):
        """Test that clear resets model."""
        for img, score in zip(sample_images_batch, sample_scores):
            trainer.add_image(img, score)
        trainer.train()
        trainer.clear()
        assert trainer._model is None


# ==================== Dataset Loading Tests ====================

class TestDatasetLoading:
    """Test dataset loading functionality."""
    
    def test_add_dataset_from_csv(self, trainer, temp_dir):
        """Test loading dataset from CSV file."""
        # Create test images
        img_dir = Path(temp_dir) / "images"
        img_dir.mkdir()
        
        for i in range(3):
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(img_dir / f"img_{i}.jpg")
        
        # Create CSV file
        csv_path = Path(temp_dir) / "scores.csv"
        with open(csv_path, 'w') as f:
            f.write("image,score\n")
            f.write("img_0.jpg,30.0\n")
            f.write("img_1.jpg,45.0\n")
            f.write("img_2.jpg,25.0\n")
        
        trainer.add_dataset(str(img_dir), str(csv_path))
        
        assert len(trainer.features) == 3
        assert len(trainer.scores) == 3
        
    def test_add_dataset_missing_directory_raises(self, trainer):
        """Test that missing image directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            trainer.add_dataset("/nonexistent/path", "scores.csv")
            
    def test_add_dataset_missing_csv_raises(self, trainer, temp_dir):
        """Test that missing CSV file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Scores file not found"):
            trainer.add_dataset(temp_dir, "/nonexistent/scores.csv")
            
    def test_add_dataset_custom_column_names(self, trainer, temp_dir):
        """Test loading dataset with custom column names."""
        img_dir = Path(temp_dir) / "images"
        img_dir.mkdir()
        
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(img).save(img_dir / "test.jpg")
        
        csv_path = Path(temp_dir) / "scores.csv"
        with open(csv_path, 'w') as f:
            f.write("filename,quality\n")
            f.write("test.jpg,35.0\n")
        
        trainer.add_dataset(
            str(img_dir), 
            str(csv_path),
            image_column="filename",
            score_column="quality"
        )
        
        assert len(trainer.features) == 1
        assert trainer.scores[0] == 35.0


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_training_workflow(self, temp_dir):
        """Test complete training workflow from scratch."""
        # Generate training data
        np.random.seed(42)
        images = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
        scores = [np.random.uniform(20, 60) for _ in range(10)]
        
        # Train
        trainer = BRISQUETrainer()
        for img, score in zip(images, scores):
            trainer.add_image(img, score)
        trainer.train()
        
        # Save
        svm_path = os.path.join(temp_dir, "custom_svm.txt")
        norm_path = os.path.join(temp_dir, "custom_norm.pkl")
        trainer.save_model(svm_path, norm_path)
        
        # Use with BRISQUE
        obj = BRISQUE(url=False, model_path={"svm": svm_path, "normalize": norm_path})
        test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        score = obj.score(test_img)
        
        assert isinstance(score, float)
        
    def test_trained_model_produces_reasonable_scores(self, temp_dir):
        """Test that trained model produces reasonable quality scores."""
        np.random.seed(42)
        
        # Create clean and noisy images
        clean_images = [np.ones((64, 64, 3), dtype=np.uint8) * 128 for _ in range(5)]
        noisy_images = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        
        # Clean images should have better scores (lower values)
        clean_scores = [10.0, 15.0, 12.0, 14.0, 11.0]
        noisy_scores = [60.0, 70.0, 65.0, 75.0, 68.0]
        
        # Train
        trainer = BRISQUETrainer()
        for img, score in zip(clean_images + noisy_images, clean_scores + noisy_scores):
            trainer.add_image(img, score)
        trainer.train()
        
        # Save and load
        svm_path = os.path.join(temp_dir, "model.txt")
        norm_path = os.path.join(temp_dir, "norm.pkl")
        trainer.save_model(svm_path, norm_path)
        
        obj = BRISQUE(url=False, model_path={"svm": svm_path, "normalize": norm_path})
        
        clean_score = obj.score(clean_images[0])
        noisy_score = obj.score(noisy_images[0])
        
        # Clean should have lower score than noisy
        assert clean_score < noisy_score


# ==================== Backward Compatibility Tests ====================

class TestBackwardCompatibility:
    """Test backward compatibility with existing BRISQUE functionality."""
    
    def test_default_brisque_still_works(self, sample_image):
        """Test that BRISQUE with default model still works."""
        obj = BRISQUE(url=False)
        score = obj.score(sample_image)
        assert isinstance(score, float)
        
    def test_default_brisque_returns_reasonable_score(self):
        """Test that default BRISQUE returns reasonable scores."""
        from PIL import Image
        
        obj = BRISQUE(url=False)
        
        # Test with sample image
        img_path = "brisque/tests/sample-image.jpg"
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            score = obj.score(img)
            # BRISQUE scores are typically between 0-100
            assert 0 <= score <= 150
            
    def test_brisque_url_mode_signature(self):
        """Test that url parameter still works."""
        obj = BRISQUE(url=True)
        assert obj.url == True
        
        obj = BRISQUE(url=False)
        assert obj.url == False
        
    def test_brisque_default_model_path_none_works(self):
        """Test that model_path=None uses default model."""
        obj = BRISQUE(url=False, model_path=None)
        assert obj.model is not None
        assert obj.scale_params is not None
        
    def test_brisque_invalid_model_path_raises(self):
        """Test that invalid model_path format raises ValueError."""
        with pytest.raises(ValueError, match="model_path must be a dict"):
            BRISQUE(url=False, model_path="invalid_path")
            
    def test_brisque_missing_keys_raises(self):
        """Test that missing keys in model_path raises ValueError."""
        with pytest.raises(ValueError, match="model_path must contain"):
            BRISQUE(url=False, model_path={"svm": "model.txt"})


# ==================== Edge Case Tests ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_image(self, trainer):
        """Test feature extraction on very small image."""
        small_img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        features = trainer._extract_features(small_img)
        assert features.shape == (36,)
        
    def test_large_image(self, trainer):
        """Test feature extraction on larger image."""
        large_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        features = trainer._extract_features(large_img)
        assert features.shape == (36,)
        
    def test_all_zeros_image(self, trainer):
        """Test feature extraction on all-zeros image."""
        zeros_img = np.zeros((64, 64, 3), dtype=np.uint8)
        features = trainer._extract_features(zeros_img)
        assert features.shape == (36,)
        assert not np.any(np.isnan(features))
        
    def test_all_ones_image(self, trainer):
        """Test feature extraction on all-ones image."""
        ones_img = np.ones((64, 64, 3), dtype=np.uint8) * 255
        features = trainer._extract_features(ones_img)
        assert features.shape == (36,)
        assert not np.any(np.isnan(features))
        
    def test_single_channel_rgb_shape(self, trainer):
        """Test image with single channel but RGB shape."""
        img = np.random.randint(0, 256, (64, 64, 1), dtype=np.uint8)
        # This should work as the image will be handled appropriately
        features = trainer._extract_features(img)
        assert features.shape == (36,)


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])