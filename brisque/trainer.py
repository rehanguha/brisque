"""
BRISQUE Trainer Module

Train custom BRISQUE quality assessment models with your own image datasets.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

import cv2
import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize
from libsvm import svmutil
from PIL import Image

from brisque.models import MODEL_PATH


class BRISQUETrainer:
    """
    Train custom BRISQUE quality assessment models.
    
    This class allows you to train a custom SVM model for image quality
    assessment using your own dataset of images with known quality scores
    (e.g., Mean Opinion Scores from subjective studies).
    
    The trainer extracts the same 36 BRISQUE features used by the standard
    model and trains an SVM regression model to predict quality scores.
    
    Attributes:
        features (list): List of extracted feature vectors.
        scores (list): List of corresponding quality scores.
        
    Example:
        >>> from brisque.trainer import BRISQUETrainer
        >>> import numpy as np
        >>> 
        >>> # Initialize trainer
        >>> trainer = BRISQUETrainer()
        >>> 
        >>> # Add images with their quality scores
        >>> trainer.add_image(image1, score=4.5)
        >>> trainer.add_image(image2, score=2.1)
        >>> 
        >>> # Or load from dataset
        >>> trainer.add_dataset("path/to/images", "path/to/scores.csv")
        >>> 
        >>> # Train the model
        >>> trainer.train()
        >>> 
        >>> # Save for later use
        >>> trainer.save_model("custom_svm.txt", "custom_normalize.pkl")
        
        >>> # Use with BRISQUE
        >>> from brisque import BRISQUE
        >>> obj = BRISQUE(model_path={"svm": "custom_svm.txt", "normalize": "custom_normalize.pkl"})
        >>> obj.score(new_image)
    """
    
    def __init__(self):
        """Initialize the BRISQUE trainer."""
        self.features: List[np.ndarray] = []
        self.scores: List[float] = []
        self._model = None
        self._scaler_params: Optional[Dict] = None
        
    def add_image(self, image: np.ndarray, score: float) -> None:
        """
        Add a single image with its quality score.
        
        Args:
            image: Input image as numpy array (RGB or grayscale).
            score: Quality score for the image (e.g., MOS, DMOS).
                   Lower scores typically indicate better quality.
        
        Raises:
            ValueError: If image has invalid shape or score is not numeric.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if not isinstance(score, (int, float)):
            raise ValueError("Score must be a numeric value")
            
        features = self._extract_features(image)
        self.features.append(features)
        self.scores.append(float(score))
        
    def add_dataset(
        self, 
        image_dir: str, 
        scores_file: str,
        image_column: str = "image",
        score_column: str = "score"
    ) -> None:
        """
        Load images and scores from a directory and CSV file.
        
        Args:
            image_dir: Path to directory containing images.
            scores_file: Path to CSV file with image names and scores.
            image_column: Column name for image filenames (default: "image").
            score_column: Column name for quality scores (default: "score").
            
        Raises:
            FileNotFoundError: If image_dir or scores_file doesn't exist.
            ValueError: If CSV format is invalid or images can't be loaded.
        """
        import csv
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(scores_file):
            raise FileNotFoundError(f"Scores file not found: {scores_file}")
            
        with open(scores_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        for row in rows:
            if image_column not in row or score_column not in row:
                raise ValueError(f"CSV must have '{image_column}' and '{score_column}' columns")
                
            image_path = image_dir / row[image_column]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            image = np.array(Image.open(image_path))
            score = float(row[score_column])
            self.add_image(image, score)
            
    def clear(self) -> None:
        """Clear all added images and scores."""
        self.features = []
        self.scores = []
        self._model = None
        self._scaler_params = None
        
    def train(
        self, 
        svm_c: float = 1.0, 
        svm_gamma: float = 0.1,
        svm_epsilon: float = 0.1,
        kernel: str = "rbf"
    ) -> None:
        """
        Train SVM regression model on collected data.
        
        Args:
            svm_c: SVM regularization parameter (default: 1.0).
                   Larger values penalize errors more.
            svm_gamma: Kernel coefficient for RBF kernel (default: 0.1).
            svm_epsilon: Epsilon in epsilon-SVR (default: 0.1).
            kernel: SVM kernel type - "rbf", "linear", "poly", or "sigmoid"
                    (default: "rbf").
        
        Raises:
            ValueError: If fewer than 2 images have been added.
        """
        if len(self.features) < 2:
            raise ValueError("Need at least 2 images to train the model")
            
        # Convert to arrays
        X = np.array(self.features, dtype=np.float64)
        y = np.array(self.scores, dtype=np.float64)
        
        # Compute normalization parameters
        min_ = np.min(X, axis=0)
        max_ = np.max(X, axis=0)
        
        # Handle constant features (max == min)
        range_ = max_ - min_
        range_[range_ == 0] = 1.0  # Avoid division by zero
        
        self._scaler_params = {
            'min_': min_.tolist(),
            'max_': max_.tolist()
        }
        
        # Scale features to [-1, 1]
        X_scaled = -1 + (2.0 / range_ * (X - min_))
        
        # Map kernel name to libsvm code
        kernel_map = {
            'linear': 0,
            'poly': 1,
            'rbf': 2,
            'sigmoid': 3
        }
        
        if kernel not in kernel_map:
            raise ValueError(f"Invalid kernel: {kernel}. Must be one of {list(kernel_map.keys())}")
        
        kernel_code = kernel_map[kernel]
        
        # Format for libsvm: svm_train(labels, features, options)
        # -s 3: epsilon-SVR (regression)
        # -t {kernel_code}: kernel type
        # -c {svm_c}: regularization
        # -g {svm_gamma}: gamma for RBF
        # -p {svm_epsilon}: epsilon
        options = f'-s 3 -t {kernel_code} -c {svm_c} -g {svm_gamma} -p {svm_epsilon} -q'
        
        # Train SVM
        problem = svmutil.svm_problem(y.tolist(), X_scaled.tolist())
        params = svmutil.svm_parameter(options)
        self._model = svmutil.svm_train(problem, params)
        
    def save_model(self, svm_path: str, norm_path: str) -> None:
        """
        Save trained model and normalization parameters.
        
        Args:
            svm_path: Path to save the SVM model file (.txt).
            norm_path: Path to save the normalization parameters (.pkl or .pickle).
        
        Raises:
            RuntimeError: If model hasn't been trained yet.
        """
        if self._model is None:
            raise RuntimeError("No trained model to save. Call train() first.")
        if self._scaler_params is None:
            raise RuntimeError("No normalization parameters. Call train() first.")
            
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(svm_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(norm_path)), exist_ok=True)
        
        # Save SVM model
        svmutil.svm_save_model(svm_path, self._model)
        
        # Save normalization parameters
        with open(norm_path, 'wb') as f:
            pickle.dump(self._scaler_params, f)
            
    def evaluate(
        self, 
        images: List[np.ndarray], 
        scores: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Computes standard image quality assessment metrics:
        - RMSE: Root Mean Square Error
        - PLCC: Pearson Linear Correlation Coefficient
        - SROCC: Spearman Rank Order Correlation Coefficient
        
        Args:
            images: List of test images as numpy arrays.
            scores: List of ground truth quality scores.
        
        Returns:
            Dictionary with 'rmse', 'plcc', and 'srocc' metrics.
        
        Raises:
            RuntimeError: If model hasn't been trained yet.
        """
        if self._model is None:
            raise RuntimeError("No trained model. Call train() first.")
            
        predictions = []
        for image in images:
            pred = self._predict_single(image)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        ground_truth = np.array(scores)
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
        
        # PLCC (Pearson correlation)
        plcc = np.corrcoef(predictions, ground_truth)[0, 1]
        
        # SROCC (Spearman correlation)
        from scipy.stats import spearmanr
        srocc, _ = spearmanr(predictions, ground_truth)
        
        return {
            'rmse': float(rmse),
            'plcc': float(plcc),
            'srocc': float(srocc)
        }
        
    def _predict_single(self, image: np.ndarray) -> float:
        """Predict quality score for a single image."""
        features = self._extract_features(image)
        features = np.array(features, dtype=np.float64)
        
        # Scale features
        min_ = np.array(self._scaler_params['min_'])
        max_ = np.array(self._scaler_params['max_'])
        range_ = max_ - min_
        range_[range_ == 0] = 1.0
        
        features_scaled = -1 + (2.0 / range_ * (features - min_))
        
        # Predict
        x, idx = svmutil.gen_svm_nodearray(
            features_scaled.tolist(),
            isKernel=False
        )
        
        nr_classifier = 1
        prob_estimates = (svmutil.c_double * nr_classifier)()
        
        return svmutil.libsvm.svm_predict_probability(self._model, x, prob_estimates)
        
    # ==================== Feature Extraction Methods ====================
    # These methods mirror the BRISQUE class for feature extraction
    
    def _remove_alpha_channel(self, image: np.ndarray) -> np.ndarray:
        """Remove alpha channel if present."""
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        return image
    
    def _rgb_to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale."""
        if len(image.shape) == 3:
            # Handle single-channel images with 3D shape (H, W, 1)
            if image.shape[2] == 1:
                return image[:, :, 0]
            # Standard RGB to grayscale conversion
            return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        return image
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract BRISQUE features from an image."""
        # Handle different image formats
        image = self._remove_alpha_channel(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray_image = self._rgb_to_gray(image.astype(np.float64))
        else:
            gray_image = image.astype(np.float64)
            
        # Normalize to [0, 1] if needed
        if gray_image.max() > 1.0:
            gray_image = gray_image / 255.0
            
        # Calculate features at original scale
        brisque_features = self._calculate_brisque_features(gray_image, kernel_size=7, sigma=7/6)
        
        # Calculate features at half scale
        downscaled_image = cv2.resize(gray_image, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)
        downscale_brisque_features = self._calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)
        
        # Concatenate features (total 36 features)
        features = np.concatenate((brisque_features, downscale_brisque_features))
        
        return features
    
    def _normalize_kernel(self, kernel: np.ndarray) -> np.ndarray:
        """Normalize a kernel to sum to 1."""
        return kernel / np.sum(kernel)
    
    def _gaussian_kernel2d(self, n: int, sigma: float) -> np.ndarray:
        """Create a 2D Gaussian kernel."""
        Y, X = np.indices((n, n)) - int(n / 2)
        gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        return self._normalize_kernel(gaussian_kernel)
    
    def _local_mean(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Calculate local mean using convolution."""
        return signal.convolve2d(image, kernel, 'same')
    
    def _local_deviation(self, image: np.ndarray, local_mean: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Calculate local deviation."""
        sigma = image ** 2
        sigma = signal.convolve2d(sigma, kernel, 'same')
        return np.sqrt(np.abs(local_mean ** 2 - sigma))
    
    def _calculate_mscn_coefficients(
        self, 
        image: np.ndarray, 
        kernel_size: int = 7, 
        sigma: float = 7/6
    ) -> np.ndarray:
        """Calculate Mean Subtracted Contrast Normalized coefficients."""
        C = 1 / 255
        kernel = self._gaussian_kernel2d(kernel_size, sigma=sigma)
        local_mean = signal.convolve2d(image, kernel, 'same')
        local_var = self._local_deviation(image, local_mean, kernel)
        return (image - local_mean) / (local_var + C)
    
    def _calculate_pair_product_coefficients(self, mscn_coefficients: np.ndarray) -> dict:
        """Calculate pair product coefficients."""
        return {
            'mscn': mscn_coefficients,
            'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
            'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
            'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
            'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
        }
    
    def _asymmetric_generalized_gaussian_fit(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """Fit asymmetric generalized Gaussian distribution."""
        def estimate_phi(alpha):
            numerator = special.gamma(2 / alpha) ** 2
            denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
            return numerator / denominator

        def estimate_r_hat(x):
            size = np.prod(x.shape)
            return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

        def estimate_R_hat(r_hat, gamma):
            numerator = (gamma ** 3 + 1) * (gamma + 1)
            denominator = (gamma ** 2 + 1) ** 2
            return r_hat * numerator / denominator

        def mean_squares_sum(x, filter_func=lambda z: z == z):
            filtered_values = x[filter_func(x)]
            squares_sum = np.sum(filtered_values ** 2)
            return squares_sum / filtered_values.size

        def estimate_gamma(x):
            left_squares = mean_squares_sum(x, lambda z: z < 0)
            right_squares = mean_squares_sum(x, lambda z: z >= 0)
            return np.sqrt(left_squares) / np.sqrt(right_squares)

        def estimate_alpha(x):
            r_hat = estimate_r_hat(x)
            gamma = estimate_gamma(x)
            R_hat = estimate_R_hat(r_hat, gamma)
            solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x
            return solution[0]

        def estimate_sigma(x, alpha, filter_func=lambda z: z < 0):
            return np.sqrt(mean_squares_sum(x, filter_func))

        def estimate_mean(alpha, sigma_l, sigma_r):
            constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
            return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))

        alpha = estimate_alpha(x)
        sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
        sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)

        constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
        mean = estimate_mean(alpha, sigma_l, sigma_r)

        return alpha, mean, sigma_l, sigma_r
    
    def _calculate_brisque_features(
        self, 
        image: np.ndarray, 
        kernel_size: int = 7, 
        sigma: float = 7/6
    ) -> np.ndarray:
        """Calculate BRISQUE features for an image."""
        def calculate_features(coefficients_name, coefficients):
            alpha, mean, sigma_l, sigma_r = self._asymmetric_generalized_gaussian_fit(coefficients)

            if coefficients_name == 'mscn':
                var = (sigma_l ** 2 + sigma_r ** 2) / 2
                return [alpha, var]

            return [alpha, mean, sigma_l ** 2, sigma_r ** 2]

        mscn_coefficients = self._calculate_mscn_coefficients(image, kernel_size, sigma)
        coefficients = self._calculate_pair_product_coefficients(mscn_coefficients)

        features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
        flatten_features = [item for sublist in features for item in sublist]
        
        return np.array(flatten_features, dtype=np.float64)