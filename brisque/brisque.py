try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is required for BRISQUE. Install it with one of:\n"
        "  pip install brisque[opencv-python]\n"
        "  pip install brisque[opencv-python-headless]\n"
        "  pip install brisque[opencv-contrib-python]\n"
        "  pip install brisque[opencv-contrib-python-headless]\n"
        "\n"
        "Or install OpenCV separately:\n"
        "  pip install opencv-python-headless\n"
        "  pip install brisque"
    ) from e

import collections
from itertools import chain
# import urllib.request as request
import pickle
import numpy as np
import scipy.signal as signal
import scipy.ndimage.filters as filters
import scipy.special as special
import scipy.optimize as optimize
import skimage.io
import skimage.color
from libsvm import svmutil
import requests
import os
from brisque.models import MODEL_PATH

# Workaround for libsvm compatibility with newer scipy versions
# libsvm uses scipy.ndarray which was removed in scipy 1.8+
import scipy
if not hasattr(scipy, 'ndarray'):
    scipy.ndarray = np.ndarray


class BRISQUE:
    """
    Blind/Referenceless Image Spatial Quality Evaluator.
    
    BRISQUE is a no-reference image quality assessment algorithm that computes
    a quality score for images without requiring a reference image.
    
    Lower scores indicate better quality. Typical scores range from 0-100,
    where 0 is perfect quality and higher values indicate more distortion.
    
    Args:
        url (bool): If True, the score() method accepts URLs. If False, 
                    it accepts numpy arrays. Default: False.
        model_path (dict, optional): Custom model paths. Should contain:
            - 'svm': Path to SVM model file (.txt)
            - 'normalize': Path to normalization parameters (.pickle)
            If None, uses the default pre-trained model.
    
    Example:
        >>> from brisque import BRISQUE
        >>> import numpy as np
        >>> from PIL import Image
        >>> 
        >>> # Using default model
        >>> obj = BRISQUE(url=False)
        >>> img = np.array(Image.open("image.jpg"))
        >>> score = obj.score(img)
        >>> 
        >>> # Using custom trained model
        >>> obj = BRISQUE(url=False, model_path={
        ...     "svm": "custom_svm.txt",
        ...     "normalize": "custom_normalize.pickle"
        ... })
        >>> score = obj.score(img)
    """
    
    def __init__(self, url=False, model_path=None):
        """Initialize BRISQUE with optional custom model."""
        self.url = url
        
        # Use default model if no custom path provided (backward compatible)
        if model_path is None:
            svm_path = os.path.join(MODEL_PATH, "svm.txt")
            norm_path = os.path.join(MODEL_PATH, "normalize.pickle")
        else:
            if not isinstance(model_path, dict):
                raise ValueError("model_path must be a dict with 'svm' and 'normalize' keys")
            if 'svm' not in model_path or 'normalize' not in model_path:
                raise ValueError("model_path must contain 'svm' and 'normalize' keys")
            svm_path = model_path['svm']
            norm_path = model_path['normalize']

        # Load model
        self.model = svmutil.svm_load_model(svm_path)
        with open(norm_path, 'rb') as f:
            self.scale_params = pickle.load(f)

    def load_image(self, img):
        if self.url:
            response = requests.get(img)
            image = response.content
            return skimage.io.imread(image, plugin='pil')
        else:
            return img


    def remove_alpha_channel(self, original_image):
        image = np.array(original_image)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        return image

    def score(self, img):
        image = self.load_image(img)
        image = self.remove_alpha_channel(image)
        gray_image = skimage.color.rgb2gray(image)
        mscn_coefficients = self.calculate_mscn_coefficients(gray_image, 7, 7 / 6)
        coefficients = self.calculate_pair_product_coefficients(mscn_coefficients)
        brisque_features = self.calculate_brisque_features(gray_image, kernel_size=7, sigma=7 / 6)
        downscaled_image = cv2.resize(gray_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_CUBIC)
        downscale_brisque_features = self.calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7 / 6)
        brisque_features = np.concatenate((brisque_features, downscale_brisque_features))

        return self.calculate_image_quality_score(brisque_features)

    def normalize_kernel(self, kernel):
        return kernel / np.sum(kernel)

    def gaussian_kernel2d(self, n, sigma):
        Y, X = np.indices((n, n)) - int(n / 2)
        gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        return self.normalize_kernel(gaussian_kernel)

    def local_mean(self, image, kernel):
        return signal.convolve2d(image, kernel, 'same')

    def local_deviation(self, image, local_mean, kernel):
        "Vectorized approximation of local deviation"
        sigma = image ** 2
        sigma = signal.convolve2d(sigma, kernel, 'same')
        return np.sqrt(np.abs(local_mean ** 2 - sigma))

    def calculate_mscn_coefficients(self, image, kernel_size=6, sigma=7 / 6):
        C = 1 / 255
        kernel = self.gaussian_kernel2d(kernel_size, sigma=sigma)
        local_mean = signal.convolve2d(image, kernel, 'same')
        local_var = self.local_deviation(image, local_mean, kernel)

        return (image - local_mean) / (local_var + C)

    def generalized_gaussian_dist(self, x, alpha, sigma):
        beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))

        coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
        return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)

    def calculate_pair_product_coefficients(self, mscn_coefficients):
        return collections.OrderedDict({
            'mscn': mscn_coefficients,
            'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
            'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
            'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
            'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
        })

    def asymmetric_generalized_gaussian(self, x, nu, sigma_l, sigma_r):
        def beta(sigma):
            return sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))

        coefficient = nu / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu))
        f = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu)

        return np.where(x < 0, f(-x, sigma_l), f(x, sigma_r))

    def asymmetric_generalized_gaussian_fit(self, x):
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

        def mean_squares_sum(x, filter=lambda z: z == z):
            filtered_values = x[filter(x)]
            squares_sum = np.sum(filtered_values ** 2)
            return squares_sum / ((filtered_values.shape))

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

        def estimate_sigma(x, alpha, filter=lambda z: z < 0):
            return np.sqrt(mean_squares_sum(x, filter))

        def estimate_mean(alpha, sigma_l, sigma_r):
            return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))

        alpha = estimate_alpha(x)
        sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
        sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)

        constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
        mean = estimate_mean(alpha, sigma_l, sigma_r)

        return alpha, mean, sigma_l, sigma_r

    def calculate_brisque_features(self, image, kernel_size=7, sigma=7 / 6):
        def calculate_features(coefficients_name, coefficients, accum=np.array([], dtype=object)):
            alpha, mean, sigma_l, sigma_r = self.asymmetric_generalized_gaussian_fit(coefficients)

            if coefficients_name == 'mscn':
                var = (sigma_l ** 2 + sigma_r ** 2) / 2
                return [alpha, var]

            return [alpha, mean, sigma_l ** 2, sigma_r ** 2]

        mscn_coefficients = self.calculate_mscn_coefficients(image, kernel_size, sigma)
        coefficients = self.calculate_pair_product_coefficients(mscn_coefficients)

        features = [calculate_features(coefficients_name=name, coefficients=coeff) for name, coeff in
                    coefficients.items()]
        flatten_features = list(chain.from_iterable(features))
        return np.array(flatten_features, dtype=object)

    def scale_features(self, features):
        min_ = np.array(self.scale_params['min_'], dtype=object)
        max_ = np.array(self.scale_params['max_'], dtype=object)
        
        # Convert features to flat float array for proper scaling
        features_flat = np.array([float(f) for f in features], dtype=np.float64)
        min_flat = np.array([float(m) for m in min_], dtype=np.float64)
        max_flat = np.array([float(m) for m in max_], dtype=np.float64)

        return -1 + (2.0 / (max_flat - min_flat) * (features_flat - min_flat))

    def calculate_image_quality_score(self, brisque_features):
        scaled_brisque_features = self.scale_features(brisque_features)
        
        # Convert to pure Python list of floats to avoid scipy.ndarray compatibility issues in libsvm
        # libsvm checks isinstance(xi, scipy.ndarray) which fails with newer scipy versions
        feature_list = [float(f) for f in scaled_brisque_features]

        x, idx = svmutil.gen_svm_nodearray(
            feature_list,
            isKernel=False)

        nr_classifier = 1
        prob_estimates = (svmutil.c_double * nr_classifier)()

        return svmutil.libsvm.svm_predict_probability(self.model, x, prob_estimates)
