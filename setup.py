import setuptools
from brisque import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

# Core dependencies (OpenCV is now optional - use extras_require)
INSTALL_REQUIRES = [
    "numpy>=1.20",
    "scikit-image>=0.19",
    "scipy>=1.7",
    "libsvm-official>=3.32",
    "requests>=2.25",
]

# Optional OpenCV variants - users choose one based on their environment
EXTRAS_REQUIRE = {
    "opencv-python": ["opencv-python>=4.5"],
    "opencv-python-headless": ["opencv-python-headless>=4.5"],
    "opencv-contrib-python": ["opencv-contrib-python>=4.5"],
    "opencv-contrib-python-headless": ["opencv-contrib-python-headless>=4.5"],
}

setuptools.setup(
    name="brisque",
    version=__version__,
    author="Rehan Guha",
    py_modules=["brisque"],
    license='Apache-2.0',
    author_email="rehanguha29@gmail.com",
    description="Image Quality Assessment using BRISQUE algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rehanguha/brisque",
    package_dir={'brisque': 'brisque'},
    packages=setuptools.find_packages(),
    include_package_data=True,
    keywords=['quality', 'svm', 'image', 'maths', 'brisque', 'image-quality'],
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        'Topic :: Software Development :: Libraries :: Python Modules',
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Processing",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
