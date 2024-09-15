import setuptools
from brisque import __version__
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brisque", # Replace with your own username
    version=__version__,
    author="Rehan Guha",
    py_modules=["brisque"],
    license='Apache-2.0',
    author_email="rehanguha29@gmail.com",
    description="Image Quality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rehanguha/brisque",
    package_dir={'brisque': 'brisque'},
    packages=setuptools.find_packages(),
    include_package_data=True,
    keywords = ['quality', 'svm', 'image', 'maths'],
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        'Topic :: Software Development :: Libraries :: Python Modules',
        "Topic :: Scientific/Engineering",
        'Intended Audience :: Developers',
    ],
    python_requires='>=2.7',
    install_requires=['numpy', 'scikit-image', 'scipy', 'opencv-python', 'libsvm-official'],
)
