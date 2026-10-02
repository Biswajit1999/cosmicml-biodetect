from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cosmicml-biodetect",
    version="0.2.0",
    author="Biswajit Jana",
    description="Synthetic spectral-amplitude inverse-problem benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Biswajit1999/cosmicml-biodetect",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26,<3",
        "scikit-learn>=1.4,<2",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.8,<4",
            "pytest>=8,<9",
            "ruff>=0.8,<1",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
)
