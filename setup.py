from setuptools import setup, find_packages

setup(
    name="field-delineation",
    version="0.1.0",
    description="Field delineation using satellite imagery and deep learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pytorch-lightning>=1.5.0",
        "segmentation-models-pytorch>=0.3.0",
        "transformers>=4.0.0",
        "rasterio>=1.2.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.18.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "tqdm>=4.60.0",
        "wandb>=0.12.0",
        "netCDF4>=1.5.0",
        "medpy>=0.4.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "field-delineation-train=scripts.train:main",
            "field-delineation-inference=scripts.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)