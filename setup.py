"""
Setup script for BRIAC2025 project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="briac2025",
    version="1.0.0",
    author="BRIAC2025 Team",
    author_email="team@briac2025.com",
    description="BRIAC2025 Competition - Classification and Segmentation Tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/briac2025/competition",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "viz": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "plotly>=5.0",
            "tensorboard>=2.8",
        ]
    },
    entry_points={
        "console_scripts": [
            "briac2025-train=main_train:main",
            "briac2025-predict=src.inference.predict:main",
            "briac2025-evaluate=src.utils.metrics:main",
        ],
    },
    include_package_data=True,
    package_data={
        "briac2025": ["config/*.yaml", "config/*.json"],
    },
    zip_safe=False,
)