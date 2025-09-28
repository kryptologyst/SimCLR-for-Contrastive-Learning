"""
Setup script for SimCLR Contrastive Learning
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="simclr-contrastive-learning",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="A modern implementation of SimCLR for contrastive learning with interactive UI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simclr-contrastive-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "ui": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
            "plotly>=5.0.0",
        ],
        "logging": [
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "simclr-train=main:main",
            "simclr-ui=streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="simclr, contrastive learning, self-supervised learning, computer vision, pytorch, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/simclr-contrastive-learning/issues",
        "Source": "https://github.com/yourusername/simclr-contrastive-learning",
        "Documentation": "https://github.com/yourusername/simclr-contrastive-learning#readme",
    },
)
