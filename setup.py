"""Setup script for two_tower_retrieval package."""

from __future__ import annotations

from setuptools import find_packages, setup


def _read_requirements(path: str = "requirements.txt") -> list[str]:
    """Read requirements from file."""
    with open(path, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="two_tower_retrieval",
    version="0.1.0",
    description=(
        "Two-Tower Neural Retrieval System trained on MovieLens-25M"
    ),
    author="ML Research",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=_read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train=scripts.train:main",
        ],
    },
)
