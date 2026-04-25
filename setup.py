"""Setuptools configuration for packaging open_bargain."""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path: Path) -> list[str]:
    """Load non-empty requirement lines from a text file."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


ROOT = Path(__file__).parent.resolve()
REQUIREMENTS = read_requirements(ROOT / "requirements.txt")
README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="open_bargain",
    version="0.1.0",
    description="OpenBargain: a multi-agent negotiation benchmark for OpenEnv.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
