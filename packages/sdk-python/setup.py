"""Setup script for backward compatibility with older build tools."""
from setuptools import setup, find_packages

# Read version from __init__.py
with open("efference/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

setup(
    name="efference",
    version=version,
    packages=find_packages(),
    install_requires=[
        "httpx>=0.25.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "visualization": ["matplotlib>=3.5.0"],
    },
    python_requires=">=3.9",
)