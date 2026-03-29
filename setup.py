from setuptools import setup, find_packages
setup(
    name="microdrug",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.9",
    author="Your Name",
    description="MicroDrugNet: Cross-Attention Fusion for Pharmacomicrobiomics",
    url="https://github.com/YOUR_USERNAME/MicroDrugNet",
)
