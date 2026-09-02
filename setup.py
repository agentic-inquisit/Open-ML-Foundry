from setuptools import setup, find_packages

setup(
    name="open-ml-foundry",
    version="0.3.0",
    description="Open ML Foundry: Open-source stack to fine-tune models locally, accelerate edge deployments",
    author="Open ML Foundry Contributors",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "colorama>=0.4.4",  # For colored output on Windows
    ],
    entry_points={
        "console_scripts": [
            "foundry=sentinel.cli.main:cli",
            "omlf=sentinel.cli.main:cli",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
