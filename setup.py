from setuptools import setup, find_packages

setup(
    name="rasta",
    version="2.0.0",
    description="Open-source floor plan recognition & material identification engine",
    author="Hadrien Majérus",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.100",
        "uvicorn>=0.20",
        "opencv-python-headless>=4.8",
        "numpy>=1.24",
    ],
    extras_require={
        "gpu": ["torch>=2.0", "onnxruntime-gpu>=1.16"],
        "full": ["pdf2image>=1.16", "requests>=2.28"],
    },
    entry_points={
        "console_scripts": [
            "rasta=rasta.api:main",
        ],
    },
)
