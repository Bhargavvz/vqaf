"""Medical VQA System - Package Setup"""

from setuptools import setup, find_packages

setup(
    name="medical_vqa",
    version="1.0.0",
    description="Knowledge-Guided Explainable Medical VQA",
    packages=["medical_vqa"] + [
        f"medical_vqa.{pkg}" for pkg in
        ["data", "knowledge", "model", "training", "explainability", "evaluation", "api"]
    ],
    package_dir={"medical_vqa": "."},
    python_requires=">=3.10",
)
