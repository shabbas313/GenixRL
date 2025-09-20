from setuptools import setup, find_packages

setup(
    name="genixrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "pysam>=0.19.0",
        "myvariant>=0.3.0",
        "tqdm>=4.65.0"
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "shap>=0.41.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "genixrl-predict=genixrl.scripts.predict:main"
        ]
    },
    author="Hassan Abbas",
    author_email="shabbas12@outlook.com",
    description="Logistic Regression with RL Fusion for variant classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shabbas313/GenixRL",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)