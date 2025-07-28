from setuptools import setup, find_packages

setup(
    name="decentralized_locale",
    version="0.1.0",
    description="Decentralized Sensor Network Localization using Matrix-Parametrized Proximal Splittings",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "cvxpy>=1.2.0",
        "mpi4py>=3.1.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "black", "flake8"],
        "mosek": ["mosek>=10.0.0"],
    },
    python_requires=">=3.8",
)