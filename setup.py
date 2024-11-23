"""
Setup file for the decoding library.
"""

import pathlib

import setuptools

core_requirements = [
    "jax==0.4.35",
    "jaxtyping==0.2.36",
    "vllm==0.6.3.post1",
]
dev_requirements = [
    "pdoc==15.0.0",
    "pre-commit==4.0.1",
    "pyright==1.1.389",
    "pytest==8.3.3",
    "pytest-cov==6.0.0",
    "pytest-html==4.1.1",
    "ruff==0.8.0",
]

with pathlib.Path("README.md").open(encoding="utf-8") as f:
    readme = f.read()

setuptools.setup(
    name="decoding",
    version="0.1.4",
    description="Composable inference algorithms with LLMs and programmable logic",
    long_description=readme,
    long_description_content_type="text/markdown",
    authors=["Ben Lipkin"],
    license="Apache 2.0",
    install_requires=core_requirements,
    extras_require={"dev": dev_requirements},
    python_requires=">=3.11",
    packages=["decoding"],
)
