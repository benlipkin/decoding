"""
Setup file for the decoding library.
"""

import setuptools

core_requirements = [
    "jax==0.4.31",
    "jaxtyping==0.2.34",
    "vllm==0.6.1.post2",
]
dev_requirements = [
    "pdoc==14.7.0",
    "pre-commit==3.8.0",
    "pyright==1.1.381",
    "pytest==8.3.3",
    "pytest-cov==5.0.0",
    "pytest-html==4.1.1",
    "ruff==0.6.7",
]

setuptools.setup(
    name="decoding",
    version="0.1.0",
    description="Composable LLM decoding algorithms",
    long_description="See https://github.com/benlipkin/decoding for more information.",
    long_description_content_type="text/markdown",
    authors=["Ben Lipkin"],
    license="Apache 2.0",
    install_requires=core_requirements,
    extras_require={"dev": dev_requirements},
    python_requires=">=3.11",
    packages=["decoding"],
)
