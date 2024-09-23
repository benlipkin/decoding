SHELL := /usr/bin/env bash
SRC = decoding
TEST = tests
RUN = python -m
INSTALL = $(RUN) pip install
SRC_FILES := $(shell find $(SRC) -name '*.py')
TEST_FILES := $(shell find $(TEST) -name '*.py')
.DEFAULT_GOAL := help

## help      : print available commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repository from GitHub.
.PHONY : update
update :
	@git pull origin

## env       : setup env and install dependencies.
.PHONY : env
env : $(SRC).egg-info/
$(SRC).egg-info/ : setup.py .pre-commit-config.yaml
	@$(INSTALL) -e ".[dev]" && pre-commit install

## format    : format code style.
.PHONY : format
format : env
	@ruff format

## docs      : build documentation
.PHONY : docs
docs: env docs/index.html
docs/index.html : $(SRC_FILES)
	@pdoc $(SRC) -o $(@D) --docformat=google

## tests     : run linting and tests.
.PHONY : tests
tests : ruff pyright pytest
ruff : env .ruff.toml
	@ruff check --fix
pyright : env pyrightconfig.json
	@pyright -p .
pytest : env html/coverage/index.html
html/coverage/index.html : $(SRC_FILES) $(TEST_FILES)
	@pytest \
	--cov=$(SRC) --cov-branch --cov-report=html:html/coverage \
	--jaxtyping-packages=$(SRC),$(TEST),typeguard.typechecked \
	--html=$@ --self-contained-html $(TEST)
