# Contributing 🌳

## Prerequisites

- make
- node (required for pyright). To install see `Install Node`
- python >= 3.11

## Getting started

Fork this repo and run `make install` to setup your local dev environment with:

- a virtualenv in _.venv/_
- pyright in _node_modules/_
- git hooks for formatting & linting on git push

`venv` activates the virtualenv.

The make targets will update the virtualenv when _pyproject.toml_ changes.

## Hugging Face models

Hugging face models (sentence-transformers, transformers) have been included as pypi packages (in pyproject.toml) and can take some time to download on first use.

## Usage

Run `make` to see the options for running tests, linting, formatting etc.

`make run` will start an instance of Qdrant (available at http://localhost:6333) and the API (available at http://localhost:8000).

`make check` will run linting, formatting and tests so can do this prior to pushing your code.

### Adding packages

Add packages to [pyproject.toml](pyproject.toml):

- in the `dev` section of `[project.optional-dependencies]` if only used for development
- in `[project.dependencies]` if required by your package when used in production or in other projects
