MAKEFLAGS += --warn-undefined-variables
SHELL = /bin/bash -o pipefail
.DEFAULT_GOAL := help
.PHONY: help clean install install-hooks check format pyright test hooks openapi

venv ?= .venv
pip := $(venv)/bin/pip

python := $(venv)/bin/python

## display help message
help:
	@awk '/^##.*$$/,/^[~\/\.0-9a-zA-Z_-]+:/' $(MAKEFILE_LIST) | awk '!(NR%2){print $$0p}{p=$$0}' | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sort

$(pip): $(if $(value CI),|,) .python-version
# create venv using system python even when another venv is active
	PATH=$${PATH#$${VIRTUAL_ENV}/bin:} python3 -m venv --clear $(venv)
	$(venv)/bin/python --version
	$(pip) install pip~=24.0

$(venv): $(if $(value CI),|,) pyproject.toml $(pip)
	$(pip) install -e '.[dev]'

# delete the venv
clean:
	rm -rf $(venv)

## create venv and install this package and hooks
install: $(venv) node_modules $(if $(value CI),,install-hooks)

node_modules: package.json
	npm install
	touch node_modules

install-hooks: .git/hooks/pre-push

.git/hooks/pre-push: $(venv)
	$(venv)/bin/pre-commit install --install-hooks -t pre-push

## format, lint and type check
check: export SKIP=test
check: hooks

## format and lint
format: export SKIP=pyright,test
format: hooks

## pyright type check
pyright: node_modules $(venv)
	node_modules/.bin/pyright

## run unit tests
test: $(venv) qdrant-up
	PATH="$(venv)/bin:$$PATH" pytest

## run pre-commit hooks on all files
hooks: node_modules $(venv)
	$(venv)/bin/pre-commit run --color=always --all-files --hook-stage push

## run the api locally
run: $(venv) qdrant-up
	$(venv)/bin/uvicorn app.main:app --reload

## generate openapi spec
openapi: $(venv)
	$(python) ./openapi/generate-openapi.py

## start up a local Qdrant server
qdrant-up:
	docker-compose up -d qdrant

## stop the local Qdrant server
qdrant-down:
	docker-compose down

## clean out the local Qdrant data store
qdrant-clean:
	rm -rf ./data/qdrant_storage
	rm -rf /tmp/xbot

# List the local Qdrant collections
qdrant-list:
	PORT=$$(docker-compose port qdrant 6333 | cut -d: -f2); \
	curl -s -X GET http://localhost:$$PORT/collections | jq '.result.collections | .[] | .name'
