.DEFAULT_GOAL := help
.PHONY: install-dev install format lint scufris

### QUICK
# ¯¯¯¯¯¯¯

install-dev: ## Install dev dependencies
	pip install -r requirements-dev.txt --upgrade --no-warn-script-location

install: ## Install dependencies
	pip install -r requirements.txt --upgrade --no-warn-script-location

format: ## Format
	python -m isort src --skip .venv/
	python -m black src --exclude .venv/

lint: ## Linter
	python -m pylint src

scufris: ## Deploy bot
	python src/scufris.py
