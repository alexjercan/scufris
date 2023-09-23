.DEFAULT_GOAL := help
.PHONY: help format lint

help: ## Show this help
	@cat Makefile | grep -E "^\w+$:"

fmt: # Format code
	poetry run isort scufris tests
	poetry run black scufris tests

lint: # Lint code
	poetry run pylint scufris tests
	poetry run mypy scufris tests --ignore-missing-imports
