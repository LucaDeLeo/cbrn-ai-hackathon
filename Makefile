.PHONY: fmt lint test

fmt:
	python -m black scripts tests
	python -m isort scripts tests

lint:
	python -m flake8 scripts tests || true

test:
	pytest -q
