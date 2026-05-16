SHELL  := bash
PYTHON := uv run python
RUNNER := $(PYTHON) scripts/make_runner.py

.PHONY: dev ci codegen lint test

dev:
	@$(RUNNER) --target dev

ci:
	@$(RUNNER) --target ci

codegen:
	@$(RUNNER) --target codegen

lint:
	@$(RUNNER) --target lint

test:
	@$(RUNNER) --target test
