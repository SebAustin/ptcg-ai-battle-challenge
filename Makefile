# PTCG battle agent — developer convenience targets.
# Everything runs inside a local .venv so your global Python stays clean.

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
KAGGLE := $(VENV)/bin/kaggle
COMP := pokemon-tcg-ai-battle-challenge-strategy

# Quality-gate tools (dev-only; see requirements.txt + pyproject.toml).
RUFF := $(VENV)/bin/ruff
BLACK := $(VENV)/bin/black
ISORT := $(VENV)/bin/isort
MYPY := $(VENV)/bin/mypy
BANDIT := $(VENV)/bin/bandit
PIPAUDIT := $(VENV)/bin/pip-audit

.DEFAULT_GOAL := help
.PHONY: help setup data test lint format typecheck audit security ci deck verify tournament soak bundle tune check submit freeze clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(PY): ## Create the virtualenv if missing
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(PY) ## Create .venv and install all dependencies
	$(PIP) install -r requirements.txt
	@echo "\nEnvironment ready. Activate with:  source $(VENV)/bin/activate"

data: ## Download the competition card data into data/ (~320 MB incl. PDFs)
	$(KAGGLE) competitions download -c $(COMP) -f EN_Card_Data.csv -p data/
	$(KAGGLE) competitions download -c $(COMP) -f JP_Card_Data.csv -p data/
	$(KAGGLE) competitions download -c $(COMP) -f "Card_ID List_EN.pdf" -p data/
	$(KAGGLE) competitions download -c $(COMP) -f "Card_ID List_JP.pdf" -p data/

test: ## Run the test suite (integration tests skip without local data/)
	$(PY) -m pytest tests/ -q

# --- Quality gate (dev-only; none of this ships in dist/main.py) --------------
format: ## Auto-fix lint, sort imports, format (ruff --fix + isort + black)
	$(RUFF) check --fix .
	$(ISORT) .
	$(BLACK) .

lint: ## Check lint + import order + formatting, no writes (CI-safe)
	$(RUFF) check .
	$(ISORT) --check-only .
	$(BLACK) --check .

typecheck: ## Static type-check the agent + deckbuilder packages (mypy)
	$(MYPY) ptcg_bot deckbuilder

audit: ## Enforce SECURITY.md — no network/subprocess imports in the agent
	@if grep -rEn '^[[:space:]]*(import|from)[[:space:]]+(socket|urllib|requests|subprocess)' ptcg_bot --include='*.py'; then \
		echo "[audit] FORBIDDEN import in ptcg_bot/ — the agent must be pure-stdlib with no egress (SECURITY.md)"; exit 1; \
	else echo "[audit] ok — no socket/urllib/requests/subprocess imports in ptcg_bot/"; fi

security: ## Static (bandit) + dependency-CVE (pip-audit) scan; pip-audit needs network
	$(BANDIT) -q -r ptcg_bot deckbuilder
	$(PIPAUDIT) -r requirements.txt

ci: lint typecheck audit security test ## Engine-free gate CI runs (lint + types + audit + security + tests)

deck: ## Build the submission deck.csv into dist/ (offline deckbuilder)
	$(PY) -m tools.build_deck

# --- Harness targets (tools land per the plan; guarded until they exist) -----
verify: ## Cross-check sim math + rule variant against the LIVE engine (plan §W2-3)
	@test -f tools/verify_env.py && $(PY) tools/verify_env.py || echo "[pending] tools/verify_env.py — wired once the simulator is local (plan §W2)"

tournament: ## Win-rate matrix vs baselines + deck-vs-deck (plan §W4)
	@test -f tools/tournament.py && $(PY) tools/tournament.py $(ARGS) || echo "[pending] tools/tournament.py — plan §W4"

soak: ## Robustness soak: no crash / no timeout over many seeds (plan §W4)
	@test -f tools/soak.py && $(PY) tools/soak.py $(or $(ARGS),--seeds 25) || echo "[pending] tools/soak.py — plan §W4"

tune: ## Coordinate-descent weight tuner -> tools/best_config.env (plan §W6)
	@test -f tools/tune.py && $(PY) -u tools/tune.py $(ARGS) || echo "[pending] tools/tune.py — plan §W6"

bundle: ## Build dist/main.py (freeze weights) + submission tarballs (plan §W4)
	@test -f tools/bundle.py && $(PY) tools/bundle.py || echo "[pending] tools/bundle.py — plan §W4"

check: test verify bundle ## Full local gate: tests + engine verify + bundle

submit: bundle ## Upload the agent to Kaggle (outward-facing — see SUBMIT.md)
	@echo "Submission is outward-facing; run the exact command in SUBMIT.md yourself."

freeze: ## Pin the full resolved dependency set to requirements.lock.txt
	$(PIP) freeze > requirements.lock.txt
	@echo "wrote requirements.lock.txt"

clean: ## Remove caches and build artifacts (keeps .venv and data/)
	rm -rf dist .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
