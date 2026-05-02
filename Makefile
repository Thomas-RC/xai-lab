.PHONY: help up down logs ps test fmt lint smoke report report-figures clean install fetch-samples

help:
	@echo "xai-lab - dostepne komendy:"
	@echo "  make install        - instalacja zaleznosci lokalnie (.venv)"
	@echo "  make fetch-samples  - pobierz publiczne obrazy testowe do data/samples/"
	@echo "  make up        - uruchom apke w Dockerze (xai.local.pl)"
	@echo "  make down      - zatrzymaj apke"
	@echo "  make logs      - logi apki"
	@echo "  make ps        - status kontenerow"
	@echo "  make test      - pytest"
	@echo "  make smoke     - smoke test (1 obraz, 6 metod)"
	@echo "  make fmt       - ruff format + fix"
	@echo "  make lint      - ruff check + mypy"
	@echo "  make clean     - usun cache i artefakty"

install:
	python3.12 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
	.venv/bin/pip install -e ".[dev]"

fetch-samples:
	.venv/bin/python -m scripts.fetch_samples

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200 app

ps:
	docker compose ps

test:
	.venv/bin/pytest -v

smoke:
	.venv/bin/python -m scripts.smoke_app

report-figures:
	.venv/bin/python -m scripts.run_batch


fmt:
	.venv/bin/ruff format src tests scripts
	.venv/bin/ruff check --fix src tests scripts

lint:
	.venv/bin/ruff check src tests scripts
	.venv/bin/mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/**/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -rf report/figures/*.png report/report.pdf
	rm -rf .cache data/cache
