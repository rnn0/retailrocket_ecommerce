.PHONY: install pipeline tune api test lint docker

install:
	pip install -r requirements-dev.txt

pipeline:
	python -m src.pipeline

tune:
	python -m src.experiments.tune_als

api:
	uvicorn api.main:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check .

docker:
	docker compose up --build
