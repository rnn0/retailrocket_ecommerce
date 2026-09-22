.PHONY: install pipeline api test docker

install:
	pip install -r requirements.txt

pipeline:
	python -m src.pipeline

api:
	uvicorn api.main:app --reload --port 8000

test:
	pytest -v

docker:
	docker compose up --build
