.PHONY: help build run stop clean test setup

help:
	@echo "Face Recognition Pipeline - Make Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup      - Setup local environment (venv + dependencies)"
	@echo "  make run        - Run application locally"
	@echo "  make test       - Run tests"
	@echo "  make build      - Build Docker image"
	@echo "  make docker-run - Run application in Docker"
	@echo "  make stop       - Stop Docker container"
	@echo "  make clean      - Clean up generated files and containers"
	@echo ""

setup:
	@echo "Setting up local environment..."
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	mkdir -p dataset models
	@echo "Setup complete! Activate with: source venv/bin/activate"

run:
	@echo "Running application..."
	python main.py

test:
	@echo "Running tests..."
	python tests/test_utils.py

build:
	@echo "Building Docker image..."
	docker build -t face-recognition:latest .

docker-run:
	@echo "Running in Docker..."
	chmod +x docker-run.sh
	./docker-run.sh

docker-compose-up:
	@echo "Starting with docker-compose..."
	xhost +local:docker
	docker-compose up -d
	docker-compose logs -f

docker-compose-down:
	@echo "Stopping docker-compose..."
	docker-compose down
	xhost -local:docker

stop:
	@echo "Stopping container..."
	docker stop face-recognition-pipeline 2>/dev/null || true
	docker-compose down 2>/dev/null || true

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf *.pyc src/*.pyc tests/*.pyc
	rm -rf venv
	docker stop face-recognition-pipeline 2>/dev/null || true
	docker rm face-recognition-pipeline 2>/dev/null || true
	docker-compose down 2>/dev/null || true
	@echo "Cleanup complete!"

clean-data:
	@echo "WARNING: This will delete all captured data and trained models!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf dataset/* models/*; \
		echo "Data cleaned!"; \
	else \
		echo "Cancelled."; \
	fi
