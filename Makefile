.PHONY: help install test quality quality-fix clean pre-commit ingest dev git-status docker-up docker-down

SHELL := /bin/bash

RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m

help:
	@echo "Multimodal RAG - Support Knowledge Base"
	@echo "========================================"
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run test suite with coverage"
	@echo "  make quality      - Code quality checks (ruff, mypy)"
	@echo "  make quality-fix  - Auto-fix linting issues"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make pre-commit   - Full pre-commit validation"
	@echo ""
	@echo "Environment:"
	@echo "  make dev          - Configure development environment"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make docker-up    - Start Weaviate (Docker)"
	@echo "  make docker-down  - Stop Weaviate (Docker)"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make ingest       - Run ingestion pipeline (YouTube + web)"
	@echo ""
	@echo "Application:"
	@echo "  make run          - Start Gradio chat interface"
	@echo ""
	@echo "Git:"
	@echo "  make git-status   - Show git overview"

install:
	@echo -e "$(BLUE)📦 Installing dependencies...$(NC)"
	@uv sync
	@echo -e "$(GREEN)✅ Dependencies installed$(NC)"

test:
	@echo -e "$(BLUE)🧪 Running test suite...$(NC)"
	@uv run pytest tests/ -v --cov=src/multimodal_rag
	@echo -e "$(GREEN)✅ Tests complete$(NC)"

quality:
	@echo -e "$(BLUE)🎨 Running code quality checks...$(NC)"
	@uv run ruff check src/ tests/
	@uv run mypy src/
	@echo -e "$(GREEN)✅ Quality checks complete$(NC)"

quality-fix:
	@echo -e "$(BLUE)🔧 Auto-fixing code quality issues...$(NC)"
	@uv run ruff check src/ tests/ --fix
	@echo -e "$(GREEN)✅ Auto-fixable issues resolved$(NC)"

clean:
	@echo -e "$(BLUE)🧹 Cleaning artifacts...$(NC)"
	@rm -rf .venv/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ dist/ *.egg-info/
	@find . -name __pycache__ -delete 2>/dev/null || true
	@echo -e "$(GREEN)✅ Cleanup complete$(NC)"

pre-commit:
	@echo -e "$(CYAN)🔍 Pre-commit Quality Gate$(NC)"
	@echo "=========================="
	@$(MAKE) quality
	@$(MAKE) test
	@echo -e "$(GREEN)✅ All pre-commit checks passed$(NC)"

dev:
	@echo -e "$(BLUE)🛠️ Configuring development environment...$(NC)"
	@if [ ! -f "config/development.env" ]; then \
		echo -e "$(RED)❌ config/development.env not found$(NC)"; \
		exit 1; \
	fi
	@cp config/development.env .env
	@echo -e "$(GREEN)✅ Development environment configured$(NC)"

docker-up:
	@echo -e "$(BLUE)🐳 Starting Weaviate...$(NC)"
	@docker compose up -d
	@echo -e "$(GREEN)✅ Weaviate running at http://localhost:8080$(NC)"

docker-down:
	@echo -e "$(BLUE)🐳 Stopping Weaviate...$(NC)"
	@docker compose down
	@echo -e "$(GREEN)✅ Weaviate stopped$(NC)"

ingest:
	@echo -e "$(BLUE)📥 Running ingestion pipeline...$(NC)"
	@if [ ! -f ".env" ]; then \
		echo -e "$(RED)❌ No .env file found. Run make dev first$(NC)"; \
		exit 1; \
	fi
	@uv run python -m multimodal_rag.ingest
	@echo -e "$(GREEN)✅ Ingestion complete$(NC)"

run:
	@echo -e "$(BLUE)🚀 Starting Multimodal RAG...$(NC)"
	@if [ ! -f ".env" ]; then \
		echo -e "$(RED)❌ No .env file found. Run make dev first$(NC)"; \
		exit 1; \
	fi
	@uv run python -m multimodal_rag.app
	@echo -e "$(GREEN)✅ Application stopped$(NC)"

git-status:
	@echo -e "$(CYAN)📊 Git Status Overview$(NC)"
	@current_branch=$$(git branch --show-current); \
	echo "Current branch: $$current_branch"; \
	if [ "$$current_branch" = "main" ]; then \
		echo -e "$(RED)⚠️  You are on MAIN branch$(NC)"; \
	elif [ "$$current_branch" = "develop" ]; then \
		echo -e "$(GREEN)✅ On develop branch$(NC)"; \
	fi
	@echo "Recent commits:"
	@git log --oneline -5
