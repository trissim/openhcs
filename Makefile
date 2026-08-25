.PHONY: help install-hooks status

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-hooks: ## Install diagnostic Git hooks for submodule drift
	@git config core.hooksPath hooks
	@echo "Git hooks installed; checkout and merge report submodule drift without changing it."

status: ## Show recursive submodule status
	git submodule status --recursive
