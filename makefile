# Variables
VENV_NAME := .venv
VENV_ACTIVATE := $(VENV_NAME)/bin/activate
REQUIREMENTS := requirements.txt

.PHONY: venv install

# Create a virtual environment if it doesn't exist
venv:
	@echo "Creating virtual environment..."
	@test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)

# Activate the virtual environment
activate:
	@echo "Activating virtual environment..."
	@test -f $(VENV_ACTIVATE) && source $(VENV_ACTIVATE)

# Install dependencies using pip
install:
	@echo "Installing dependencies..."
	@test -f $(REQUIREMENTS) && $(VENV_NAME)/bin/pip install -r $(REQUIREMENTS)

# Run all steps: create virtual environment, activate it, and install dependencies
setup: venv activate install
