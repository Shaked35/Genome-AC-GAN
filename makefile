# Set the Python interpreter
PYTHON = python3

# Virtual environment directory
VENV_DIR = venv
# Path to requirements file
REQUIREMENTS_FILE = requirements.txt

# Create a virtual environment
venv:
	$(PYTHON) -m venv $(VENV_DIR)

# Activate the virtual environment
activate:
	@echo "Activating virtual environment..."
	@. $(VENV_DIR)/bin/activate

# Install requirements using pip
install:
	$(PYTHON) -m pip install -r $(REQUIREMENTS_FILE)
	$(PYTHON) pip install tensorflow==2.7.0

# Clean up
clean:
	rm -rf $(VENV_DIR)

# Set up the virtual environment and install requirements
setup: venv activate install

.PHONY: venv activate install clean setup
