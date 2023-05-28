VENV_DIR = venv

.PHONY: install install_conda

install:
	@echo "Activating virtual environment..."
	@. $(VENV_DIR)/bin/activate && \
		pip install -r requirements.txt
	@echo "Installation complete."

install_conda:
	@echo "Activating virtual environment..."
	@. $(VENV_DIR)/bin/activate && \
		conda install --file requirements.txt
	@echo "Installation complete."

$(VENV_DIR):
	@echo "Creating virtual environment..."
	@python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created."

clean:
	rm -rf $(VENV_DIR)
