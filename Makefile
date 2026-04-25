.PHONY: venv setup-venv clean-venv install-deps train predict streamlit-run flask-run fastapi-run help

help:
	@echo "Available commands:"
	@echo "  make venv           - Create and setup virtual environment"
	@echo "  make setup-venv     - Alias for venv"
	@echo "  make clean-venv     - Remove virtual environment"
	@echo "  make install-deps   - Install Python dependencies"
	@echo "  make train          - Train the model"
	@echo "  make predict        - Generate predictions"
	@echo "  make streamlit-run  - Run Streamlit UI"
	@echo "  make flask-run      - Run Flask API"
	@echo "  make fastapi-run    - Run FastAPI server"

venv: clean-venv
	python -m venv .venv
	.venv\Scripts\python.exe -m pip install --upgrade pip
	.venv\Scripts\pip.exe install -r requirements.txt
	@echo "Virtual environment created and dependencies installed!"

setup-venv: venv

clean-venv:
	@if exist .venv rmdir /s /q .venv
	@echo "Virtual environment removed"

install-deps:
	.venv\Scripts\pip.exe install -r requirements.txt

train:
	.venv\Scripts\python.exe src/train.py

predict:
	.venv\Scripts\python.exe src/predict.py

streamlit-run:
	.venv\Scripts\python.exe -m streamlit run src/app_streamlit.py

flask-run:
	.venv\Scripts\python.exe src/app_flask.py

fastapi-run:
	.venv\Scripts\python.exe src/app_fastapi.py
