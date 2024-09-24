python -m venv .venv || exit /b
CALL .\.venv\Scripts\activate || exit /b
python -m ensurepip
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" || exit /b
