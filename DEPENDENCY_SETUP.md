# Python Environment Setup

## Recommended version
Use Python 3.11 for TensorFlow compatibility.

## Create a clean virtual environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Install dependencies
```powershell
pip install -r backend/requirements.txt
```

## Verify imports
```powershell
python verify_environment.py
```

## VS Code steps
1. Open the command palette.
2. Run `Python: Select Interpreter`.
3. Choose `.venv\\Scripts\\python.exe`.
4. Run `Developer: Reload Window`.
5. If Pylance still shows stale errors, run `Python: Restart Language Server`.
