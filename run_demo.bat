@echo off
REM Roda a API + vitrine com UM duplo clique (Windows). Requer Python 3.12 e a pasta models/ treinada.
cd /d "%~dp0"
where py >nul 2>nul || (echo Python nao encontrado. Instale o Python 3.12 em python.org marcando "Add python.exe to PATH". & pause & exit /b 1)
py -3.12 --version >nul 2>nul || (echo Python 3.12 nao encontrado. Instale o 3.12 em python.org/downloads e rode de novo. & pause & exit /b 1)
if not exist models\metadata.json (echo Pasta models\ vazia: use o pacote com modelos ou rode: python -m src.pipeline & pause & exit /b 1)
if not exist .venv\Scripts\python.exe (
  echo Criando ambiente e instalando dependencias ^(so na primeira vez, alguns minutos^)...
  py -3.12 -m venv .venv || (pause & exit /b 1)
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Falha na instalacao. Tire um print deste erro. & pause & exit /b 1)
)
start "" http://localhost:8000/
.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
pause
