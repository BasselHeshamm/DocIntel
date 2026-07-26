@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
set GEMINI_API_KEY=your-api-key-here
uvicorn api:app --reload