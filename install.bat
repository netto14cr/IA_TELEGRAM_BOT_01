@echo off

REM Create a virtual environment named 'env'
python -m venv env

REM Activate the virtual environment
call env\Scripts\activate

REM Install required libraries
pip install -r requirements.txt
