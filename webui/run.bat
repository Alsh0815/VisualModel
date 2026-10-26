@echo off
pushd "%~dp0\.."
echo Starting VisualModel Dataset Studio WebUI...
python webui\server.py %*
popd
pause
