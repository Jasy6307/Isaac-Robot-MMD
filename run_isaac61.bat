@echo off
setlocal EnableExtensions
set "ISAAC61_ROOT=I:\isaac61"
set "ISAACSIM_PATH=%ISAAC61_ROOT%\IsaacSim"
set "ISAACLAB_PATH=%ISAAC61_ROOT%\IsaacLab"
for %%D in (tmp logs data documents config cache\pip cache\uv cache\huggingface cache\torch cache\cuda cache\warp cache\kit) do if not exist "%ISAAC61_ROOT%\%%D" mkdir "%ISAAC61_ROOT%\%%D"
set "TEMP=%ISAAC61_ROOT%\tmp"
set "TMP=%TEMP%"
set "PIP_CACHE_DIR=%ISAAC61_ROOT%\cache\pip"
set "UV_CACHE_DIR=%ISAAC61_ROOT%\cache\uv"
set "HF_HOME=%ISAAC61_ROOT%\cache\huggingface"
set "TORCH_HOME=%ISAAC61_ROOT%\cache\torch"
set "CUDA_CACHE_PATH=%ISAAC61_ROOT%\cache\cuda"
set "XDG_CACHE_HOME=%ISAAC61_ROOT%\cache"
set "WARP_CACHE_PATH=%ISAAC61_ROOT%\cache\warp"
set "OMNI_KIT_ACCEPT_EULA=yes"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "VIRTUAL_ENV="
set "PYTHONEXE="
set "PYTHONHOME="
set "PYTHONPATH=%~dp0"
if not exist "%ISAACSIM_PATH%\python.bat" (
    echo Isaac Sim 6.1 is not installed at %ISAACSIM_PATH%.
    exit /b 1
)
cd /d "%~dp0"
if "%~1"=="" (
    call "%ISAACSIM_PATH%\python.bat" source\train_workflow\smoke_isaac61.py --viz kit
) else (
    call "%ISAACSIM_PATH%\python.bat" %*
)
exit /b %ERRORLEVEL%
