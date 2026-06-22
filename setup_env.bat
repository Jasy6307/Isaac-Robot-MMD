@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

:: ===============================================
:: NVIDIA Isaac Sim and Isaac Lab Setup Script
:: Isaac Sim v5.1.0 + Isaac Lab v2.3.0
:: ===============================================

:: === 记录脚本所在目录（去掉末尾反斜杠）===
set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"
echo 一键安装环境脚本所在目录: %REPO_ROOT%

:: === 配置参数 ===
set "ISAACSIM_VERSION=v5.1.0"
set "ISAACLAB_VERSION=v2.3.0"
:: 可通过环境变量覆盖，例如: set ISAAC_WORKSPACE=D:\isaac_workspace
if not defined ISAAC_WORKSPACE set "WORKSPACE=%USERPROFILE%\isaac_workspace"
if defined ISAAC_WORKSPACE set "WORKSPACE=%ISAAC_WORKSPACE%"
set "PYTHON_VERSION=3.11"
set "ENV_NAME=env_isaaclab_mmd"
set "ISAAC_SIM_ZIP=isaac-sim-standalone-5.1.0-windows-x86_64.zip"
set "ISAAC_SIM_DIR=isaac-sim-standalone-5.1.0-windows-x86_64"
set "ISAAC_SIM_URL=https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-windows-x86_64.zip"

:: === 检查管理员权限（创建符号链接需要，此处使用目录联接 /J 可免管理员）===
:: 如需使用 /D 符号链接，请以管理员身份运行

:: === [1/7] 系统依赖（可选：Git、CMake）===
echo [1/7] 检查系统依赖...
where git >nul 2>&1
if errorlevel 1 (
    echo 未检测到 Git，请先安装 Git for Windows 或运行: winget install Git.Git
    pause
)
where cmake >nul 2>&1
if errorlevel 1 (
    echo 未检测到 CMake，建议安装: winget install Kitware.CMake
)

:: === 创建工作目录 ===
echo 工作目录: %WORKSPACE%
if not exist "%WORKSPACE%" mkdir "%WORKSPACE%"
cd /d "%WORKSPACE%"

:: === [2/7] 下载并解压 Isaac Sim 预编译版本 ===
echo [2/7] 下载并解压 Isaac Sim %ISAACSIM_VERSION% 预编译版本...

:: 检查工作目录是否已有同名 zip 或目录，有则跳过下载
set "SKIP_DOWNLOAD=0"
if exist "%WORKSPACE%\%ISAAC_SIM_ZIP%" (
    set "SKIP_DOWNLOAD=1"
    echo 已存在 %ISAAC_SIM_ZIP%，跳过下载
)
if exist "%WORKSPACE%\%ISAAC_SIM_DIR%" (
    set "SKIP_DOWNLOAD=1"
    echo 已存在 %ISAAC_SIM_DIR% 目录，跳过下载
)
if exist "%WORKSPACE%\IsaacSim\python.bat" (
    set "SKIP_DOWNLOAD=1"
    echo 已存在 IsaacSim 安装，跳过下载
)
if exist "%WORKSPACE%\IsaacSim\python.exe" (
    set "SKIP_DOWNLOAD=1"
    echo 已存在 IsaacSim 安装，跳过下载
)

if "!SKIP_DOWNLOAD!"=="0" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%ISAAC_SIM_URL%' -OutFile '%WORKSPACE%\%ISAAC_SIM_ZIP%' -UseBasicParsing }"
    if errorlevel 1 (
        echo 下载失败，请检查网络或手动下载: %ISAAC_SIM_URL%
        pause
        exit /b 1
    )
)

:: 解压：仅当 IsaacSim 尚未就绪时执行
set "NEED_EXTRACT=0"
if not exist "%WORKSPACE%\IsaacSim\python.bat" if not exist "%WORKSPACE%\IsaacSim\python.exe" set "NEED_EXTRACT=1"
if "!NEED_EXTRACT!"=="1" (
    REM 若已有同名目录（手动解压），直接重命名为 IsaacSim 使用
    if exist "%WORKSPACE%\%ISAAC_SIM_DIR%\python.bat" (
        if exist "%WORKSPACE%\IsaacSim" rmdir /s /q "%WORKSPACE%\IsaacSim"
        ren "%WORKSPACE%\%ISAAC_SIM_DIR%" IsaacSim
        echo 已使用现有目录 %ISAAC_SIM_DIR% 作为 IsaacSim
    ) else if exist "%WORKSPACE%\%ISAAC_SIM_DIR%\python.exe" (
        if exist "%WORKSPACE%\IsaacSim" rmdir /s /q "%WORKSPACE%\IsaacSim"
        ren "%WORKSPACE%\%ISAAC_SIM_DIR%" IsaacSim
        echo 已使用现有目录 %ISAAC_SIM_DIR% 作为 IsaacSim
    ) else if exist "%WORKSPACE%\%ISAAC_SIM_ZIP%" (
        if not exist "%WORKSPACE%\IsaacSim" mkdir "%WORKSPACE%\IsaacSim"
        echo 正在解压...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%WORKSPACE%\%ISAAC_SIM_ZIP%' -DestinationPath '%WORKSPACE%\IsaacSim' -Force"
        if errorlevel 1 (
            echo 解压失败
            pause
            exit /b 1
        )
        REM 若解压后根目录是 IsaacSim 下的子文件夹，需调整结构
        if exist "%WORKSPACE%\IsaacSim\%ISAAC_SIM_DIR%" (
            move "%WORKSPACE%\IsaacSim\%ISAAC_SIM_DIR%\*" "%WORKSPACE%\IsaacSim\"
            rmdir "%WORKSPACE%\IsaacSim\%ISAAC_SIM_DIR%"
        )
        if exist "%WORKSPACE%\IsaacSim\isaac-sim-standalone-5.1.0" (
            move "%WORKSPACE%\IsaacSim\isaac-sim-standalone-5.1.0\*" "%WORKSPACE%\IsaacSim\"
            rmdir "%WORKSPACE%\IsaacSim\isaac-sim-standalone-5.1.0"
        )
    ) else (
        echo 错误: 未找到 %ISAAC_SIM_ZIP% 或 %ISAAC_SIM_DIR%，无法解压
        pause
        exit /b 1
    )
) else (
    echo IsaacSim 已就绪，跳过解压
)

:: === 设置并验证环境变量 ===
echo [4/7] 设置并验证 Isaac Sim 环境变量...
set "ISAACSIM_PATH=%WORKSPACE%\IsaacSim"
:: Windows 下 Isaac Sim 提供的 Python 入口多为 python.bat 或 python.exe
if exist "%ISAACSIM_PATH%\python.bat" (
    set "ISAACSIM_PYTHON_EXE=%ISAACSIM_PATH%\python.bat"
) else if exist "%ISAACSIM_PATH%\python.exe" (
    set "ISAACSIM_PYTHON_EXE=%ISAACSIM_PATH%\python.exe"
) else (
    echo 未找到 Isaac Sim 的 python.bat/python.exe，请检查解压路径
    pause
    exit /b 1
)
call "%ISAACSIM_PYTHON_EXE%" -c "print('Isaac Sim configuration is now complete.')"
if errorlevel 1 (
    echo Isaac Sim Python 验证失败
    pause
    exit /b 1
)

:: === [5/7] 克隆 Isaac Lab ===
echo [5/7] 克隆 Isaac Lab %ISAACLAB_VERSION%...
cd /d "%WORKSPACE%"
if not exist "IsaacLab" (
    git clone https://github.com/isaac-sim/IsaacLab.git
)
cd IsaacLab
git fetch --tags
git checkout tags/v2.3.0 -b v2.3.0 2>nul || git checkout v2.3.0
cd ..

:: === [6/7] 建立 Isaac Sim 目录联接===
echo [6/7] 建立 Isaac Sim 目录联接...
cd /d "%WORKSPACE%\IsaacLab"
if exist _isaac_sim (
    rmdir _isaac_sim 2>nul || del /q _isaac_sim 2>nul
)
mklink /J _isaac_sim "%ISAACSIM_PATH%"
cd ..

:: === [7/7] 使用 Conda 创建环境并安装 Isaac Lab 依赖 ===
echo [7/7] 创建 Python 虚拟环境并安装依赖...
cd /d "%WORKSPACE%\IsaacLab"

:: 尝试找到 conda（Anaconda/Miniconda）
set "CONDA_ACTIVATE="
if defined CONDA_PREFIX (
    set "CONDA_ACTIVATE=%CONDA_PREFIX%\..\Scripts\activate.bat"
)
if not exist "!CONDA_ACTIVATE!" (
    if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
)
if not exist "!CONDA_ACTIVATE!" (
    if exist "%ProgramData%\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%ProgramData%\miniconda3\Scripts\activate.bat"
)
if not exist "!CONDA_ACTIVATE!" (
    if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
)
if not exist "!CONDA_ACTIVATE!" (
    where conda >nul 2>&1
    if errorlevel 1 (
        echo 未找到 Conda。请先安装 Miniconda/Anaconda 并在此脚本所在目录从 "Anaconda Prompt" 运行本脚本。
        pause
        exit /b 1
    )
)

:: 创建 Conda 环境（使用 isaaclab.bat）
call isaaclab.bat -c %ENV_NAME%
if errorlevel 1 (
    echo isaaclab.bat -c 创建环境失败
    pause
    exit /b 1
)

:: 激活环境并安装依赖（先激活 base 使 conda 可用）
if defined CONDA_ACTIVATE if exist "!CONDA_ACTIVATE!" (
    call "!CONDA_ACTIVATE!"
)
call conda activate %ENV_NAME%

:: 预装 setuptools<70 和 flatdict（setuptools 70+ 移除了 pkg_resources，flatdict 构建会失败）
echo 预装 setuptools 和 flatdict...
python -m pip install "setuptools<70" -q
python -m pip install flatdict==4.0.1 --no-build-isolation -q

call isaaclab.bat --install
if errorlevel 1 (
    echo isaaclab.bat --install 失败
    pause
    exit /b 1
)
:: 解决 ray 与 click 8.3.* 的依赖冲突（ray 要求 click!=8.3.*）
python -m pip install "click>=7.0,!=8.3.*" -q
:: 修复 numpy 和 h5py 版本（需在 isaaclab 安装后执行，覆盖可能被升级的不兼容版本）
echo 修复 numpy 和 h5py 版本...
python -m pip install numpy==1.26.4 -q
python -m pip install h5py==3.10.0 -q
cd ..

echo Isaac Sim %ISAACSIM_VERSION%, Isaac Lab %ISAACLAB_VERSION% 均已安装完成。

:: === 创建 isaac_workspace 目录联接到脚本所在目录 ===
echo [额外步骤1/2] 创建 isaac_workspace 目录联接到脚本所在目录...
cd /d "%REPO_ROOT%"
if exist "isaac_workspace" (
    rmdir "isaac_workspace" 2>nul || del /q "isaac_workspace" 2>nul
)
mklink /J isaac_workspace "%WORKSPACE%"
echo %WORKSPACE% 已联接到 %REPO_ROOT%\isaac_workspace

:: === 安装 source 包到 Isaac Lab 环境 ===
echo [额外步骤2/2] 将 source 安装到 Isaac Lab 环境...
if defined CONDA_ACTIVATE if exist "!CONDA_ACTIVATE!" (call "!CONDA_ACTIVATE!")
call conda activate %ENV_NAME%
cd /d "%REPO_ROOT%"
if exist "setup.py" (
    python -m pip install -e .
    echo source 已安装到 %ENV_NAME% 环境
) else if exist "pyproject.toml" (
    python -m pip install -e .
    echo source 已安装到 %ENV_NAME% 环境
) else (
    echo 注意: 当前目录无 setup.py 或 pyproject.toml，跳过 source 安装
)

echo.
echo ============================================
echo 所有安装步骤完成！
echo ============================================
echo.
echo 验证安装:
echo   conda activate %ENV_NAME%
echo   %WORKSPACE%\IsaacLab\isaaclab.bat -p %WORKSPACE%\IsaacLab\scripts\tutorials\00_sim\create_empty.py
echo.
echo 运行 source 回放:
echo   conda activate %ENV_NAME%
echo   %WORKSPACE%\IsaacLab\isaaclab.bat -p %REPO_ROOT%\source\train_workflow\g1_vmd_0_replay.py
echo.
pause
