#!/usr/bin/env bash

###############################################
# NVIDIA Isaac Sim & Isaac Lab 自动化安装脚本
# Ubuntu 22.04 + Conda
# Isaac Sim v5.1.0 + Isaac Lab v2.3.0
###############################################

set -e

# === 记录脚本所在目录 ===
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "一键安装环境脚本所在目录: ${REPO_ROOT}"

# === 配置参数 ===
ISAACSIM_VERSION="v5.1.0"
ISAACLAB_VERSION="v2.3.0"
WORKSPACE="${ISAAC_WORKSPACE:-${HOME}/isaac_workspace}"
PYTHON_VERSION="3.11"
ENV_NAME="env_isaaclab_mmd"

# === 系统依赖 ===
echo "[1/7] 安装系统依赖..."
sudo apt update
sudo apt install -y git cmake build-essential

# === 创建工作目录 ===
rm -rf ${WORKSPACE}
mkdir -p ${WORKSPACE}
cd ${WORKSPACE}

# Using Isaac Sim prebuild binaries
echo "[2/7] 下载并解压 Isaac Sim ${ISAACSIM_VERSION} 预编译版本..."
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip -d IsaacSim
rm isaac-sim-standalone-5.1.0-linux-x86_64.zip

# === 设置并验证环境变量 ===
echo "[4/7] 设置并验证 Isaac Sim 环境变量..."
export ISAACSIM_PATH="${WORKSPACE}/IsaacSim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
# checks that python path is set correctly
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"

# === 克隆 Isaac Lab ===
echo "[5/7] 克隆 Isaac Lab ${ISAACLAB_VERSION}..."
if [ ! -d "IsaacLab" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git
  cd IsaacLab
  git checkout tags/v2.3.0 -b v2.3.0
  cd ..
fi

# === 建立符号链接 ===
echo "[6/7] 建立 Isaac Sim 符号链接..."
cd IsaacLab
ln -sf ${ISAACSIM_PATH} _isaac_sim
cd ..

# === 使用 Conda 创建 Python 虚拟环境并安装依赖 ===
echo "[7/7] 创建 Python 虚拟环境(Conda版)..."
cd IsaacLab
eval "$(conda shell.bash hook)"
./isaaclab.sh -c ${ENV_NAME}
cd ..

echo "[7/7] 安装 Isaac Lab 依赖..."
cd IsaacLab
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
./isaaclab.sh --install
cd ..

echo "✅ IsaacSim ${ISAACSIM_VERSION}, IsaacLab ${ISAACLAB_VERSION} 均已安装完成！"

# === 创建软链接到脚本所在目录 ===
echo "[额外步骤1/2] 创建 isaac_workspace 软链接到脚本所在目录..."
cd ${REPO_ROOT}
echo "当前目录: $(pwd)"
if [ -L "isaac_workspace" ] || [ -e "isaac_workspace" ]; then
  rm -f isaac_workspace
fi
ln -s ${WORKSPACE} isaac_workspace
echo "✅ ${WORKSPACE} 已软链接到 ${REPO_ROOT}/isaac_workspace"

# === 安装 robot_mmd 包到 Isaac Lab 环境 ===
echo "[额外步骤2/2] 将 robot_mmd 安装到 Isaac Lab 环境..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
cd ${REPO_ROOT}
python3 -m pip install -e .
echo "✅ robot_mmd 已安装到 ${ENV_NAME} 环境"


echo ""
echo "============================================"
echo "✅ 所有安装步骤完成！"
echo "============================================"
echo ""
echo "验证安装:"
echo "  conda activate ${ENV_NAME}"
echo "  ./isaac_workspace/IsaacLab/isaaclab.sh -p ./isaac_workspace/IsaacLab/scripts/tutorials/00_sim/create_empty.py"
echo ""
echo "运行 robot_mmd 回放:"
echo "  conda activate ${ENV_NAME}"
echo "  ./isaac_workspace/IsaacLab/isaaclab.sh -p ${REPO_ROOT}/robot_mmd/train_workflow/g1_vmd_0_replay.py"
echo ""
