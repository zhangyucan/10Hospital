#!/bin/bash
# 自动安装脚本 for Ubuntu/Debian

set -e  # 遇到错误立即退出

echo "=========================================="
echo "10Hospital 项目安装脚本"
echo "=========================================="
echo ""

# 检查是否在虚拟环境中
if [[ -z "${VIRTUAL_ENV}" ]] && [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "警告: 未检测到虚拟环境"
    echo "建议使用虚拟环境安装。是否继续? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "安装已取消"
        exit 1
    fi
fi

# 检查操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "检测到 Linux 系统"
    
    # 检查 CMake
    if ! command -v cmake &> /dev/null; then
        echo "CMake 未安装，正在安装..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y cmake build-essential
        elif command -v yum &> /dev/null; then
            sudo yum install -y cmake gcc gcc-c++
        else
            echo "错误: 无法自动安装 CMake，请手动安装"
            exit 1
        fi
    else
        echo "✓ CMake 已安装: $(cmake --version | head -n1)"
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "检测到 macOS 系统"
    
    # 检查 Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Homebrew 未安装，请先安装 Homebrew:"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    
    # 检查 CMake
    if ! command -v cmake &> /dev/null; then
        echo "CMake 未安装，正在安装..."
        brew install cmake
    else
        echo "✓ CMake 已安装: $(cmake --version | head -n1)"
    fi
else
    echo "不支持的操作系统: $OSTYPE"
    echo "请参考 INSTALL.md 手动安装"
    exit 1
fi

# 升级 pip
echo ""
echo "升级 pip..."
pip install --upgrade pip

# 安装 cmake (Python 包)
echo ""
echo "安装 cmake Python 包..."
pip install cmake

# 安装依赖
echo ""
echo "安装项目依赖 (包含人脸检测)..."
pip install -r requirements-full.txt

# 验证安装
echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python test_face_detection.py

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "启动应用:"
echo "  streamlit run streamlit_app.py"
echo ""
