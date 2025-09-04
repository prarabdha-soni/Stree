# Installation Guide

This guide will walk you through setting up the complete Stree2.2 + VibeVoice-7B MoE pipeline for perfect lip-sync video generation.

## 🚀 **Quick Start (5 minutes)**

```bash
# 1. Clone the repository
git clone https://github.com/prarabdha-soni/Stree.git
cd Stree

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install FFmpeg
brew install ffmpeg  # macOS
# OR
sudo apt install ffmpeg  # Ubuntu/Debian

# 4. Clone VibeVoice-7B
git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git

# 5. Test the setup
python -c "from stree import create_moe_pipeline; print('✅ Setup complete!')"
```

## 📋 **System Requirements**

### **Minimum Requirements**
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+
- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### **Recommended Requirements**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, RTX 3080, RTX 4090)
- **CUDA**: 11.8 or 12.1
- **RAM**: 32GB or higher
- **Storage**: SSD with 50GB+ free space

### **CPU-Only Setup**
- **CPU**: Intel i7/AMD Ryzen 7 or higher
- **RAM**: 32GB minimum
- **Storage**: SSD recommended

## 🛠️ **Step-by-Step Installation**

### **Step 1: Environment Setup**

#### **Option A: Using Conda (Recommended)**
```bash
# Create new conda environment
conda create -n stree python=3.9
conda activate stree

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### **Option B: Using Virtual Environment**
```bash
# Create virtual environment
python3 -m venv stree_env
source stree_env/bin/activate  # Linux/macOS
# OR
stree_env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### **Step 2: Install Dependencies**

```bash
# Install core dependencies
pip install -r requirements.txt

# Install additional development dependencies (optional)
pip install pytest black isort yapf
```

### **Step 3: Install FFmpeg**

#### **macOS**
```bash
# Using Homebrew
brew install ffmpeg

# Verify installation
ffmpeg -version
```

#### **Ubuntu/Debian**
```bash
# Update package list
sudo apt update

# Install FFmpeg
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

#### **Windows**
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH environment variable
4. Restart your terminal and verify: `ffmpeg -version`

### **Step 4: Setup VibeVoice-7B**

```bash
# Clone VibeVoice-7B repository
git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git

# Navigate to VibeVoice-7B directory
cd VibeVoice-7B

# Install VibeVoice-7B dependencies
pip install -r requirements.txt

# Return to main directory
cd ..
```

### **Step 5: Download Stree2.2 Models**

#### **Automatic Download (Recommended)**
The models will be automatically downloaded when you first run the pipeline. This requires:
- Stable internet connection
- Sufficient storage space (5-10GB)
- Patience for the first run

#### **Manual Download**
If you prefer manual download:

1. **Stree2.2-I2V-A14B**: Download from official sources
2. **Stree2.2-VAE**: Download VAE checkpoints
3. Place models in `./checkpoints/` directory

### **Step 6: Verify Installation**

```bash
# Test basic imports
python -c "
import torch
import librosa
import cv2
from scipy import signal
print('✅ All dependencies imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name()}')
"

# Test Stree package
python -c "
from stree import create_moe_pipeline
print('✅ Stree package imported successfully')
"
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Set CUDA device (optional)
export CUDA_VISIBLE_DEVICES=0

# Set model cache directory
export HF_HOME=./models
export TRANSFORMERS_CACHE=./models

# Set temporary directory
export TMPDIR=./temp
```

### **Model Paths**
```bash
# Create necessary directories
mkdir -p checkpoints
mkdir -p models
mkdir -p temp
mkdir -p output
```

## 🧪 **Testing Your Installation**

### **Test 1: Basic Functionality**
```bash
# Test MoE pipeline creation
python -c "
from stree import create_moe_pipeline
print('✅ MoE pipeline creation successful')
"
```

### **Test 2: Audio Processing**
```bash
# Test audio analysis capabilities
python -c "
import librosa
import numpy as np
print('✅ Audio processing libraries working')
"
```

### **Test 3: Video Generation (GPU Required)**
```bash
# Test video generation (will fail on CPU-only systems)
python -c "
from stree.image2video import StreeI2V
print('✅ Video generation module imported')
"
```

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. CUDA Not Available**
```bash
# Error: Torch not compiled with CUDA enabled
# Solution: Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# OR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **2. FFmpeg Not Found**
```bash
# Error: ffmpeg: command not found
# Solution: Install FFmpeg
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu
```

#### **3. Out of Memory**
```bash
# Error: CUDA out of memory
# Solution: Reduce model size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

#### **4. Missing Dependencies**
```bash
# Error: No module named 'librosa'
# Solution: Install missing packages
pip install librosa opencv-python scipy
```

#### **5. VibeVoice-7B Issues**
```bash
# Error: VibeVoice-7B not found
# Solution: Reinstall VibeVoice-7B
rm -rf VibeVoice-7B
git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git
cd VibeVoice-7B
pip install -r requirements.txt
cd ..
```

### **Performance Optimization**

#### **GPU Memory Management**
```bash
# Set PyTorch memory fraction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use mixed precision for faster inference
export TORCH_AMP_ENABLED=1
```

#### **CPU Optimization**
```bash
# Set number of threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Use Intel MKL for better performance
conda install mkl mkl-include
```

## 📱 **Mobile/Cloud Deployment**

### **Docker Setup**
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

### **Google Colab**
```python
# Install dependencies in Colab
!pip install -r requirements.txt
!apt-get install ffmpeg

# Clone repositories
!git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git
!git clone https://github.com/prarabdha-soni/Stree.git

# Test installation
from stree import create_moe_pipeline
print("✅ Installation successful!")
```

## 🔍 **Verification Checklist**

- [ ] Python 3.9+ installed
- [ ] PyTorch with CUDA support (if using GPU)
- [ ] All Python dependencies installed
- [ ] FFmpeg installed and accessible
- [ ] VibeVoice-7B cloned and dependencies installed
- [ ] Stree package imports successfully
- [ ] Basic MoE pipeline creation works
- [ ] Audio processing libraries functional
- [ ] Model directories created
- [ ] Environment variables set (optional)

## 📞 **Getting Help**

If you encounter issues during installation:

1. **Check the troubleshooting section above**
2. **Search existing GitHub issues**
3. **Create a new issue with detailed error information**
4. **Include your system specifications and error logs**

## 🎉 **Next Steps**

After successful installation:

1. **Read the main README.md** for usage examples
2. **Try the basic examples** in the `examples/` directory
3. **Experiment with different sync quality levels**
4. **Customize the configuration** for your needs
5. **Join the community** and share your results!

---

**Happy lip-sync video generation! 🎬✨** 