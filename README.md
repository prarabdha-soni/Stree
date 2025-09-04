# Stree2.2 + VibeVoice-7B: Mixture of Experts (MoE) Pipeline

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **Professional AI Pipeline for Text+Image to Video Generation with Perfect Lip-Sync**

A production-ready Mixture of Experts (MoE) pipeline that combines **VibeVoice-7B** for text-to-speech and **Stree2.2-I2V-A14B** for image-to-video generation, with intelligent expert routing and perfect audio-video synchronization.

## 🎯 **What This Does**

**Input**: Text + Image  
**Output**: Perfect lip-sync video with synchronized audio and video

**Example**: Give it "Hello, welcome to our insurance policy!" + a photo of a person → Get a video where the person's mouth moves perfectly with the speech!

## 🚀 **Key Features**

- **🎵 Advanced TTS**: VibeVoice-7B for natural speech generation
- **🎬 Video Generation**: Stree2.2-I2V-A14B for high-quality video creation
- **🔗 Perfect Lip-Sync**: Audio-video synchronization with frame accuracy
- **🧠 Intelligent Routing**: MoE system automatically selects best expert
- **🎯 Multiple Quality Levels**: Basic, enhanced, high, and native sync options
- **🔄 Input Flexibility**: Works with text or direct speech input
- **⚡ Performance Optimized**: GPU acceleration with fallback to CPU

## 🏗️ **Architecture Overview**

```
Text Input → VibeVoice-7B → Audio Analysis → Enhanced Prompts → Stree2.2-I2V-A14B → Perfect Lip-Sync Video
     ↓              ↓              ↓              ↓                    ↓              ↓
  MoE Router → Expert Selection → Audio Features → Video Guidance → Frame Generation → Synchronization
```

## 🛠️ **Installation**

### **Prerequisites**
- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- FFmpeg

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/prarabdha-soni/Stree.git
cd Stree

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (macOS)
brew install ffmpeg

# Install FFmpeg (Ubuntu/Debian)
sudo apt update && sudo apt install ffmpeg
```

### **Model Setup**
```bash
# Clone VibeVoice-7B (for text-to-speech)
git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git

# Download Stree2.2 checkpoints (instructions in INSTALL.md)
# Download Wan2.2-S2V-14B (optional, for native speech-to-video)
```

## 🎬 **Usage Examples**

### **Basic Text-to-Video with Lip-Sync**
```python
from stree import create_moe_pipeline

# Create MoE pipeline
moe = create_moe_pipeline()

# Generate lip-sync video
results = moe.generate(
    text="Hello, this is a perfect lip-sync example!",
    image_path="face.jpg",
    duration=5,
    voice="en_female_1"
)

print(f"Video generated: {results['final_video_path']}")
```

### **High-Quality Lip-Sync with Audio Analysis**
```python
from stree import create_audio_sync_moe_pipeline

# Create advanced audio sync pipeline
audio_sync_moe = create_audio_sync_moe_pipeline()

# Generate with perfect lip-sync
results = audio_sync_moe.generate_with_lip_sync(
    text="Welcome to our comprehensive insurance policy!",
    image_path="agent.jpg",
    duration=8,
    sync_quality="high"
)

print(f"Perfect lip-sync video: {results['final_video_path']}")
```

### **Intelligent Expert Routing with Wan2.2-S2V-14B**
```python
from stree import create_integrated_moe_pipeline

# Create integrated pipeline with native S2V support
integrated_moe = create_integrated_moe_pipeline()

# Let the system choose the best expert automatically
results = integrated_moe.generate_with_smart_routing(
    input_data="Hello, this is perfect lip-sync!",
    image_path="face.jpg",
    duration=5,
    sync_quality="high",
    use_native_if_available=True  # Prefer Wan2.2-S2V-14B when available
)

print(f"Generated using: {results['selected_expert']}")
```

## 🎯 **Command Line Usage**

### **Basic MoE Pipeline**
```bash
python examples/moe_example.py \
    --text "Welcome to our insurance policy!" \
    --image examples/agent.jpg \
    --duration 8 \
    --voice en_female_1
```

### **Advanced Audio Sync**
```bash
python examples/lip_sync_example.py \
    --text "Let me explain the benefits..." \
    --image examples/agent.jpg \
    --duration 10 \
    --sync-quality high
```

### **Integrated MoE with Smart Routing**
```bash
python examples/integrated_moe_example.py \
    --input "Hello, this is perfect lip-sync!" \
    --image examples/face.jpg \
    --duration 5 \
    --sync-quality high \
    --prefer-native
```

## 🧠 **MoE Expert System**

### **Expert 1: VibeVoice-7B**
- **Purpose**: Text-to-speech generation
- **Capabilities**: Natural voice, multiple accents, emotional control
- **Input**: Text strings
- **Output**: High-quality WAV audio

### **Expert 2: Stree2.2-I2V-A14B**
- **Purpose**: Image-to-video generation
- **Capabilities**: 720p@24fps, MoE architecture, motion control
- **Input**: Image + prompt
- **Output**: Video frames

### **Expert 3: Wan2.2-S2V-14B (Optional)**
- **Purpose**: Native speech-to-video
- **Capabilities**: Direct audio-video sync, speech optimization
- **Input**: Audio + image
- **Output**: Synchronized video

### **Expert 4: Audio Sync MoE**
- **Purpose**: Intelligent routing and synchronization
- **Capabilities**: Audio analysis, beat sync, quality control
- **Input**: Audio features + requirements
- **Output**: Expert selection + sync strategy

## 📊 **Performance & Quality**

| Feature | Basic | Enhanced | High | Native |
|---------|-------|----------|------|--------|
| **Lip-Sync** | ✅ Basic | ✅ Good | ✅ Perfect | ✅ Native |
| **Audio Analysis** | ❌ None | ✅ Basic | ✅ Advanced | ✅ Built-in |
| **Motion Guidance** | ❌ None | ✅ Basic | ✅ Beat Sync | ✅ Optimized |
| **Processing Speed** | ⚡ Fast | ⚡ Fast | 🐌 Medium | ⚡ Fast |
| **Quality Control** | ❌ None | ✅ Basic | ✅ Advanced | ✅ Native |

## 🔧 **Configuration**

### **Stree2.2-I2V-A14B Settings**
```python
# Customize video generation
config = {
    "guidance_scale": 7.5,        # How closely to follow prompt
    "num_inference_steps": 50,    # Denoising steps (quality vs. speed)
    "num_frames": 120,            # 5 seconds at 24 FPS
    "fps": 24                     # Frames per second
}
```

### **VibeVoice-7B Settings**
```python
# Voice customization
voice_options = [
    "en_female_1",    # Professional female voice
    "en_male_1",      # Professional male voice
    "en_female_2",    # Casual female voice
    "en_male_2"       # Casual male voice
]
```

## 🚀 **Advanced Features**

### **Audio-Driven Video Generation**
- **Beat Synchronization**: Visual elements align with audio beats
- **Energy-Based Motion**: Motion intensity responds to audio energy
- **Phoneme Awareness**: Lip movements match speech patterns
- **Temporal Alignment**: Frame-accurate synchronization

### **Intelligent Expert Routing**
- **Automatic Selection**: Chooses best expert for each task
- **Performance Monitoring**: Tracks expert success rates
- **Load Balancing**: Distributes tasks efficiently
- **Quality Prediction**: Estimates output quality before generation

### **Customizable Sync Quality**
- **Basic**: Simple audio-video combination
- **Enhanced**: Audio influences video generation
- **High**: Perfect lip-sync with audio analysis
- **Native**: Wan2.2-S2V-14B direct processing

## 📁 **Project Structure**

```
Stree/
├── stree/                          # Core MoE pipeline
│   ├── __init__.py                # Main package exports
│   ├── moe_pipeline.py            # Basic MoE implementation
│   ├── advanced_moe.py            # Advanced MoE with monitoring
│   ├── audio_sync_moe.py          # Audio sync and lip-sync
│   ├── integrated_moe.py          # Wan2.2-S2V-14B integration
│   ├── image2video.py             # Image-to-video generation
│   ├── configs/                   # Model configurations
│   └── modules/                   # Neural network modules
├── examples/                       # Usage examples
│   ├── moe_example.py             # Basic MoE usage
│   ├── lip_sync_example.py        # Lip-sync generation
│   └── integrated_moe_example.py  # Integrated pipeline usage
├── tests/                         # Test scripts
├── requirements.txt                # Python dependencies
├── INSTALL.md                     # Detailed installation guide
└── README.md                      # This file
```

## 🧪 **Testing**

### **Run All Tests**
```bash
# Test basic functionality
python test_moe_pipeline.py

# Test audio sync capabilities
python test_audio_sync.py

# Test integrated pipeline
python -c "from stree import create_integrated_moe_pipeline; print('✅ All imports successful')"
```

### **Test Audio Analysis**
```bash
# Test audio processing (no CUDA required)
python -c "
import librosa
import numpy as np
print('✅ Audio libraries working')
"
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce video quality
   --num_frames 60  # 2.5 seconds instead of 5
   --num_inference_steps 25  # Faster generation
   ```

2. **Audio Generation Fails**
   ```bash
   # Check VibeVoice-7B installation
   ls VibeVoice-7B/
   # Reinstall if needed
   rm -rf VibeVoice-7B && git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git
   ```

3. **Video Generation Fails**
   ```bash
   # Verify Stree checkpoints
   ls checkpoints/
   # Check GPU availability
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

4. **FFmpeg Errors**
   ```bash
   # Install FFmpeg
   brew install ffmpeg  # macOS
   sudo apt install ffmpeg  # Ubuntu
   ```

### **Performance Tips**

- **GPU Memory**: Use 8GB+ VRAM for optimal performance
- **Batch Processing**: Process multiple videos in sequence
- **Quality vs. Speed**: Adjust `num_inference_steps` for trade-offs
- **Audio Length**: Shorter audio = faster video generation

## 📈 **Roadmap**

### **Phase 1: Core MoE Pipeline** ✅
- [x] Basic MoE implementation
- [x] Audio sync capabilities
- [x] Lip-sync generation
- [x] Expert routing

### **Phase 2: Advanced Features** ✅
- [x] Wan2.2-S2V-14B integration
- [x] Audio analysis and guidance
- [x] Beat synchronization
- [x] Quality prediction

### **Phase 3: Production Ready** 🚧
- [ ] Web interface
- [ ] API endpoints
- [ ] Batch processing
- [ ] Cloud deployment

### **Phase 4: Enterprise Features** 📋
- [ ] Multi-user support
- [ ] Advanced analytics
- [ ] Custom model training
- [ ] Enterprise integrations

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Clone and setup development environment
git clone https://github.com/prarabdha-soni/Stree.git
cd Stree
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## 📄 **License**

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Stree2.2 Team**: For the image-to-video model
- **VibeVoice-7B Team**: For the text-to-audio model
- **Wan2.2 Team**: For the speech-to-video model
- **FFmpeg Project**: For video processing capabilities
- **Open Source Community**: For the amazing tools and libraries

## 📞 **Support & Contact**

- **GitHub Issues**: [Report bugs or request features](https://github.com/prarabdha-soni/Stree/issues)
- **Discussions**: [Join the community](https://github.com/prarabdha-soni/Stree/discussions)
- **Email**: [Your email here]

---

**⭐ Star this repository if you find it useful!**

**🚀 Ready to create perfect lip-sync videos? Get started with the examples above!** 