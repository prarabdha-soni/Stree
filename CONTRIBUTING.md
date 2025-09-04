# Contributing to Stree

Thank you for your interest in contributing to Stree! This document provides guidelines and information for contributors.

## 🎯 **What We're Building**

Stree is a **Mixture of Experts (MoE) pipeline** that combines:
- **VibeVoice-7B**: Advanced text-to-speech generation
- **Stree2.2-I2V-A14B**: High-quality image-to-video generation
- **Wan2.2-S2V-14B**: Native speech-to-video capability
- **Audio Sync MoE**: Intelligent expert routing and lip-sync

Our goal is to create **perfect lip-sync videos** from text and images with professional-grade quality.

## 🚀 **How to Contribute**

### **1. Report Bugs**
- Use the [GitHub Issues](https://github.com/prarabdha-soni/Stree/issues) page
- Include detailed reproduction steps
- Attach error logs and system information
- Use the "Bug Report" template

### **2. Suggest Features**
- Open a [Feature Request](https://github.com/prarabdha-soni/Stree/issues/new?template=feature_request.md)
- Describe the use case and benefits
- Include mockups or examples if applicable
- Discuss with the community first

### **3. Submit Code Changes**
- Fork the repository
- Create a feature branch
- Make your changes
- Add tests if applicable
- Submit a pull request

### **4. Improve Documentation**
- Fix typos and clarify explanations
- Add missing examples
- Improve installation guides
- Translate to other languages

## 🛠️ **Development Setup**

### **Prerequisites**
```bash
# Clone the repository
git clone https://github.com/prarabdha-soni/Stree.git
cd Stree

# Create virtual environment
python3 -m venv stree_dev
source stree_dev/bin/activate  # Linux/macOS
# OR
stree_dev\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### **Development Dependencies**
```bash
# Code quality tools
pip install black isort yapf flake8 mypy

# Testing tools
pip install pytest pytest-cov pytest-mock

# Documentation tools
pip install sphinx sphinx-rtd-theme
```

## 📝 **Code Style Guidelines**

### **Python Code Style**
- **PEP 8**: Follow Python style guide
- **Black**: Use Black for code formatting
- **Type Hints**: Include type annotations
- **Docstrings**: Use Google-style docstrings

### **Example Code Style**
```python
def generate_lip_sync_video(
    text: str,
    image_path: str,
    duration: int = 5,
    sync_quality: str = "high"
) -> Dict[str, Any]:
    """
    Generate lip-sync video from text and image.
    
    Args:
        text: Text to convert to speech
        image_path: Path to input image
        duration: Duration in seconds
        sync_quality: Lip-sync quality level
        
    Returns:
        Dictionary containing generation results
        
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If generation fails
    """
    # Validate inputs
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    # Implementation here...
    pass
```

### **File Organization**
```
stree/
├── __init__.py              # Package exports
├── moe_pipeline.py          # Core MoE implementation
├── audio_sync_moe.py        # Audio sync capabilities
├── integrated_moe.py        # Expert integration
├── configs/                 # Configuration files
├── modules/                 # Neural network modules
└── utils/                   # Utility functions
```

## 🧪 **Testing Guidelines**

### **Running Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=stree

# Run specific test file
pytest tests/test_moe_pipeline.py

# Run with verbose output
pytest -v
```

### **Writing Tests**
```python
import pytest
from unittest.mock import Mock, patch
from stree.moe_pipeline import MoEPipeline

class TestMoEPipeline:
    """Test cases for MoE Pipeline."""
    
    def test_pipeline_creation(self):
        """Test that pipeline can be created successfully."""
        pipeline = MoEPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'experts')
    
    def test_expert_routing(self):
        """Test expert routing logic."""
        pipeline = MoEPipeline()
        
        # Test with valid input
        result = pipeline.select_expert("text", "high")
        assert result in ["audio_sync_moe", "basic_moe"]
    
    @patch('stree.moe_pipeline.VibeVoiceExpert')
    def test_audio_generation(self, mock_vibevoice):
        """Test audio generation with mocked dependencies."""
        mock_vibevoice.return_value.generate_speech.return_value = "test.wav"
        
        pipeline = MoEPipeline()
        result = pipeline.generate_audio("Hello world")
        
        assert result == "test.wav"
        mock_vibevoice.return_value.generate_speech.assert_called_once_with("Hello world")
```

### **Test Coverage Requirements**
- **Unit Tests**: 90%+ coverage for core modules
- **Integration Tests**: Test complete pipeline workflows
- **Edge Cases**: Test error conditions and boundary cases
- **Performance Tests**: Test with large inputs and long durations

## 📚 **Documentation Standards**

### **Code Documentation**
- **Module Docstrings**: Describe purpose and usage
- **Function Docstrings**: Include parameters, returns, and examples
- **Class Docstrings**: Explain purpose and key methods
- **Inline Comments**: Clarify complex logic

### **README Updates**
- Update examples when adding new features
- Include new configuration options
- Update installation instructions
- Add troubleshooting tips

### **API Documentation**
- Document all public functions and classes
- Include usage examples
- Specify parameter types and constraints
- Document return values and exceptions

## 🔄 **Pull Request Process**

### **1. Before Submitting**
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] No sensitive information is included

### **2. Pull Request Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### **3. Review Process**
- Maintainers review within 48 hours
- Address feedback and suggestions
- Ensure CI/CD checks pass
- Merge after approval

## 🚨 **Security Guidelines**

### **Data Privacy**
- **No Personal Data**: Never commit personal information
- **Model Weights**: Don't commit large model files
- **API Keys**: Use environment variables for secrets
- **User Data**: Handle user data securely

### **Code Security**
- **Input Validation**: Validate all user inputs
- **Error Handling**: Don't expose sensitive information in errors
- **Dependencies**: Keep dependencies updated
- **Vulnerability Scanning**: Run security scans regularly

## 🌟 **Recognition & Rewards**

### **Contributor Recognition**
- **Contributors List**: Added to README.md
- **Commit History**: Preserved in git history
- **Release Notes**: Acknowledged in releases
- **Community Spotlight**: Featured in discussions

### **Types of Contributions**
- **Code**: New features, bug fixes, improvements
- **Documentation**: Guides, examples, tutorials
- **Testing**: Test cases, bug reports, feedback
- **Community**: Support, discussions, evangelism

## 📞 **Getting Help**

### **Communication Channels**
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community chat
- **Pull Requests**: Code review and collaboration
- **Email**: Direct contact for sensitive issues

### **Resources**
- **README.md**: Main documentation
- **INSTALL.md**: Installation guide
- **Examples**: Usage examples in `examples/` directory
- **Tests**: Test files for reference

## 🎯 **Current Priorities**

### **High Priority**
- [ ] Performance optimization for large videos
- [ ] Better error handling and user feedback
- [ ] Web interface for easy usage
- [ ] Batch processing capabilities

### **Medium Priority**
- [ ] Additional voice options
- [ ] More video generation models
- [ ] Cloud deployment support
- [ ] Mobile app development

### **Low Priority**
- [ ] Additional language support
- [ ] Custom model training
- [ ] Enterprise features
- [ ] Advanced analytics

## 📋 **Code of Conduct**

### **Community Standards**
- **Be Respectful**: Treat all contributors with respect
- **Be Inclusive**: Welcome contributors from all backgrounds
- **Be Constructive**: Provide helpful, constructive feedback
- **Be Patient**: Understand that everyone learns at different paces

### **Unacceptable Behavior**
- **Harassment**: No harassment or discrimination
- **Spam**: No spam or off-topic content
- **Trolling**: No trolling or inflammatory behavior
- **Spam**: No commercial spam or advertising

## 🎉 **Getting Started**

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests if applicable**
5. **Update documentation**
6. **Submit a pull request**

### **First Contribution Ideas**
- Fix a typo in documentation
- Add a simple test case
- Improve error messages
- Add usage examples
- Update installation instructions

## 🙏 **Thank You**

Thank you for contributing to Stree! Your contributions help make this project better for everyone in the AI and video generation community.

**Together, we're building the future of perfect lip-sync video generation! 🎬✨**

---

**Questions? Open an issue or start a discussion!** 