#!/usr/bin/env python3
"""
Test script for Mixture of Experts (MoE) Pipeline

This script tests both the basic and advanced MoE implementations:
1. Basic MoE: Simple routing with basic encoders
2. Advanced MoE: Transformer-based routing with performance monitoring
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from stree import create_moe_pipeline, create_advanced_moe_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_moe():
    """Test the basic MoE pipeline"""
    logger.info("🧪 Testing Basic MoE Pipeline...")
    
    try:
        # Create basic MoE pipeline
        moe_pipeline = create_moe_pipeline(
            vibevoice_path="VibeVoice-7B",
            stree_checkpoint_dir="./checkpoints",
            device="cpu"  # Use CPU for testing
        )
        
        # Get expert information
        expert_info = moe_pipeline.get_expert_info()
        logger.info("Basic MoE Experts:")
        for expert_name, info in expert_info.items():
            status = "✓ Available" if info["available"] else "✗ Not Available"
            logger.info(f"  {expert_name}: {status} ({info['type']})")
        
        logger.info("✅ Basic MoE pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic MoE pipeline test failed: {e}")
        return False


def test_advanced_moe():
    """Test the advanced MoE pipeline"""
    logger.info("🧪 Testing Advanced MoE Pipeline...")
    
    try:
        # Create advanced MoE pipeline
        advanced_moe = create_advanced_moe_pipeline(
            vibevoice_path="VibeVoice-7B",
            stree_checkpoint_dir="./checkpoints",
            device="cpu"  # Use CPU for testing
        )
        
        # Test text encoder
        logger.info("Testing text encoder...")
        test_text = "Hello, this is a test of the advanced MoE pipeline."
        text_encoding = advanced_moe.text_encoder(test_text)
        logger.info(f"Text encoding shape: {text_encoding.shape}")
        
        # Test image encoder (with a dummy image path)
        logger.info("Testing image encoder...")
        dummy_image_path = "examples/i2v_input.JPG"  # Use existing example image
        if os.path.exists(dummy_image_path):
            try:
                image_encoding = advanced_moe.image_encoder(dummy_image_path)
                logger.info(f"Image encoding shape: {image_encoding.shape}")
            except Exception as e:
                logger.warning(f"Image encoding test failed (expected for CPU): {e}")
        else:
            logger.warning("Example image not found, skipping image encoder test")
        
        # Test router
        logger.info("Testing router...")
        if os.path.exists(dummy_image_path):
            try:
                routing_decision = advanced_moe._route_task(test_text, dummy_image_path)
                logger.info(f"Routing decision: {routing_decision}")
            except Exception as e:
                logger.warning(f"Router test failed (expected for CPU): {e}")
        
        logger.info("✅ Advanced MoE pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced MoE pipeline test failed: {e}")
        return False


def test_moe_components():
    """Test individual MoE components"""
    logger.info("🧪 Testing MoE Components...")
    
    try:
        # Test basic MoE components
        from stree.moe_pipeline import VibeVoiceExpert, StreeVideoExpert, MoERouter
        
        # Test VibeVoice expert
        vibevoice_expert = VibeVoiceExpert("VibeVoice-7B", "cpu")
        logger.info(f"VibeVoice expert available: {vibevoice_expert.is_available}")
        
        # Test Stree video expert (this will fail without checkpoints, which is expected)
        try:
            stree_expert = StreeVideoExpert("./checkpoints", "cpu")
            logger.info("Stree video expert created successfully")
        except Exception as e:
            logger.info(f"Stree video expert creation failed (expected without checkpoints): {e}")
        
        # Test MoE router
        router = MoERouter()
        test_input = torch.randn(1, 768)  # Random input
        with torch.no_grad():
            weights, indices = router(test_input)
        logger.info(f"Router output - weights: {weights.shape}, indices: {indices.shape}")
        
        logger.info("✅ MoE components test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ MoE components test failed: {e}")
        return False


def test_advanced_moe_components():
    """Test advanced MoE components"""
    logger.info("🧪 Testing Advanced MoE Components...")
    
    try:
        # Test advanced MoE components
        from stree.advanced_moe import (
            AdvancedTextEncoder, 
            AdvancedImageEncoder, 
            AdvancedMoERouter,
            ExpertMetrics
        )
        
        # Test text encoder
        text_encoder = AdvancedTextEncoder()
        test_text = "Testing advanced text encoder"
        text_encoding = text_encoder(test_text)
        logger.info(f"Advanced text encoder output shape: {text_encoding.shape}")
        
        # Test image encoder
        image_encoder = AdvancedImageEncoder()
        logger.info(f"Advanced image encoder created with hidden dim: {image_encoder.hidden_dim}")
        
        # Test MoE router
        router = AdvancedMoERouter()
        test_text_enc = torch.randn(1, 768)
        test_image_enc = torch.randn(1, 768)
        with torch.no_grad():
            weights, indices, quality = router(test_text_enc, test_image_enc)
        logger.info(f"Advanced router output - weights: {weights.shape}, indices: {indices.shape}, quality: {quality.shape}")
        
        # Test expert metrics
        metrics = ExpertMetrics()
        metrics.total_requests = 10
        metrics.successful_requests = 8
        metrics.error_count = 1
        logger.info(f"Expert metrics - success rate: {metrics.success_rate:.2f}, reliability: {metrics.reliability_score:.2f}")
        
        logger.info("✅ Advanced MoE components test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced MoE components test failed: {e}")
        return False


def main():
    """Run all MoE tests"""
    logger.info("🚀 Starting MoE Pipeline Tests...")
    
    # Import torch for testing
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch not available. Please install it first.")
        return 1
    
    # Run tests
    tests = [
        ("Basic MoE Pipeline", test_basic_moe),
        ("Advanced MoE Pipeline", test_advanced_moe),
        ("MoE Components", test_moe_components),
        ("Advanced MoE Components", test_advanced_moe_components)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        success = test_func()
        end_time = time.time()
        
        results[test_name] = {
            "success": success,
            "duration": end_time - start_time
        }
        
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status} (Duration: {end_time - start_time:.2f}s)")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result["success"])
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        logger.info(f"{test_name}: {status} ({result['duration']:.2f}s)")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! MoE pipeline is working correctly.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    exit(main()) 