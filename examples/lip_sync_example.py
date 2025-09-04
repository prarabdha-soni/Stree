#!/usr/bin/env python3
"""
Lip-Sync and Audio Synchronization Example

This script demonstrates how to use the enhanced MoE pipeline for:
1. Perfect lip-sync generation
2. Audio-driven video creation
3. Temporal alignment between audio and video
4. High-quality synchronization

Requirements:
- librosa (for audio analysis)
- opencv-python (for video processing)
- scipy (for signal processing)
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import stree
sys.path.append(str(Path(__file__).parent.parent))

from stree import create_audio_sync_moe_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Lip-Sync and Audio Sync Example")
    parser.add_argument("--text", type=str, required=True, 
                       help="Text to convert to speech for lip-sync")
    parser.add_argument("--image", type=str, required=True, 
                       help="Path to input image (should contain face/character)")
    parser.add_argument("--duration", type=int, default=5, 
                       help="Duration of output in seconds")
    parser.add_argument("--voice", type=str, default="en_female_1", 
                       help="Voice type for speech generation")
    parser.add_argument("--sync-quality", type=str, default="high", 
                       choices=["basic", "enhanced", "high"],
                       help="Lip-sync quality level")
    parser.add_argument("--vibevoice-path", type=str, default="VibeVoice-7B", 
                       help="Path to VibeVoice-7B repository")
    parser.add_argument("--stree-checkpoint-dir", type=str, default="./checkpoints",
                       help="Path to Stree checkpoints")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.vibevoice_path):
        logger.warning(f"VibeVoice-7B not found at: {args.vibevoice_path}")
        logger.warning("Please clone the VibeVoice-7B repository first:")
        logger.warning("git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git")
    
    # Check for required dependencies
    try:
        import librosa
        import cv2
        from scipy import signal
        logger.info("✅ All required dependencies are available")
    except ImportError as e:
        logger.error(f"❌ Missing required dependency: {e}")
        logger.error("Please install: pip install librosa opencv-python scipy")
        return 1
    
    # Create audio-sync MoE pipeline
    try:
        logger.info("🚀 Initializing Audio-Sync MoE Pipeline...")
        pipeline = create_audio_sync_moe_pipeline(
            vibevoice_path=args.vibevoice_path,
            stree_checkpoint_dir=args.stree_checkpoint_dir,
            device=args.device
        )
        
        # Get synchronization capabilities
        capabilities = pipeline.get_sync_capabilities()
        logger.info("🎯 Audio-Sync Capabilities:")
        logger.info(f"  Lip-sync: {capabilities['lip_sync']['available']}")
        logger.info(f"  Quality levels: {', '.join(capabilities['lip_sync']['quality_levels'])}")
        logger.info(f"  Max FPS: {capabilities['lip_sync']['max_fps']}")
        logger.info(f"  Audio-driven: {capabilities['audio_driven']['available']}")
        
        # Generate lip-sync content
        logger.info(f"\n🎬 Generating lip-sync content...")
        logger.info(f"  Text: '{args.text}'")
        logger.info(f"  Image: {args.image}")
        logger.info(f"  Duration: {args.duration} seconds")
        logger.info(f"  Voice: {args.voice}")
        logger.info(f"  Sync Quality: {args.sync_quality}")
        
        results = pipeline.generate_with_lip_sync(
            text=args.text,
            image_path=args.image,
            duration=args.duration,
            voice=args.voice,
            sync_quality=args.sync_quality
        )
        
        if results["success"]:
            logger.info("\n🎉 Lip-sync generation completed successfully!")
            logger.info("Generated files:")
            logger.info(f"  🎵 Audio: {results['audio_path']}")
            logger.info(f"  🎬 Video: {results['video_path']}")
            logger.info(f"  🔗 Final Sync Video: {results['final_video_path']}")
            logger.info(f"  📊 Video Expert: {results['video_expert']}")
            logger.info(f"  🎯 Sync Quality: {results['sync_quality']}")
            logger.info(f"  🎞️  FPS: {results['fps']}")
            logger.info(f"  📹 Total Frames: {results['total_frames']}")
            
            # Audio analysis results
            audio_features = results.get('audio_features', {})
            if audio_features:
                logger.info("\n🔍 Audio Analysis Results:")
                logger.info(f"  Duration: {audio_features['duration']:.2f}s")
                logger.info(f"  Sample Rate: {audio_features['sample_rate']} Hz")
                logger.info(f"  Beat Count: {audio_features['beat_count']}")
                logger.info(f"  Energy Mean: {audio_features['energy_stats']['mean']:.3f}")
                logger.info(f"  Energy Variance: {audio_features['energy_stats']['variance']:.3f}")
            
            logger.info(f"\n✨ Your lip-sync video is ready: {results['final_video_path']}")
            return 0
        else:
            logger.error(f"❌ Lip-sync generation failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Failed to create or use audio-sync MoE pipeline: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 