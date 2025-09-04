#!/usr/bin/env python3
"""
Example script demonstrating the Mixture of Experts (MoE) Pipeline

This script shows how to use the MoE system that combines:
1. VibeVoice-7B for text-to-speech generation
2. Stree2.2-I2V-A14B for image-to-video generation

The MoE router intelligently decides which expert to use based on the input.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import stree
sys.path.append(str(Path(__file__).parent.parent))

from stree import create_moe_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MoE Pipeline Example")
    parser.add_argument("--text", type=str, required=True, help="Text to convert to speech")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--duration", type=int, default=5, help="Duration of output in seconds")
    parser.add_argument("--voice", type=str, default="en_female_1", help="Voice type for speech")
    parser.add_argument("--vibevoice-path", type=str, default="VibeVoice-7B", 
                       help="Path to VibeVoice-7B repository")
    parser.add_argument("--stree-checkpoint-dir", type=str, default="./checkpoints",
                       help="Path to Stree checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--no-routing", action="store_true", 
                       help="Disable MoE routing and run both experts")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.vibevoice_path):
        logger.warning(f"VibeVoice-7B not found at: {args.vibevoice_path}")
        logger.warning("Please clone the VibeVoice-7B repository first:")
        logger.warning("git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git")
    
    # Create MoE pipeline
    try:
        logger.info("Initializing MoE Pipeline...")
        moe_pipeline = create_moe_pipeline(
            vibevoice_path=args.vibevoice_path,
            stree_checkpoint_dir=args.stree_checkpoint_dir,
            device=args.device
        )
        
        # Get expert information
        expert_info = moe_pipeline.get_expert_info()
        logger.info("Available Experts:")
        for expert_name, info in expert_info.items():
            status = "✓ Available" if info["available"] else "✗ Not Available"
            logger.info(f"  {expert_name}: {status} ({info['type']})")
        
        # Generate content using MoE
        logger.info(f"Generating content with text: '{args.text}'")
        logger.info(f"Input image: {args.image}")
        logger.info(f"Duration: {args.duration} seconds")
        logger.info(f"Voice: {args.voice}")
        logger.info(f"MoE Routing: {'Enabled' if not args.no_routing else 'Disabled'}")
        
        results = moe_pipeline.generate(
            text=args.text,
            image_path=args.image,
            duration=args.duration,
            voice=args.voice,
            use_routing=not args.no_routing
        )
        
        if results["success"]:
            logger.info("🎉 MoE generation completed successfully!")
            logger.info("Generated files:")
            
            if "audio_path" in results:
                logger.info(f"  Audio: {results['audio_path']}")
                logger.info(f"  Audio Expert: {results['audio_expert']}")
                if 'audio_weight' in results:
                    logger.info(f"  Audio Weight: {results['audio_weight']:.3f}")
            
            if "video_path" in results:
                logger.info(f"  Video: {results['video_path']}")
                logger.info(f"  Video Expert: {results['video_expert']}")
                if 'video_weight' in results:
                    logger.info(f"  Video Weight: {results['video_weight']:.3f}")
            
            if "final_video_path" in results:
                logger.info(f"  Final Combined Video: {results['final_video_path']}")
            
            if "routing_decision" in results:
                logger.info("MoE Routing Decision:")
                for expert, weight in results["routing_decision"].items():
                    logger.info(f"  {expert}: {weight:.3f}")
            
            return 0
        else:
            logger.error(f"MoE generation failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Failed to create or use MoE pipeline: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 