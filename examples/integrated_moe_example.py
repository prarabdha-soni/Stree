#!/usr/bin/env python3
"""
Integrated MoE Pipeline Example

This script demonstrates the integrated MoE pipeline that combines:
1. Wan2.2-S2V-14B native speech-to-video (when available)
2. Enhanced audio sync MoE for customizable lip-sync
3. Intelligent expert routing based on input type and requirements

The pipeline automatically selects the best expert for each task.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import stree
sys.path.append(str(Path(__file__).parent.parent))

from stree import create_integrated_moe_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Integrated MoE Pipeline Example")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input: text string or 'audio:path/to/audio.wav' for speech")
    parser.add_argument("--image", type=str, required=True, 
                       help="Path to input image (should contain face/character)")
    parser.add_argument("--duration", type=int, default=5, 
                       help="Duration of output in seconds (for text input)")
    parser.add_argument("--voice", type=str, default="en_female_1", 
                       help="Voice type for speech generation (for text input)")
    parser.add_argument("--sync-quality", type=str, default="high", 
                       choices=["basic", "enhanced", "high", "native"],
                       help="Lip-sync quality level")
    parser.add_argument("--prefer-native", action="store_true",
                       help="Prefer native Wan2.2-S2V-14B when available")
    parser.add_argument("--vibevoice-path", type=str, default="VibeVoice-7B", 
                       help="Path to VibeVoice-7B repository")
    parser.add_argument("--stree-checkpoint-dir", type=str, default="./checkpoints",
                       help="Path to Stree checkpoints")
    parser.add_argument("--wan-s2v-path", type=str, default=None,
                       help="Path to Wan2.2-S2V-14B model (optional)")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return 1
    
    # Parse input type
    if args.input.startswith("audio:"):
        # Speech input
        audio_path = args.input[6:]  # Remove "audio:" prefix
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return 1
        input_data = {"audio_path": audio_path, "text": "Audio input"}
        input_type = "speech"
    else:
        # Text input
        input_data = args.input
        input_type = "text"
    
    # Check for VibeVoice-7B (only needed for text input)
    if input_type == "text" and not os.path.exists(args.vibevoice_path):
        logger.warning(f"VibeVoice-7B not found at: {args.vibevoice_path}")
        logger.warning("Please clone the VibeVoice-7B repository first:")
        logger.warning("git clone https://github.com/OpenVoiceOS/VibeVoice-7B.git")
    
    # Create integrated MoE pipeline
    try:
        logger.info("🚀 Initializing Integrated MoE Pipeline...")
        pipeline = create_integrated_moe_pipeline(
            vibevoice_path=args.vibevoice_path,
            stree_checkpoint_dir=args.stree_checkpoint_dir,
            wan_s2v_path=args.wan_s2v_path,
            device=args.device
        )
        
        # Get pipeline information
        pipeline_info = pipeline.get_pipeline_info()
        logger.info("🎯 Integrated MoE Pipeline Capabilities:")
        logger.info(f"  Name: {pipeline_info['name']}")
        logger.info(f"  Description: {pipeline_info['description']}")
        logger.info(f"  Supported inputs: {', '.join(pipeline_info['supported_inputs'])}")
        logger.info(f"  Sync qualities: {', '.join(pipeline_info['sync_qualities'])}")
        
        # Get expert capabilities
        expert_capabilities = pipeline.get_expert_capabilities()
        logger.info("\n📊 Expert Status:")
        for expert_name, capabilities in expert_capabilities.items():
            status = "✅ Available" if capabilities.get('available', True) else "❌ Not Available"
            logger.info(f"  {expert_name}: {status}")
            if 'description' in capabilities:
                logger.info(f"    {capabilities['description']}")
        
        # Generate content with smart routing
        logger.info(f"\n🎬 Generating content with smart routing...")
        logger.info(f"  Input type: {input_type}")
        logger.info(f"  Image: {args.image}")
        if input_type == "text":
            logger.info(f"  Text: '{input_data}'")
            logger.info(f"  Duration: {args.duration} seconds")
            logger.info(f"  Voice: {args.voice}")
        else:
            logger.info(f"  Audio: {input_data['audio_path']}")
        logger.info(f"  Sync quality: {args.sync_quality}")
        logger.info(f"  Prefer native: {args.prefer_native}")
        
        results = pipeline.generate_with_smart_routing(
            input_data=input_data,
            image_path=args.image,
            duration=args.duration,
            voice=args.voice,
            sync_quality=args.sync_quality,
            use_native_if_available=args.prefer_native
        )
        
        if results["success"]:
            logger.info("\n🎉 Content generation completed successfully!")
            logger.info("Results:")
            logger.info(f"  Selected expert: {results['selected_expert']}")
            logger.info(f"  Input type: {results['input_type']}")
            logger.info(f"  Sync quality: {results['sync_quality']}")
            logger.info(f"  Expert selection reason: {results['expert_selection_reason']}")
            
            # Show expert-specific results
            if results['selected_expert'] == 'wan2_2_s2v':
                logger.info("\n🎬 Wan2.2-S2V-14B Results:")
                logger.info(f"  Method: {results['method']}")
                logger.info(f"  Model type: {results['model_type']}")
                logger.info(f"  Capabilities: {', '.join(results['capabilities'])}")
                logger.info(f"  Audio: {results['audio_path']}")
                logger.info(f"  Image: {results['image_path']}")
                logger.info(f"  Video: {results['video_path']}")
                
            elif results['selected_expert'] == 'audio_sync_moe':
                logger.info("\n🎬 Audio Sync MoE Results:")
                logger.info(f"  Expert: {results['expert']}")
                logger.info(f"  Audio: {results['audio_path']}")
                logger.info(f"  Video: {results['video_path']}")
                logger.info(f"  Final video: {results['final_video_path']}")
                
                # Audio analysis results
                audio_features = results.get('audio_features', {})
                if audio_features:
                    logger.info("\n🔍 Audio Analysis Results:")
                    logger.info(f"  Duration: {audio_features['duration']:.2f}s")
                    logger.info(f"  Sample rate: {audio_features['sample_rate']} Hz")
                    logger.info(f"  Beat count: {audio_features['beat_count']}")
            
            logger.info(f"\n✨ Your content is ready! Generated using: {results['selected_expert']}")
            return 0
        else:
            logger.error(f"❌ Content generation failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Failed to create or use integrated MoE pipeline: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 