"""
Mixture of Experts (MoE) Pipeline for Text+Image to Video Generation

This module combines:
1. VibeVoice-7B: Text-to-Speech generation
2. Stree2.2-I2V-A14B: Image-to-Video generation

The MoE system intelligently routes tasks and combines outputs for optimal results.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import subprocess
import tempfile
import json

from .image2video import StreeI2V
from .configs import STREE_CONFIGS


class VibeVoiceExpert:
    """Expert model for text-to-speech generation using VibeVoice-7B"""
    
    def __init__(self, model_path: str = "VibeVoice-7B", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.is_available = self._check_availability()
        
        if self.is_available:
            logging.info("VibeVoice-7B expert initialized successfully")
        else:
            logging.warning("VibeVoice-7B not found. Please clone the repository.")
    
    def _check_availability(self) -> bool:
        """Check if VibeVoice-7B is available"""
        return os.path.exists(os.path.join(self.model_path, "generate.py"))
    
    def generate_speech(self, text: str, duration: int = 5, voice: str = "en_female_1") -> str:
        """
        Generate speech from text using VibeVoice-7B
        
        Args:
            text: Input text to convert to speech
            duration: Duration of audio in seconds
            voice: Voice type to use
            
        Returns:
            Path to generated audio file
        """
        if not self.is_available:
            raise RuntimeError("VibeVoice-7B not available. Please install it first.")
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            audio_path = tmp_file.name
        
        try:
            # Run VibeVoice-7B generation
            cmd = [
                "python", os.path.join(self.model_path, "generate.py"),
                "--text", text,
                "--output", audio_path,
                "--duration", str(duration),
                "--voice", voice
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"VibeVoice-7B generated audio: {audio_path}")
            
            return audio_path
            
        except subprocess.CalledProcessError as e:
            logging.error(f"VibeVoice-7B generation failed: {e.stderr}")
            raise RuntimeError(f"Speech generation failed: {e.stderr}")


class StreeVideoExpert:
    """Expert model for image-to-video generation using Stree2.2-I2V-A14B"""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.config = STREE_CONFIGS['i2v-A14B']
        self.model = self._load_model()
        
    def _load_model(self) -> StreeI2V:
        """Load the Stree I2V model"""
        try:
            model = StreeI2V(
                config=self.config,
                checkpoint_dir=self.checkpoint_dir,
                device_id=0 if self.device == "cuda" else -1
            )
            logging.info("Stree2.2-I2V-A14B expert loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Failed to load Stree model: {e}")
            raise
    
    def generate_video(self, image_path: str, prompt: str, duration: int = 5) -> str:
        """
        Generate video from image using Stree2.2-I2V-A14B
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for video generation
            duration: Duration of video in seconds
            
        Returns:
            Path to generated video file
        """
        try:
            # Calculate frames based on duration (8 FPS default)
            num_frames = duration * 8
            
            # Generate video frames
            video_frames = self.model.generate(
                image_path=image_path,
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=7.5
            )
            
            # Save video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                video_path = tmp_file.name
            
            self.model.save_video(video_frames, video_path)
            logging.info(f"Stree generated video: {video_path}")
            
            return video_path
            
        except Exception as e:
            logging.error(f"Video generation failed: {e}")
            raise RuntimeError(f"Video generation failed: {e}")


class MoERouter(nn.Module):
    """Router for Mixture of Experts that decides which expert to use and how to combine outputs"""
    
    def __init__(self, input_dim: int = 768, num_experts: int = 2, top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts)
        )
        
        # Expert weights
        self.expert_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to experts
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            expert_weights: Routing weights [batch_size, num_experts]
            expert_indices: Top-k expert indices [batch_size, top_k]
        """
        # Get routing logits
        routing_logits = self.router(x)
        
        # Apply expert weights
        routing_logits = routing_logits + self.expert_weights.unsqueeze(0)
        
        # Get top-k experts
        expert_weights = F.softmax(routing_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        
        return top_k_weights, top_k_indices


class MoEPipeline:
    """Main Mixture of Experts pipeline combining VibeVoice-7B and Stree2.2-I2V-A14B"""
    
    def __init__(self, 
                 vibevoice_path: str = "VibeVoice-7B",
                 stree_checkpoint_dir: str = "./checkpoints",
                 device: str = "cuda"):
        
        self.device = device
        
        # Initialize experts
        self.vibevoice_expert = VibeVoiceExpert(vibevoice_path, device)
        self.stree_expert = StreeVideoExpert(stree_checkpoint_dir, device)
        
        # Initialize router
        self.router = MoERouter()
        
        # Expert mapping
        self.experts = {
            0: self.vibevoice_expert,
            1: self.stree_expert
        }
        
        logging.info("MoE Pipeline initialized successfully")
    
    def _encode_input(self, text: str, image_path: str) -> torch.Tensor:
        """Encode text and image for routing decision"""
        # Simple encoding - in practice, you might use a more sophisticated encoder
        text_encoding = torch.tensor([len(text)], dtype=torch.float32)
        image_size = Image.open(image_path).size
        image_encoding = torch.tensor([image_size[0] * image_size[1]], dtype=torch.float32)
        
        # Combine encodings
        combined = torch.cat([text_encoding, image_encoding])
        
        # Pad to router input dimension
        if combined.size(0) < 768:
            padding = torch.zeros(768 - combined.size(0))
            combined = torch.cat([combined, padding])
        else:
            combined = combined[:768]
            
        return combined.unsqueeze(0)  # Add batch dimension
    
    def _route_task(self, text: str, image_path: str) -> Dict[str, float]:
        """Route the task to appropriate experts"""
        # Encode input
        input_encoding = self._encode_input(text, image_path)
        
        # Get routing decision
        with torch.no_grad():
            expert_weights, expert_indices = self.router(input_encoding)
        
        # Map to expert names
        expert_names = ["VibeVoice-7B", "Stree2.2-I2V-A14B"]
        routing_decision = {}
        
        for i in range(expert_weights.size(1)):
            expert_idx = expert_indices[0, i].item()
            weight = expert_weights[0, i].item()
            routing_decision[expert_names[expert_idx]] = weight
        
        logging.info(f"MoE Routing Decision: {routing_decision}")
        return routing_decision
    
    def generate(self, 
                text: str, 
                image_path: str, 
                duration: int = 5,
                voice: str = "en_female_1",
                use_routing: bool = True) -> Dict[str, Any]:
        """
        Generate video with audio using MoE pipeline
        
        Args:
            text: Input text for speech generation
            image_path: Input image for video generation
            duration: Duration of output in seconds
            voice: Voice type for speech generation
            use_routing: Whether to use MoE routing or run both experts
            
        Returns:
            Dictionary containing paths to generated files and metadata
        """
        try:
            results = {}
            
            if use_routing:
                # Use MoE routing
                routing_decision = self._route_task(text, image_path)
                
                # Generate based on routing weights
                if routing_decision.get("VibeVoice-7B", 0) > 0.3:
                    logging.info("Generating speech with VibeVoice-7B...")
                    audio_path = self.vibevoice_expert.generate_speech(text, duration, voice)
                    results["audio_path"] = audio_path
                    results["audio_expert"] = "VibeVoice-7B"
                    results["audio_weight"] = routing_decision["VibeVoice-7B"]
                
                if routing_decision.get("Stree2.2-I2V-A14B", 0) > 0.3:
                    logging.info("Generating video with Stree2.2-I2V-A14B...")
                    video_path = self.stree_expert.generate_video(image_path, text, duration)
                    results["video_path"] = video_path
                    results["video_expert"] = "Stree2.2-I2V-A14B"
                    results["video_weight"] = routing_decision["Stree2.2-I2V-A14B"]
                
                results["routing_decision"] = routing_decision
                
            else:
                # Run both experts without routing
                logging.info("Running both experts without routing...")
                
                # Generate speech
                audio_path = self.vibevoice_expert.generate_speech(text, duration, voice)
                results["audio_path"] = audio_path
                results["audio_expert"] = "VibeVoice-7B"
                
                # Generate video
                video_path = self.stree_expert.generate_video(image_path, text, duration)
                results["video_path"] = video_path
                results["video_expert"] = "Stree2.2-I2V-A14B"
                
                results["routing_decision"] = {"VibeVoice-7B": 0.5, "Stree2.2-I2V-A14B": 0.5}
            
            # Combine audio and video if both are generated
            if "audio_path" in results and "video_path" in results:
                final_video_path = self._combine_audio_video(
                    results["video_path"], 
                    results["audio_path"]
                )
                results["final_video_path"] = final_video_path
            
            results["success"] = True
            return results
            
        except Exception as e:
            logging.error(f"MoE generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _combine_audio_video(self, video_path: str, audio_path: str) -> str:
        """Combine audio and video using FFmpeg"""
        try:
            # Create output path
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # FFmpeg command to combine audio and video
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"Combined audio and video: {output_path}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg combination failed: {e.stderr}")
            raise RuntimeError(f"Failed to combine audio and video: {e.stderr}")
    
    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about available experts"""
        return {
            "VibeVoice-7B": {
                "available": self.vibevoice_expert.is_available,
                "type": "Text-to-Speech",
                "model_path": self.vibevoice_expert.model_path
            },
            "Stree2.2-I2V-A14B": {
                "available": True,  # Assuming Stree is always available
                "type": "Image-to-Video",
                "checkpoint_dir": self.stree_expert.checkpoint_dir
            }
        }


# Convenience function for easy usage
def create_moe_pipeline(vibevoice_path: str = "VibeVoice-7B",
                       stree_checkpoint_dir: str = "./checkpoints",
                       device: str = "cuda") -> MoEPipeline:
    """
    Create a MoE pipeline instance
    
    Args:
        vibevoice_path: Path to VibeVoice-7B repository
        stree_checkpoint_dir: Path to Stree checkpoints
        device: Device to use (cuda/cpu)
        
    Returns:
        Configured MoE pipeline
    """
    return MoEPipeline(vibevoice_path, stree_checkpoint_dir, device) 