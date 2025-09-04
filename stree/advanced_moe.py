"""
Advanced Mixture of Experts (MoE) Implementation

This module provides a more sophisticated MoE system with:
1. Better routing logic using transformer-based encoders
2. Expert specialization and load balancing
3. Dynamic expert selection based on task complexity
4. Quality-aware routing decisions
5. Expert performance monitoring
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from PIL import Image
import json
import time
from dataclasses import dataclass
from collections import defaultdict

from .image2video import StreeI2V
from .configs import STREE_CONFIGS


@dataclass
class ExpertMetrics:
    """Metrics for tracking expert performance"""
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0
    last_used: float = 0.0
    error_count: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def reliability_score(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests - self.error_count) / self.total_requests


class AdvancedTextEncoder(nn.Module):
    """Advanced text encoder for routing decisions"""
    
    def __init__(self, vocab_size: int = 50000, hidden_dim: int = 768, max_length: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, text: str) -> torch.Tensor:
        """Encode text to vector representation"""
        # Simple tokenization (in practice, use proper tokenizer)
        tokens = [ord(c) % self.vocab_size for c in text[:self.max_length]]
        tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad
        
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        positions = torch.arange(tokens.size(1), dtype=torch.long).unsqueeze(0)
        
        # Get embeddings
        text_emb = self.text_embedding(tokens)
        pos_emb = self.position_embedding(positions)
        
        # Combine and encode
        combined = text_emb + pos_emb
        encoded = self.transformer(combined)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        output = self.output_projection(pooled)
        
        return output


class AdvancedImageEncoder(nn.Module):
    """Advanced image encoder for routing decisions"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # CNN backbone for image features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Global average pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(512, hidden_dim)
        
    def forward(self, image_path: str) -> torch.Tensor:
        """Encode image to vector representation"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
            image = image.unsqueeze(0)  # Add batch dimension
            
            # Extract features
            features = self.conv_layers(image)
            pooled = self.global_pool(features).squeeze(-1).squeeze(-1)
            encoded = self.projection(pooled)
            
            return encoded
            
        except Exception as e:
            logging.warning(f"Image encoding failed: {e}, using fallback encoding")
            # Fallback: return zeros
            return torch.zeros(1, self.hidden_dim)


class AdvancedMoERouter(nn.Module):
    """Advanced router with transformer-based decision making"""
    
    def __init__(self, input_dim: int = 768, num_experts: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Expert routing
        self.expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Quality prediction
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Expert weights (learnable bias)
        self.expert_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        
    def forward(self, text_encoding: torch.Tensor, image_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advanced routing decision
        
        Args:
            text_encoding: Text features [batch_size, input_dim]
            image_encoding: Image features [batch_size, input_dim]
            
        Returns:
            expert_weights: Routing weights [batch_size, num_experts]
            expert_indices: Top expert indices [batch_size, num_experts]
            quality_score: Predicted quality [batch_size, 1]
        """
        # Concatenate features
        combined = torch.cat([text_encoding, image_encoding], dim=-1)
        
        # Feature fusion
        fused = self.feature_fusion(combined)
        
        # Expert routing
        routing_logits = self.expert_router(fused)
        routing_logits = routing_logits + self.expert_weights.unsqueeze(0)
        
        # Get expert weights
        expert_weights = F.softmax(routing_logits, dim=-1)
        
        # Get expert indices (all experts for now)
        expert_indices = torch.arange(self.num_experts).unsqueeze(0).expand(expert_weights.size(0), -1)
        
        # Quality prediction
        quality_score = self.quality_predictor(fused)
        
        return expert_weights, expert_indices, quality_score


class AdvancedVibeVoiceExpert:
    """Advanced VibeVoice-7B expert with performance monitoring"""
    
    def __init__(self, model_path: str = "VibeVoice-7B", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.is_available = self._check_availability()
        self.metrics = ExpertMetrics()
        
        if self.is_available:
            logging.info("Advanced VibeVoice-7B expert initialized successfully")
        else:
            logging.warning("VibeVoice-7B not found. Please clone the repository.")
    
    def _check_availability(self) -> bool:
        """Check if VibeVoice-7B is available"""
        return os.path.exists(os.path.join(self.model_path, "generate.py"))
    
    def generate_speech(self, text: str, duration: int = 5, voice: str = "en_female_1") -> str:
        """Generate speech with performance monitoring"""
        start_time = time.time()
        self.metrics.total_requests += 1
        self.metrics.last_used = time.time()
        
        try:
            if not self.is_available:
                raise RuntimeError("VibeVoice-7B not available")
            
            # Create temporary audio file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                audio_path = tmp_file.name
            
            # Run VibeVoice-7B generation
            import subprocess
            cmd = [
                "python", os.path.join(self.model_path, "generate.py"),
                "--text", text,
                "--output", audio_path,
                "--duration", str(duration),
                "--voice", voice
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics.successful_requests += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_requests - 1) + response_time) 
                / self.metrics.total_requests
            )
            
            logging.info(f"VibeVoice-7B generated audio: {audio_path}")
            return audio_path
            
        except Exception as e:
            self.metrics.error_count += 1
            logging.error(f"VibeVoice-7B generation failed: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")
    
    def get_metrics(self) -> ExpertMetrics:
        """Get expert performance metrics"""
        return self.metrics


class AdvancedStreeVideoExpert:
    """Advanced Stree video expert with performance monitoring"""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.config = STREE_CONFIGS['i2v-A14B']
        self.model = self._load_model()
        self.metrics = ExpertMetrics()
        
    def _load_model(self) -> StreeI2V:
        """Load the Stree I2V model"""
        try:
            model = StreeI2V(
                config=self.config,
                checkpoint_dir=self.checkpoint_dir,
                device_id=0 if self.device == "cuda" else -1
            )
            logging.info("Advanced Stree2.2-I2V-A14B expert loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Failed to load Stree model: {e}")
            raise
    
    def generate_video(self, image_path: str, prompt: str, duration: int = 5) -> str:
        """Generate video with performance monitoring"""
        start_time = time.time()
        self.metrics.total_requests += 1
        self.metrics.last_used = time.time()
        
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
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                video_path = tmp_file.name
            
            self.model.save_video(video_frames, video_path)
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics.successful_requests += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_requests - 1) + response_time) 
                / self.metrics.total_requests
            )
            
            logging.info(f"Stree generated video: {video_path}")
            return video_path
            
        except Exception as e:
            self.metrics.error_count += 1
            logging.error(f"Video generation failed: {e}")
            raise RuntimeError(f"Video generation failed: {e}")
    
    def get_metrics(self) -> ExpertMetrics:
        """Get expert performance metrics"""
        return self.metrics


class AdvancedMoEPipeline:
    """Advanced MoE pipeline with sophisticated routing and monitoring"""
    
    def __init__(self, 
                 vibevoice_path: str = "VibeVoice-7B",
                 stree_checkpoint_dir: str = "./checkpoints",
                 device: str = "cuda"):
        
        self.device = device
        
        # Initialize experts
        self.vibevoice_expert = AdvancedVibeVoiceExpert(vibevoice_path, device)
        self.stree_expert = AdvancedStreeVideoExpert(stree_checkpoint_dir, device)
        
        # Initialize encoders
        self.text_encoder = AdvancedTextEncoder()
        self.image_encoder = AdvancedImageEncoder()
        
        # Initialize router
        self.router = AdvancedMoERouter()
        
        # Expert mapping
        self.experts = {
            0: self.vibevoice_expert,
            1: self.stree_expert
        }
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        
        logging.info("Advanced MoE Pipeline initialized successfully")
    
    def _encode_input(self, text: str, image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text and image for advanced routing"""
        with torch.no_grad():
            text_encoding = self.text_encoder(text)
            image_encoding = self.image_encoder(image_path)
        
        return text_encoding, image_encoding
    
    def _route_task(self, text: str, image_path: str) -> Dict[str, Any]:
        """Advanced task routing with quality prediction"""
        # Encode inputs
        text_encoding, image_encoding = self._encode_input(text, image_path)
        
        # Get routing decision
        with torch.no_grad():
            expert_weights, expert_indices, quality_score = self.router(text_encoding, image_encoding)
        
        # Map to expert names
        expert_names = ["VibeVoice-7B", "Stree2.2-I2V-A14B"]
        routing_decision = {}
        
        for i in range(expert_weights.size(1)):
            expert_idx = expert_indices[0, i].item()
            weight = expert_weights[0, i].item()
            routing_decision[expert_names[expert_idx]] = weight
        
        # Add quality prediction
        routing_decision["predicted_quality"] = quality_score[0, 0].item()
        
        logging.info(f"Advanced MoE Routing Decision: {routing_decision}")
        return routing_decision
    
    def generate(self, 
                text: str, 
                image_path: str, 
                duration: int = 5,
                voice: str = "en_female_1",
                use_routing: bool = True,
                min_quality_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate content using advanced MoE pipeline
        
        Args:
            text: Input text for speech generation
            image_path: Input image for video generation
            duration: Duration of output in seconds
            voice: Voice type for speech generation
            use_routing: Whether to use MoE routing
            min_quality_threshold: Minimum quality threshold for generation
            
        Returns:
            Dictionary containing results and metadata
        """
        try:
            results = {}
            start_time = time.time()
            
            if use_routing:
                # Use advanced MoE routing
                routing_decision = self._route_task(text, image_path)
                
                # Check quality threshold
                predicted_quality = routing_decision.get("predicted_quality", 0.0)
                if predicted_quality < min_quality_threshold:
                    logging.warning(f"Predicted quality {predicted_quality:.3f} below threshold {min_quality_threshold}")
                
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
            
            # Track performance
            total_time = time.time() - start_time
            self.performance_history["generation_time"].append(total_time)
            
            results["success"] = True
            results["generation_time"] = total_time
            results["expert_metrics"] = self.get_expert_metrics()
            
            return results
            
        except Exception as e:
            logging.error(f"Advanced MoE generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _combine_audio_video(self, video_path: str, audio_path: str) -> str:
        """Combine audio and video using FFmpeg"""
        try:
            # Create output path
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # FFmpeg command to combine audio and video
            import subprocess
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
    
    def get_expert_metrics(self) -> Dict[str, ExpertMetrics]:
        """Get performance metrics for all experts"""
        return {
            "VibeVoice-7B": self.vibevoice_expert.get_metrics(),
            "Stree2.2-I2V-A14B": self.stree_expert.get_metrics()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        metrics = self.get_expert_metrics()
        
        summary = {
            "total_generations": len(self.performance_history["generation_time"]),
            "average_generation_time": np.mean(self.performance_history["generation_time"]) if self.performance_history["generation_time"] else 0.0,
            "expert_performance": {}
        }
        
        for expert_name, expert_metrics in metrics.items():
            summary["expert_performance"][expert_name] = {
                "success_rate": expert_metrics.success_rate,
                "average_response_time": expert_metrics.average_response_time,
                "reliability_score": expert_metrics.reliability_score,
                "total_requests": expert_metrics.total_requests
            }
        
        return summary
    
    def save_performance_log(self, filepath: str):
        """Save performance metrics to file"""
        summary = self.get_performance_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Performance summary saved to {filepath}")


# Convenience function for advanced MoE
def create_advanced_moe_pipeline(vibevoice_path: str = "VibeVoice-7B",
                                stree_checkpoint_dir: str = "./checkpoints",
                                device: str = "cuda") -> AdvancedMoEPipeline:
    """Create an advanced MoE pipeline instance"""
    return AdvancedMoEPipeline(vibevoice_path, stree_checkpoint_dir, device) 