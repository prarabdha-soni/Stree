import numpy as np
import wave
import json
from pathlib import Path
import time
import os

def generate_synthetic_audio(duration, category, sample_rate=44100):
    """Generate synthetic audio data based on category"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Base frequency for the voice
    base_freq = 220  # Hz
    
    if category == "horror_laugh":
        # Generate a creepy laugh with varying frequency
        freq = base_freq * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        # Add some modulation
        audio *= (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
        
    elif category == "horror_whisper":
        # Generate a whisper with high frequency components
        audio = 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
        # Add some noise
        audio += 0.1 * np.random.randn(len(t))
        
    elif category == "horror_threat":
        # Generate a threatening voice with low frequency
        audio = 0.7 * np.sin(2 * np.pi * (base_freq * 0.8) * t)
        # Add some modulation
        audio *= (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
        
    elif category == "horror_claim":
        # Generate a claiming voice with medium frequency
        audio = 0.6 * np.sin(2 * np.pi * base_freq * t)
        # Add some echo effect
        echo = np.zeros_like(audio)
        echo[1000:] = audio[:-1000] * 0.3
        audio += echo
        
    else:
        # Default horror voice
        audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
        # Add some modulation
        audio *= (1 + 0.2 * np.sin(2 * np.pi * 4 * t))
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Convert to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)
    
    return audio, sample_rate

def save_audio(audio_data, filename, sample_rate=44100):
    """Save audio data to a WAV file"""
    with wave.open(str(filename), 'wb') as wf:  # Convert Path to string
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    print(f"Saved to {filename}")

def play_audio(filename):
    """Simulate playing audio (just print a message)"""
    print(f"Would play audio from {filename}")
    time.sleep(1)  # Simulate playback time

def main():
    # Create audio_files directory if it doesn't exist
    audio_dir = Path("training_data/audio_files")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the horror voice samples
    with open("training_data/hindi_horror.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("Welcome to the Horror Voice Generation Tool!")
    print("This tool will generate synthetic horror voice samples.")
    print("Each sample will be generated with different characteristics based on its category.")
    
    input("\nPress Enter to begin generating samples...")
    
    for i, sample in enumerate(data["samples"], 1):
        print(f"\nSample {i}/{len(data['samples'])}")
        print(f"Text: {sample['text']}")
        print(f"Category: {sample['category']}")
        
        # Calculate appropriate duration based on text length
        words = len(sample['text'].split())
        duration = max(3, words / 3)  # Minimum 3 seconds
        
        print(f"Generating {duration:.1f} seconds of audio...")
        
        # Generate synthetic audio
        audio_data, sample_rate = generate_synthetic_audio(duration, sample['category'])
        
        # Save the audio
        filename = audio_dir / sample['audio_file']
        save_audio(audio_data, filename, sample_rate)
        
        # Simulate playback
        print("\nSimulating playback...")
        play_audio(filename)
        
        print("Moving to next sample...")
        time.sleep(0.5)
    
    print("\nAll samples have been generated!")
    print(f"Audio files have been saved to {audio_dir}")
    print("\nNote: These are synthetic audio samples for testing purposes.")
    print("For real training data, you'll need to replace these with actual voice recordings.")

if __name__ == "__main__":
    main() 