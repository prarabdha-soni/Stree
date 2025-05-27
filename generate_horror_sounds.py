import numpy as np
import soundfile as sf
from scipy import signal
import librosa
import os

def create_horror_laugh():
    # Create a creepy laugh using frequency modulation
    duration = 3.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base frequency that modulates
    base_freq = 200
    mod_freq = 5
    carrier_freq = 400
    
    # Create modulation
    mod = np.sin(2 * np.pi * mod_freq * t)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    
    # Combine signals
    audio = 0.7 * carrier * (1 + 0.5 * mod) + 0.3 * noise
    
    # Add reverb effect
    reverb = signal.convolve(audio, np.exp(-np.linspace(0, 5, 1000)))[:len(audio)]
    audio = 0.7 * audio + 0.3 * reverb
    
    return audio, sr

def create_ghost_whisper():
    # Create a ghostly whisper
    duration = 4.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate noise
    noise = np.random.normal(0, 1, len(t))
    
    # Apply bandpass filter for whisper effect
    b, a = signal.butter(4, [300/(sr/2), 3000/(sr/2)], btype='band')
    whisper = signal.filtfilt(b, a, noise)
    
    # Add some modulation
    mod = np.sin(2 * np.pi * 0.5 * t)
    audio = whisper * (0.5 + 0.5 * mod)
    
    return audio, sr

def create_haunted_cry():
    # Create a haunted crying sound
    duration = 5.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create base tone
    freq = np.linspace(400, 200, len(t))
    base = np.sin(2 * np.pi * freq * t)
    
    # Add tremolo
    tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    audio = base * tremolo
    
    # Add reverb
    reverb = signal.convolve(audio, np.exp(-np.linspace(0, 3, 1000)))[:len(audio)]
    audio = 0.6 * audio + 0.4 * reverb
    
    return audio, sr

def create_evil_laugh():
    # Create an evil laugh
    duration = 4.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create base frequency sweep
    freq = np.linspace(300, 600, len(t))
    base = np.sin(2 * np.pi * freq * t)
    
    # Add distortion
    audio = np.tanh(base * 2)
    
    # Add some noise
    noise = np.random.normal(0, 0.2, len(t))
    audio = 0.8 * audio + 0.2 * noise
    
    return audio, sr

def create_ghost_warning():
    # Create a ghostly warning sound
    duration = 3.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create eerie tone
    freq = 250
    base = np.sin(2 * np.pi * freq * t)
    
    # Add modulation
    mod = np.sin(2 * np.pi * 2 * t)
    audio = base * (0.5 + 0.5 * mod)
    
    # Add reverb
    reverb = signal.convolve(audio, np.exp(-np.linspace(0, 4, 1000)))[:len(audio)]
    audio = 0.7 * audio + 0.3 * reverb
    
    return audio, sr

def create_demonic_voice():
    # Create a demonic voice effect
    duration = 4.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create base tone
    freq = 150
    base = np.sin(2 * np.pi * freq * t)
    
    # Add distortion and modulation
    audio = np.tanh(base * 3)
    mod = np.sin(2 * np.pi * 1.5 * t)
    audio = audio * (0.5 + 0.5 * mod)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    audio = 0.9 * audio + 0.1 * noise
    
    return audio, sr

def create_horror_sound():
    # Create a general horror sound
    duration = 3.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create multiple frequencies
    freqs = [200, 400, 600]
    audio = np.zeros_like(t)
    
    for freq in freqs:
        audio += np.sin(2 * np.pi * freq * t)
    
    # Add noise and modulation
    noise = np.random.normal(0, 0.2, len(t))
    mod = np.sin(2 * np.pi * 3 * t)
    audio = 0.7 * audio + 0.3 * noise
    audio = audio * (0.5 + 0.5 * mod)
    
    return audio, sr

def create_possessed_voice():
    # Create a possessed voice effect
    duration = 4.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create base tone with frequency sweep
    freq = np.linspace(200, 400, len(t))
    base = np.sin(2 * np.pi * freq * t)
    
    # Add distortion and modulation
    audio = np.tanh(base * 2.5)
    mod = np.sin(2 * np.pi * 2 * t)
    audio = audio * (0.5 + 0.5 * mod)
    
    # Add reverb
    reverb = signal.convolve(audio, np.exp(-np.linspace(0, 3, 1000)))[:len(audio)]
    audio = 0.8 * audio + 0.2 * reverb
    
    return audio, sr

def main():
    # Create output directory if it doesn't exist
    output_dir = "horror_audio_prompts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save all horror sounds
    sound_generators = {
        "horror_laugh_1.wav": create_horror_laugh,
        "ghost_whisper.wav": create_ghost_whisper,
        "haunted_cry.wav": create_haunted_cry,
        "evil_laugh.wav": create_evil_laugh,
        "ghost_warning.wav": create_ghost_warning,
        "demonic_voice.wav": create_demonic_voice,
        "horror_sound.wav": create_horror_sound,
        "possessed_voice.wav": create_possessed_voice
    }
    
    for filename, generator in sound_generators.items():
        print(f"Generating {filename}...")
        audio, sr = generator()
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save the file
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, audio, sr)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main() 