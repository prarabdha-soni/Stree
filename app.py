import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    dtype_map = {
        "cpu": "float32",
        "mps": "float32",  # Apple M series – better with float32
        "cuda": "float16",  # NVIDIA – better with float16
    }

    dtype = dtype_map.get(device.type, "float16")
    print(f"Using device: {device}, attempting to load model with {dtype}")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype=dtype, device=device)
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise


def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, device  # Access global model, config, device

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    temp_txt_file_path = None
    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            # Check if audio_data is valid
            if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:  # Check for silence/empty
                gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.")
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(f"Failed to convert audio prompt to float32: {conv_e}")

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                            )
                            audio_data = (
                                audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(audio_data)  # Ensure contiguous after slicing/mean

                    # Write using soundfile
                    try:
                        sf.write(
                            temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                        )  # Explicitly use FLOAT subtype
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})")
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise gr.Error(f"Failed to save audio prompt: {write_e}")

        # 3. Run Generation

        start_time = time.time()

        # Use torch.inference_mode() context manager for the generation call
        with torch.inference_mode():
            output_audio_np = model.generate(
                text_input,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,  # Pass the value here
                use_torch_compile=False,  # Keep False for Gradio stability
                audio_prompt=prompt_path_for_generate,
            )

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # 4. Convert Codes to Audio
        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(original_len / speed_factor)  # Target length based on speed_factor
            if target_len != original_len and target_len > 0:  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (
                    output_sr,
                    resampled_audio_np.astype(np.float32),
                )  # Use resampled audio
                print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
            else:
                output_audio = (
                    output_sr,
                    output_audio_np,
                )  # Keep original if calculation fails or no change
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            # --- End slowdown ---

            print(f"Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}")

            # Explicitly convert to int16 to prevent Gradio warning
            if output_audio[1].dtype == np.float32 or output_audio[1].dtype == np.float64:
                audio_for_gradio = np.clip(output_audio[1], -1.0, 1.0)
                audio_for_gradio = (audio_for_gradio * 32767).astype(np.int16)
                output_audio = (output_sr, audio_for_gradio)
                print("Converted audio to int16 for Gradio output.")

        else:
            print("\nGeneration finished, but no valid tokens were produced.")
            # Return default silence
            gr.Warning("Generation produced no output.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        # Re-raise as Gradio error to display nicely in the UI
        raise gr.Error(f"Inference failed: {e}")

    finally:
        # 5. Cleanup Temporary Files defensively
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f"Deleted temporary text file: {temp_txt_file_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}")
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}")

    return output_audio


# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
/* Add styles to make the buttons full width and stacked */
#button-container { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
#button-container > div { flex-grow: 1; }
/* Style the buttons with orange background */
#generate-sound-button button, #play-audio-button button {
    background-color: #FFA500; /* A shade of orange */
    color: white;
}
#generate-sound-button button:hover, #play-audio-button button:hover {
    background-color: #FF8C00; /* Darker shade on hover */
}
/* Style the generated audio label */
#generated-audio-label {
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 5px;
}
"""
# Attempt to load default text from example.txt
default_text = "[S1] कहानी अनजानी है रेनू... तुझे भी तो दिखी? \n[S2] खेल हुआ, खिलौने के पीछे... उस लाल साड़ी वाली... \n[S1] सोचते हुए नहीं, वो जब दरवाज़े के पीछे खड़ी है। \n[S2] रोते हुए मगर... वो हमारी सदा की आवाज़ सुन रही है..."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# Build Gradio UI
with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 0; font-size: 2.5em;'>India's 1st AI स्त्री ✨</h1><p style='text-align: center; margin-top: 0; font-size: 1em;'>with <span style='color: green; font-size: 1.8em;'>emotion</span>.</p>")

    with gr.Column(variant="panel", scale=1, elem_id="col-container"):
        text_input = gr.Textbox(
            label=None,
            placeholder=None, # Remove placeholder
            value=default_text,
            lines=8, # Increase lines
            container=False # Remove default container
        )

        with gr.Column(elem_id="button-container"):
            generate_button = gr.Button("✏️ Generate Sound", variant="primary", elem_id="generate-sound-button")
            play_button = gr.Button("▶️ Play Audio", variant="primary", elem_id="play-audio-button")

        # Move audio prompt input and generation parameters into an Accordion
        with gr.Accordion("Advanced Options", open=False):
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                show_label=True,
                sources=["upload", "microphone"],
                type="numpy",
            )
            with gr.Accordion("Generation Parameters", open=False):
                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=3072,
                    value=model.config.data.audio_length,  # Use config default if available, else fallback
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,  # Default from inference.py
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=1.0,
                    maximum=1.5,
                    value=1.3,  # Default from inference.py
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,  # Default from inference.py
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=50,
                    value=30,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=0.94,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )

        # Add the output audio component clearly
        # Note: Gradio's gr.Audio includes its own play button
        # The separate 'Play Audio' button above will likely need JavaScript to trigger playback
        # For now, we'll keep the gr.Audio component visible below.
        gr.Markdown("<p id='generated-audio-label'>Generated Audio</p>")
        audio_output = gr.Audio(
            label=None, # Remove default label as we added a markdown label
            type="numpy",
            autoplay=False,
            interactive=False # Make it non-interactive initially
        )

    # Link generate button click to function
    generate_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input, # Keep audio prompt as input
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
        ],
        outputs=[audio_output], # Output the audio to the audio_output component
        api_name="generate_audio",
    )

    # Link play button click - This will require custom JavaScript to trigger the audio_output playback.
    # Gradio's Python API doesn't have a direct way to trigger playback of another component.
    # We'll leave this button linked but it won't function without frontend JS.
    # A simpler approach might be to rely solely on the built-in play button of audio_output
    # once it's populated, but the image shows a separate button.
    # For now, I will just link it to a dummy function or leave it unlinked.
    # Linking it to the same output might visually update it but won't trigger play.
    # Let's link it to a dummy function that just returns the current audio output.
    def play_generated_audio(audio_data):
        # In a real scenario, this would need JS to call .play() on the audio element
        # This Python function doesn't actually *play* audio in the browser.
        print("Play button clicked (requires frontend JS for actual playback)")
        return audio_data # Just return the audio data to potentially update the component

    play_button.click(
        fn=play_generated_audio,
        inputs=[audio_output], # Take the generated audio as input
        outputs=[audio_output] # Output to the same audio component
    )

    # Add examples (ensure the prompt path is correct or remove it if example file doesn't exist)
    example_prompt_path = "./example_prompt.mp3"  # Adjust if needed
    examples_list = [

    ]

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[
                text_input,
                audio_prompt_input,
                max_new_tokens,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                speed_factor_slider,
            ],
            outputs=[audio_output],
            fn=run_inference,
            cache_examples=False,
            label="Examples (Click to Run)",
        )
    else:
        gr.Markdown("_(No examples configured or example prompt file missing)_")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")

    # set `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT` env vars to override default values
    # use `GRADIO_SERVER_NAME=0.0.0.0` for Docker
    demo.launch(share=args.share)
