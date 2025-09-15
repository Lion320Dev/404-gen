import os
import torch

import sys
from pathlib import Path
miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)
from pathlib import Path
from torchvision.io import read_image
from engine.metrics.quality_scorer import ImageQualityMetric
from diffusers import FluxPipeline
import gradio as gr

# Ensure xformers is used for attention
os.environ['ATTN_BACKEND'] = 'xformers'

# Load Flux pipeline once
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
).to("cuda")

# Load Image Quality Metric model once
image_quality_metric = ImageQualityMetric()
image_quality_metric.load_models()

def generate_and_score(
    prompt: str,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    max_sequence_length: int,
    height: int,
    width: int
):
    # Create generator with seed
    generator = torch.Generator("cuda").manual_seed(seed)

    # Generate image
    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        height=height,
        width=width,
        generator=generator
    ).images[0]

    # Save temporarily
    temp_path = "temp.png"
    image.save(temp_path)

    # Read and prepare for scoring
    img_tensor = read_image(temp_path).permute(1, 2, 0)  # (H, W, C)
    score = image_quality_metric.score_images_quality([img_tensor], "mean", False)

    return image, f"Quality score: {score}"

# Gradio UI
demo = gr.Interface(
    fn=generate_and_score,
    inputs=[
        gr.Textbox(label="Prompt", value="A trendy yellow silk scarf"),
        gr.Number(label="Seed", value=0, precision=0),
        gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance Scale"),
        gr.Slider(1, 50, value=8, step=1, label="Num Inference Steps"),
        gr.Slider(64, 1024, value=512, step=64, label="Max Sequence Length"),
        gr.Slider(128, 2048, value=512, step=64, label="Height"),
        gr.Slider(128, 2048, value=512, step=64, label="Width"),
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Quality Score")
    ],
    title="Flux Image Generator + Quality Scorer",
    description="Control all Flux parameters and evaluate image quality."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)