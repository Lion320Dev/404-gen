import os
from io import BytesIO
import torch
from pathlib import Path
import sys
from pathlib import Path
from time import time
import logging
import asyncio
from fastapi import FastAPI, Form, Depends
from fastapi.responses import Response
from contextlib import asynccontextmanager
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.validation_engine import ValidationEngine
from engine.data_structures import GaussianSplattingData, ValidationResult
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from trellis.pipelines import TrellisImageTo3DPipeline
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.validation_engine import ValidationEngine
from engine.data_structures import GaussianSplattingData
from engine.metrics.quality_scorer import ImageQualityMetric

from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler, TorchAoConfig, AutoModel
import math
from openai import OpenAI

# OpenAI key
openai = 

currentPath = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, currentPath)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading models...")
    
    # Store models in app state
    app.state.models = {}
    
    if using_qwen:
        # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": False,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        quantization_config = TorchAoConfig("int8wo")

        transformer = AutoModel.from_pretrained("Qwen/Qwen-Image", subfolder="transformer", device_map="auto",
                                                quantization_config=quantization_config,
                                                torch_dtype=torch.bfloat16)

        qwen_pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image", scheduler=scheduler, transformer=transformer, torch_dtype=torch.bfloat16
        ).to("cuda")
        qwen_pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
        )
        app.state.models["qwen_pipe"] = qwen_pipe

    # Trellis Load
    ckpt_path = "./DSO-finetuned-TRELLIS/dpo-8000iters.safetensors"  # "./DSO-finetuned-TRELLIS/dpo-8000iters.safetensors"
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("nuvita2811/trellis_hi3dgen")
    trellis_pipeline.to(device)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
    )
    trellis_pipeline.models["sparse_structure_flow_model"] = get_peft_model(trellis_pipeline.models["sparse_structure_flow_model"], peft_config)
    trellis_pipeline.models["sparse_structure_flow_model"].load_state_dict(load_file(ckpt_path))
    app.state.models["trellis_pipeline"] = trellis_pipeline

    renderer = Renderer()
    app.state.models["renderer"] = renderer
    
    ply_loader = PlyLoader()
    app.state.models["ply_loader"] = ply_loader
    
    validator = ValidationEngine()
    validator.load_pipelines()
    app.state.models["validator"] = validator
    
    image_quality_metric = ImageQualityMetric()
    image_quality_metric.load_models()
    app.state.models["image_quality_metric"] = image_quality_metric
    
    logger.info("Models loaded successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# -----------------------------
# Environment and device
# -----------------------------

# os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
device = "cuda" if torch.cuda.is_available() else "cpu"

using_qwen = True

app = FastAPI(
    title="3D Generation API", 
    description="API for generating 3D models from text prompts",
    lifespan=lifespan
)

# -----------------------------
# Helper Functions
# -----------------------------

async def load_empty_ply() -> BytesIO:
    """Load and return empty.ply file as BytesIO buffer."""
    try:
        with open("empty.ply", 'rb') as f:
            ply_content = f.read()
        buffer = BytesIO(ply_content)
        buffer.seek(0)
        return buffer
    except FileNotFoundError:
        logger.warning("Could not find empty.ply, returning empty buffer")
        buffer = BytesIO()
        buffer.write(b"")
        buffer.seek(0)
        return buffer

def generate_qwen_img(qwen_pipe, prompt, seed_dalle):
    generator = torch.Generator(device="cuda").manual_seed(seed_dalle)
    negative_prompt = " "
    result = qwen_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=4,
        true_cfg_scale=1.0,
        generator=generator,
    ).images[0]
    return result
    
def ply_data_with_name(ply_loader, ply_path: str) -> GaussianSplattingData:
    name_without_ext = os.path.splitext(ply_path)[0]
    ply_path = Path(name_without_ext)
    return ply_loader.from_file(ply_path.name, str(ply_path.parent))

def custom_validator(validator, ply_data, prompt, cam_rad=2.5, ref_bbox_size=1.5):
    ply_data = ply_data.send_to_device('cuda')
    render = Renderer()
    images = render.render_gs(ply_data, 16, 512, 512, cam_rad=cam_rad, ref_bbox_size=ref_bbox_size)
    # render.save_rendered_images(images, prompt, "render_image")
    score: ValidationResult = validator.validate_text_to_gs(prompt, images)
    return score
  
def expand_prompt(origin_prompt: str) -> str:
        """Helper function to expand origin prompt via GPT."""
        template = f"""
        You are a prompt expander specialized in 3D asset generation. 
        I will give you a short origin prompt that only names the subject, such as:
        - DRONE WITH RED UNDERBELLY
        - PLUSH YELLOW BLANKET ON BED
        - CELLO GLOSSY BLACK VARNISH
        - VIOLIN
        - HARP
        - ...

        Your task: expand it into a detailed description suitable for generating 3D assets 
        in a neutral preview style. Follow this format:

        "The asset is a clean 3D model of [expanded subject]. The object should be displayed 
        in a vertical position orientation, fully visible, and centered in the frame. The camera view 
        should be straightforward (not cinematic, not catoonish), as if showing a preview of a 3D asset in 
        a modeling tool. Lighting should be flat and uniform, avoiding dramatic shadows or 
        highlights. The object should have realistic materials and a consistent scale, 
        with slightly muted colors. The background should be plain white, ensuring the object 
        is the only focus. The rendering should look like a technical preview of a 3D asset, 
        not a product photo or artistic render. Object must be placed on walls, not on the floor."

        Special rules:
        - Remove mismath Style and extra object for prompt
        - Update mismath color for prompt and missing element for prompt
        - For violin, viola, cello, double bass, guitar, etc.:
        * Only the body and structural parts should be modeled.
        * Do NOT include the strings themselves.
        * Always display the instrument in a vertical standing position.
        - For harp:
        * Only the body and frame should be modeled.
        * Do NOT include the strings.
        * The harp should be displayed in a sideways orientation (not upright).
        - For drones: describe the body as close to spherical, with simple arms and propellers, avoiding modern or futuristic aesthetics.
        - For the cup, there should be a saucer or coaster  for the cup and it should be adjusted so that the contents of the cup are raised evenly above the cup so that they can be seen.
        - Keep geometry stable and remains balanced under gravity, clear, suitable for reuse in 3D pipelines.
        - Avoid close-ups, tilted angles, or artistic compositions.
        - Always display in a vertical standing position
        
        Now expand this origin prompt:
        [{origin_prompt}]
        """
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": template}]
        )

        return response.choices[0].message.content

def run_trellis_and_score(models, imageOf3D, prompt):
        """Helper function to run trellis pipeline and scoring."""
        trellis_pipeline = models["trellis_pipeline"]
        ply_loader = models["ply_loader"]
        validator = models["validator"]
        
        trellis_outputs = trellis_pipeline.run(
            imageOf3D,
            seed=42,
            formats=["gaussian"],
            sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 25, "cfg_strength": 3},
        )

        # ply_dalle_path = "temp.ply"
        gaussian = trellis_outputs['gaussian'][0]
        # gaussian.save_ply(ply_dalle_path)
        ply_data = gaussian.convert_ply()

        # ply_data = ply_data_with_name(ply_loader, ply_dalle_path)
        score_obj = custom_validator(validator, ply_data, prompt)

        return gaussian, score_obj

async def run_trellis_and_score_with_timeout(models, imageOf3D, prompt, timeout_seconds=25):
    """Async wrapper for run_trellis_and_score with timeout."""
    try:
        # Run the synchronous function in a thread pool to make it cancellable
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, run_trellis_and_score, models, imageOf3D, prompt)
        
        # Wait for completion or timeout
        return await asyncio.wait_for(task, timeout=timeout_seconds)
        
    except asyncio.TimeoutError:
        logger.warning(f"3D generation timed out after {timeout_seconds} seconds")
        raise

async def generate_qwen_img_with_timeout(models, prompt, seed_dalle, timeout_seconds=25):
    """Async wrapper for generate_qwen_img with timeout."""
    try:
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, generate_qwen_img, models["qwen_pipe"], prompt, seed_dalle)
        
        return await asyncio.wait_for(task, timeout=timeout_seconds)
        
    except asyncio.TimeoutError:
        logger.warning(f"Qwen image generation timed out after {timeout_seconds} seconds")
        raise
    
def ply_and_score(models, ply_path, prompt):
    ply_loader = models["ply_loader"]
    validator = models["validator"]
    ply_data = ply_data_with_name(ply_loader, ply_path)
    score_obj = custom_validator(validator, ply_data, prompt)

    return score_obj

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
async def root():
    return {"message": "3D Generation API is running"}

@app.post("/generate/")
async def generate(
    prompt: str = Form(),
) -> Response:
    """Generate 3D model from text prompt with 25-second timeout"""
    t0 = time()
    logger.info(f"Starting generation for prompt: {prompt}")
    
    models = app.state.models
    
    # Default parameters
    seed_qwen_1 = 42
    seed_qwen_2 = 0
    ply_path = "empty.ply"
    
    try:
        if using_qwen and "qwen_pipe" in models:
            logger.info("Generating Qwen Image using Origin prompt...")
            
            # Try generation with timeout
            try:
                # Generate image with timeout
                new_prompt = prompt + "。 3D用，直立，浅色背景。"
                imageOf3D = await generate_qwen_img_with_timeout(models, new_prompt, seed_qwen_1, timeout_seconds=25)
                
                # Run trellis and scoring with remaining timeout
                elapsed_0 = time() - t0
                remaining_timeout = max(1, 25 - elapsed_0)  # At least 1 second remaining
                
                gaussian, score_obj = await run_trellis_and_score_with_timeout(models, imageOf3D, prompt, timeout_seconds=remaining_timeout)
                score_flux_final_origin = score_obj.final_score

                logger.info(f"Image Quality Score: {score_flux_final_origin}")
                # If score is low, try different approaches
                if score_flux_final_origin < 0.61:
                    logger.info("Generating Qwen Image using Origin prompt seed 42...")
                    new_prompt = prompt + "。 3D用，直立，浅色背景。"
                    imageOf3D = await generate_qwen_img_with_timeout(models, new_prompt, seed_qwen_2, timeout_seconds=25)
                    
                    # Run trellis and scoring with remaining timeout
                    elapsed_42 = time() - t0
                    remaining_timeout = max(1, 25 - elapsed_42)  # At least 1 second remaining
                    
                    gaussian, score_obj = await run_trellis_and_score_with_timeout(models, imageOf3D, prompt, timeout_seconds=remaining_timeout)
                    score_flux_final_origin = score_obj.final_score
                    
                    # If score is low, use empty PLY
                    if score_flux_final_origin < 0.61:
                        logger.info("Score too low, using empty PLY")
                        buffer = await load_empty_ply()
                        
                        t1 = time()
                        logger.info(f"Generation took: {(t1 - t0):.2f} seconds")
                        return Response(bytes(buffer.getbuffer()), media_type="application/octet-stream")
                
                # Save successful result
                buffer = BytesIO()
                gaussian.save_ply_cuda(buffer)
                buffer.seek(0)
                
            except asyncio.TimeoutError:
                logger.warning("Generation timed out after 25 seconds, returning empty PLY")
                buffer = await load_empty_ply()
                
                t1 = time()
                logger.info(f"Generation timed out after: {(t1 - t0):.2f} seconds")
                return Response(bytes(buffer.getbuffer()), media_type="application/octet-stream")
        
        else:
            logger.warning("Qwen pipeline not available, using empty PLY")
            buffer = await load_empty_ply()
        
        t1 = time()
        logger.info(f"Generation took: {(t1 - t0):.2f} seconds")
        
        return Response(bytes(buffer.getbuffer()), media_type="application/octet-stream")
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        # Return empty PLY file on error
        buffer = await load_empty_ply()
        return Response(bytes(buffer.getbuffer()), media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10006)
        
