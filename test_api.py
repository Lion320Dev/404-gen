# #!/usr/bin/env python3
# """
# Test script for the 3D Generation API
# """
# import requests
# import time
# import os
# import torch
# import io
# import csv
# import re
# from pathlib import Path
# from PIL import Image
# from engine.io.ply import PlyLoader
# from engine.rendering.renderer import Renderer
# from engine.validation_engine import ValidationEngine
# from engine.data_structures import GaussianSplattingData, TimeStat
# from engine.utils.gs_data_checker_utils import is_input_data_valid
# from engine.metrics.quality_scorer import ImageQualityMetric
# from engine.data_structures import GaussianSplattingData, ValidationResult

# renderer = Renderer()
# ply_loader = PlyLoader()
# validator = ValidationEngine()
# validator.load_pipelines()
# image_quality_metric = ImageQualityMetric()
# image_quality_metric.load_models()

# def prepare_input_data(assets: bytes, render_views_number=16, render_img_width=512, render_img_height=512, render_theta_angles=None):
#     time_stat = TimeStat()
#     t1 = torch.cuda.Event(enable_timing=True)
#     t2 = torch.cuda.Event(enable_timing=True)
    
#     pcl_buffer = io.BytesIO(assets)
#     gs_data: GaussianSplattingData = ply_loader.from_buffer(pcl_buffer)
    
#     if not is_input_data_valid(gs_data):
#         return None, [], time_stat
    
#     gs_data_gpu = gs_data.send_to_device(validator.device)
#     images = renderer.render_gs(
#         gs_data_gpu,
#         views_number=render_views_number,
#         img_width=render_img_width,
#         img_height=render_img_height,
#         theta_angles=render_theta_angles,
#     )
#     return gs_data_gpu, images, time_stat

# def combine_images16(images, img_width, img_height, gap=5, resize_factor=1.0):
#     # Grid size for 16 images (4x4)
#     grid_rows, grid_cols = 2, 2

#     # Final canvas size
#     row_width = img_width * grid_cols + gap * (grid_cols - 1)
#     column_height = img_height * grid_rows + gap * (grid_rows - 1)

#     combined_image = Image.new("RGB", (row_width, column_height), color="black")

#     # Convert to PIL if necessary
#     pil_images = [Image.fromarray(img.detach().cpu().numpy()) for img in images[:4]]

#     # Place each image in the 4x4 grid
#     for idx, img in enumerate(pil_images):
#         r = idx // grid_cols
#         c = idx % grid_cols
#         x = c * (img_width + gap)
#         y = r * (img_height + gap)
#         combined_image.paste(img, (x, y))

#     # Optional resize
#     if resize_factor != 1.0:
#         w, h = combined_image.size
#         combined_image = combined_image.resize(
#             (int(w * resize_factor), int(h * resize_factor)),
#             Image.Resampling.LANCZOS
#         )

#     return combined_image

# def ply_and_score(ply_path, prompt):
#         with open(ply_path, "rb") as f:
#             assets_flux = f.read()

#         _, gs_rendered_images_flux, _ = prepare_input_data(
#             assets_flux,
#             render_views_number=4,
#             render_img_width=512,
#             render_img_height=512,
#             render_theta_angles=[20.0, 120.0, 220.0, 310.0],
#         )
#         combined_image = combine_images16(
#             gs_rendered_images_flux[:4],
#             512,
#             512,
#             gap=5,
#         )

#         ply_data = ply_data_with_name(ply_path)
#         score_obj = custom_validator(ply_data, prompt)

#         return combined_image, score_obj

# def ply_data_with_name(ply_path: str) -> GaussianSplattingData:
#     loader = PlyLoader()
#     name_without_ext = os.path.splitext(ply_path)[0]
#     ply_path = Path(name_without_ext)
#     return loader.from_file(ply_path.name, str(ply_path.parent))

# def custom_validator(ply_data, prompt, cam_rad=2.5, ref_bbox_size=1.5):
#     ply_data = ply_data.send_to_device('cuda')
#     render = Renderer()
#     images = render.render_gs(ply_data, 16, 512, 512, cam_rad=cam_rad, ref_bbox_size=ref_bbox_size)
#     # render.save_rendered_images(images, prompt, "render_image")
#     score: ValidationResult = validator.validate_text_to_gs(prompt, images)
#     return score

# def text_to_filename(text: str, base_path: str = ".", extension: str = ".png", max_len: int = 10, dual_number: str = "") -> str:
#     """
#     Convert text to a safe filename with underscores, inside a given base path.
#     Truncate filename if it exceeds a max number of tokens (words).

#     Args:
#         text (str): Input text (e.g., "A cozy gray sweatshirt").
#         base_path (str): Directory path where the file will be saved.
#         extension (str): File extension (default: ".png").
#         max_len (int): Maximum number of words to keep in the filename.

#     Returns:
#         str: Full file path.
#     """
#     # lowercase, split words
#     words = text.lower().split()

#     # truncate if longer than max_len
#     if len(words) > max_len:
#         words = words[:max_len]

#     # join with underscores
#     filename = "_".join(words)

#     # remove invalid filename characters
#     filename = re.sub(r'[^a-z0-9_]', '', filename)

#     # ensure the directory exists
#     os.makedirs(base_path, exist_ok=True)

#     # full file path
#     return os.path.join(base_path, dual_number + extension)


# def test_api():
#     # API endpoint
#     url = "http://localhost:10006/generate/"
    
#     # Test prompt
#     # prompt = "a red apple"
    
#     save_path = "/root/subnet/result"

#     with open("/root/subnet/data/test_data.csv", newline='', encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:

#             pipeline_start_time = time.time()
#             origin_prompt = row["prompt"]
#             duel_number = row["duel_id"]
            
#             try:
#                 print("Testing API endpoint...")
                
#                 # Test the root endpoint first
#                 response = requests.get("http://localhost:10006/")
#                 print(f"Root endpoint status: {response.status_code}")
#                 if response.status_code == 200:
#                     print(f"Response: {response.json()}")
                
#                 print(f"Sending generation request with prompt: '{origin_prompt}'")
                
#                 # Send generation request
#                 response = requests.post(
#                     url,
#                     data={"prompt": origin_prompt},
#                     timeout=300  # 5 minutes timeout
#                 )
#                 pipeline_end_time = time.time()
                
#                 if response.status_code == 200:
#                     print("Success! Received PLY file data")
#                     print(f"Response size: {len(response.content)} bytes")
                    
#                     # Save the PLY file
#                     ply_save_path = "generated_model.ply"
#                     with open(ply_save_path, "wb") as f:
#                         f.write(response.content)
#                     print("Saved as 'generated_model.ply'")
#                     combined_image, score_obj = ply_and_score(ply_save_path, origin_prompt)
#                     score_A = score_obj.alignment_score
#                     score_Q = score_obj.combined_quality_score
#                     score_F = score_obj.final_score
#                     dalle_combine_path = text_to_filename(origin_prompt, base_path=save_path, extension="_imageof3D_combine.png", dual_number=duel_number)

#                     if combined_image is not None:
#                         combined_image.save(dalle_combine_path)
#                     else:
#                         print("combined_image_dalle is None, skipping save.")
                    
#                     # ✅ Always write CSV, even on error
#                     csv_file = os.path.join(save_path, "log.csv")
#                     os.makedirs(save_path, exist_ok=True)

#                     header = ["duel_id", "prompt",
#                             "score_A", "score_Q", "score_F", "duel", "Time", "Image Gen", "Empty PLY", "using_api", "duel issue", "file_path"]
                    
#                     file_exists = os.path.isfile(csv_file)
#                     with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                         writer = csv.writer(f)
#                         if not file_exists:
#                             writer.writerow(header)
#                         writer.writerow([
#                             duel_number,
#                             origin_prompt,
#                             score_A,
#                             score_Q,
#                             score_F,
#                             pipeline_end_time - pipeline_start_time,
#                         ])
#                 else:
#                     print(f"Error: HTTP {response.status_code}")
#                     print(f"Response: {response.text}")
                    
#             except requests.exceptions.RequestException as e:
#                 print(f"Request failed: {e}")
#             except Exception as e:
#                 print(f"Error: {e}")

# if __name__ == "__main__":
#     # Wait a bit for the server to start
#     # print("Waiting 10 seconds for server to be ready...")
#     # time.sleep(10)
#     test_api()

#!/usr/bin/env python3
"""
Test script for the 3D Generation API
"""
import requests
import time
import os
import torch
import io
import re
from pathlib import Path
from PIL import Image
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.validation_engine import ValidationEngine
from engine.data_structures import GaussianSplattingData, TimeStat, ValidationResult
from engine.utils.gs_data_checker_utils import is_input_data_valid
from engine.metrics.quality_scorer import ImageQualityMetric

renderer = Renderer()
ply_loader = PlyLoader()
validator = ValidationEngine()
validator.load_pipelines()
image_quality_metric = ImageQualityMetric()
image_quality_metric.load_models()


def prepare_input_data(assets: bytes, render_views_number=16, render_img_width=512, render_img_height=512, render_theta_angles=None):
    time_stat = TimeStat()
    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)

    pcl_buffer = io.BytesIO(assets)
    gs_data: GaussianSplattingData = ply_loader.from_buffer(pcl_buffer)

    if not is_input_data_valid(gs_data):
        return None, [], time_stat

    gs_data_gpu = gs_data.send_to_device(validator.device)
    images = renderer.render_gs(
        gs_data_gpu,
        views_number=render_views_number,
        img_width=render_img_width,
        img_height=render_img_height,
        theta_angles=render_theta_angles,
    )
    return gs_data_gpu, images, time_stat


def combine_images16(images, img_width, img_height, gap=5, resize_factor=1.0):
    grid_rows, grid_cols = 2, 2
    row_width = img_width * grid_cols + gap * (grid_cols - 1)
    column_height = img_height * grid_rows + gap * (grid_rows - 1)
    combined_image = Image.new("RGB", (row_width, column_height), color="black")
    pil_images = [Image.fromarray(img.detach().cpu().numpy()) for img in images[:4]]

    for idx, img in enumerate(pil_images):
        r = idx // grid_cols
        c = idx % grid_cols
        x = c * (img_width + gap)
        y = r * (img_height + gap)
        combined_image.paste(img, (x, y))

    if resize_factor != 1.0:
        w, h = combined_image.size
        combined_image = combined_image.resize(
            (int(w * resize_factor), int(h * resize_factor)),
            Image.Resampling.LANCZOS
        )
    return combined_image


def ply_and_score(ply_path, prompt):
    with open(ply_path, "rb") as f:
        assets_flux = f.read()

    _, gs_rendered_images_flux, _ = prepare_input_data(
        assets_flux,
        render_views_number=4,
        render_img_width=512,
        render_img_height=512,
        render_theta_angles=[20.0, 120.0, 220.0, 310.0],
    )
    combined_image = combine_images16(
        gs_rendered_images_flux[:4],
        512,
        512,
        gap=5,
    )

    ply_data = ply_data_with_name(ply_path)
    score_obj = custom_validator(ply_data, prompt)
    return combined_image, score_obj


def ply_data_with_name(ply_path: str) -> GaussianSplattingData:
    loader = PlyLoader()
    name_without_ext = os.path.splitext(ply_path)[0]
    ply_path = Path(name_without_ext)
    return loader.from_file(ply_path.name, str(ply_path.parent))


def custom_validator(ply_data, prompt, cam_rad=2.5, ref_bbox_size=1.5):
    ply_data = ply_data.send_to_device('cuda')
    render = Renderer()
    images = render.render_gs(ply_data, 16, 512, 512, cam_rad=cam_rad, ref_bbox_size=ref_bbox_size)
    score: ValidationResult = validator.validate_text_to_gs(prompt, images)
    return score


def text_to_filename(text: str, base_path: str = ".", extension: str = ".png", max_len: int = 10, dual_number: str = "") -> str:
    words = text.lower().split()
    if len(words) > max_len:
        words = words[:max_len]
    filename = "_".join(words)
    filename = re.sub(r'[^a-z0-9_]', '', filename)
    os.makedirs(base_path, exist_ok=True)
    return os.path.join(base_path, filename + dual_number + extension)


def send_prompt(prompt, duel_number):
    url = "http://localhost:10006/generate/"
    save_path = "/root/subnet/result_1"

    try:
        pipeline_start_time = time.time()
        response = requests.post(
            url,
            data={"prompt": prompt},
            timeout=300
        )
        pipeline_end_time = time.time()

        if response.status_code == 200:
            print(f"Success! Prompt '{prompt}' generated PLY file.")
            ply_save_path = f"generated_model_{duel_number}.ply"
            with open(ply_save_path, "wb") as f:
                f.write(response.content)

            combined_image, score_obj = ply_and_score(ply_save_path, prompt)
            score_A = score_obj.alignment_score
            score_Q = score_obj.combined_quality_score
            score_F = score_obj.final_score

            img_save_path = text_to_filename(
                prompt,
                base_path=save_path,
                extension="_imageof3D_combine.png",
                dual_number=f"_{duel_number}"
            )

            if combined_image is not None:
                combined_image.save(img_save_path)
                print(f"Saved combined image to {img_save_path}")
            else:
                print("combined_image is None, skipping save.")

            print(f"Scores → A: {score_A}, Q: {score_Q}, F: {score_F}, Time: {pipeline_end_time - pipeline_start_time:.2f}s")
        else:
            print(f"Error: HTTP {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")


def test_api():
    prompts = [
        ("METALLIC GOLD CRICKET BAT BALANCED PERFECTLY", "1"),
        ("STURDY BRONZE CYMBAL", "2"),
        ("BLUE HORSE GRAZING IN MEADOW", "3"),
        ("ONYX GREAVES PAIRED WITH SILVER SPURS", "4"),
        ("ORCHID BLOOM ON BROWN POT", "5"),
        ("BURGUNDY VELVET CAPE LINED IN SATIN", "6"),
        ("A RHODODENDRON VASE WITH A GLOSSY, DEEP RED FINISH AND A SMOOTH, CYLINDRICAL SHAPE", "7"),
    ]
    for prompt, duel_number in prompts:
        send_prompt(prompt, duel_number)
        # time.sleep(1)  # wait before sending next prompt


if __name__ == "__main__":
    test_api()
