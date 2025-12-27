import os
from datetime import datetime

from diffusers import AutoPipelineForText2Image
import torch

import transformers
import accelerate
import ollama

output_directory = "output"
method = "cpu"
dtype = torch.float16

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    method = "cuda"
    dtype = torch.float32

def generate_image(prompt):
    # Load the pipeline (this might take time on the first run)
    # 'stabilityai/stable-diffusion-xl-base-1.0' is a common choice
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        #"stabilityai/sd-turbo",
        dtype=dtype,
        variant="fp16",
        use_safetensors=True
    )
    # Ensure the pipeline runs on the GPU if available
    pipeline.to(method)
    # Generate the image using the refined prompt
    image = pipeline(prompt, num_inference_steps=10, guidance_scale=5).images[0]
    # Save or display the image
    os.makedirs(output_directory, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    output_file = os.path.join(output_directory, f"generated_image-{timestamp}.png")
    image.save(output_file)
    print(f"Image saved as {output_file}")
    return output_file