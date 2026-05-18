import os
from datetime import datetime

from diffusers import AutoPipelineForText2Image
import torch
import xformers  # noqa: F401  — registers optimised attention kernels in PyTorch on import

output_directory = "output"
method = "cpu"
dtype = torch.float16

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    method = "cuda"
# Global pipeline instance - loaded once and reused
_pipeline = None

def get_pipeline():
    """Lazy load and return the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        print("Loading SDXL pipeline (first time only)...")
        _pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True
        ).to(method)

        # Enable VAE slicing for better memory efficiency
        _pipeline.enable_vae_slicing()

        # Enable xformers if available for faster attention
        try:
            _pipeline.enable_xformers_memory_efficient_attention()
            print("xformers enabled for faster attention")
        except Exception:
            print("xformers not available, using default attention")

        # Compile UNet for faster inference (PyTorch 2.0+)
        # Note: torch.compile requires triton which doesn't work on Windows
        # Skipping compilation to avoid runtime errors
        # On Linux/Mac with triton installed, you can uncomment this block for 20-30% speedup
        # try:
        #     if hasattr(torch, 'compile') and method == "cuda":
        #         print("Attempting to compile UNet model (one-time cost)...")
        #         _pipeline.unet = torch.compile(_pipeline.unet, mode="default")
        #         print("UNet compiled successfully")
        # except Exception as e:
        #     print(f"torch.compile failed: {type(e).__name__}")

    return _pipeline

def encode_prompt_xl(pipeline, prompt, negative_prompt=""):
    """Encode prompts for SDXL using both text encoders to bypass 77 token limit."""
    # SDXL has two text encoders
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2] if hasattr(pipeline, 'tokenizer_2') else [pipeline.tokenizer]
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2] if hasattr(pipeline, 'text_encoder_2') else [pipeline.text_encoder]

    prompt_embeds_list = []
    pooled_prompt_embeds = None

    # Process with both text encoders
    for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
        max_length = tokenizer.model_max_length
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).input_ids.to(pipeline.device)

        # Split into chunks if needed
        if input_ids.shape[-1] > max_length:
            chunks = []
            for i in range(0, input_ids.shape[-1], max_length - 2):
                chunk = input_ids[:, i:i + max_length - 2]
                chunk = torch.cat([
                    input_ids[:, :1],
                    chunk,
                    input_ids[:, -1:] if i + max_length - 2 >= input_ids.shape[-1] else input_ids[:, :1]
                ], dim=-1)
                chunks.append(chunk)

            # Encode each chunk
            chunk_embeds_list = []
            for chunk in chunks:
                outputs = text_encoder(chunk, output_hidden_states=True)
                chunk_embeds_list.append(outputs.hidden_states[-2])

            encoder_prompt_embeds = torch.cat(chunk_embeds_list, dim=1)

            # Get pooled from last chunk (only from second encoder for SDXL)
            if idx == len(text_encoders) - 1:
                last_output = text_encoder(chunks[-1], output_hidden_states=True)
                pooled_prompt_embeds = last_output.hidden_states[-1][:, 0, :]
        else:
            outputs = text_encoder(input_ids, output_hidden_states=True)
            encoder_prompt_embeds = outputs.hidden_states[-2]
            # Pooled output only from second encoder for SDXL
            if idx == len(text_encoders) - 1:
                pooled_prompt_embeds = outputs.hidden_states[-1][:, 0, :]

        prompt_embeds_list.append(encoder_prompt_embeds)

    # Concatenate embeddings from both encoders
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)

    # Process negative prompt
    target_length = prompt_embeds.shape[1]
    neg_prompt_embeds_list = []
    negative_pooled_prompt_embeds = None

    for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
        max_length = tokenizer.model_max_length
        neg_text = negative_prompt if negative_prompt else ""
        neg_input_ids = tokenizer(
            neg_text,
            return_tensors="pt",
            truncation=False,
        ).input_ids.to(pipeline.device)

        # Pad to match target length
        if neg_input_ids.shape[-1] < target_length * max_length:
            padding_length = target_length * max_length - neg_input_ids.shape[-1]
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            padding = torch.full((1, padding_length), pad_token_id, device=pipeline.device)
            neg_input_ids = torch.cat([neg_input_ids, padding], dim=-1)

        # Chunk and encode
        neg_chunks = []
        for i in range(0, neg_input_ids.shape[-1], max_length - 2):
            chunk = neg_input_ids[:, i:i + max_length - 2]
            chunk = torch.cat([
                neg_input_ids[:, :1],
                chunk,
                neg_input_ids[:, -1:] if i + max_length - 2 >= neg_input_ids.shape[-1] else neg_input_ids[:, :1]
            ], dim=-1)
            neg_chunks.append(chunk)

        neg_chunk_embeds = []
        for chunk in neg_chunks:
            outputs = text_encoder(chunk, output_hidden_states=True)
            neg_chunk_embeds.append(outputs.hidden_states[-2])

        encoder_neg_embeds = torch.cat(neg_chunk_embeds, dim=1)

        # Get pooled from last chunk (only from second encoder for SDXL)
        if idx == len(text_encoders) - 1:
            last_output = text_encoder(neg_chunks[-1], output_hidden_states=True)
            negative_pooled_prompt_embeds = last_output.hidden_states[-1][:, 0, :]

        neg_prompt_embeds_list.append(encoder_neg_embeds)

    negative_prompt_embeds = torch.cat(neg_prompt_embeds_list, dim=-1)

    # Ensure matching shapes
    if negative_prompt_embeds.shape[1] > target_length:
        negative_prompt_embeds = negative_prompt_embeds[:, :target_length, :]
    elif negative_prompt_embeds.shape[1] < target_length:
        padding = torch.zeros(
            (1, target_length - negative_prompt_embeds.shape[1], negative_prompt_embeds.shape[2]),
            device=pipeline.device,
            dtype=negative_prompt_embeds.dtype
        )
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def generate_image(prompt, negative_prompt="", num_inference_steps=25, guidance_scale=12):
    """
    Generate an image from a text prompt using SDXL.

    Args:
        prompt: Text description of the image to generate
        negative_prompt: Things to avoid in the generation (default: "")
        num_inference_steps: Number of denoising steps (default: 25, range: 20-50)
        guidance_scale: How closely to follow the prompt (default: 7.5, range: 5-15)

    Returns:
        Path to the saved image file
    """
    # Get the cached pipeline instance
    pipeline = get_pipeline()

    # Encode the prompt to bypass 77 token limit
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_xl(
        pipeline, prompt, negative_prompt
    )

    # Generate the image using the prompt embeddings
    image = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    print("Image generation completed!")
    # Save or display the image
    os.makedirs(output_directory, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    output_file = os.path.join(output_directory, f"generated_image-{timestamp}.png")
    image.save(output_file)
    print(f"Image saved as {output_file}")
    return output_file
