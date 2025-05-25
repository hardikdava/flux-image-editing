from typing import List
import time
import os
import subprocess
import random

from cog import BasePredictor, Input, Path
import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import hf_hub_download
load_dotenv()


MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
EDIT_MODEL = "sayakpaul/FLUX.1-dev-edit-v0"
MAX_IMAGE_SIZE = 1440


def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")
        edit_transformer = FluxTransformer2DModel.from_pretrained(EDIT_MODEL, torch_dtype=torch.bfloat16)
        self.pipeline = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=edit_transformer, torch_dtype=torch.bfloat16
        ).to("cuda")

    def load_hyper_lora(self):
        self.pipeline.load_lora_weights(
            hf_hub_download(
                "ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"
            ),
            adapter_name="hyper-sd",
        )
        self.pipeline.set_adapters(["hyper-sd"], adapter_weights=[0.125])

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        image: Path = Input(
                description="The image for the generation",
                default=None
        ),
        guidance_scale: float = Input(
                description="Guidance scale",
                default=30,
                ge=0,
                le=50
        ),
        num_inference_steps: int = Input(
                description="Number of inference steps",
                ge=1, le=80, default=50,
            ),
        seed: int = Input(
                description="Random seed. Set for reproducible generation", default=None
        ),
        num_outputs: int = Input(
                description="Number of outputs to generate", default=1, le=4, ge=1
        ),
        use_hyper_lora: bool = Input(
                description="Use Hyper Lora for faster generation. This speeds up the process.",
                default=False,
        ),
        output_format: str = Input(
                description="Format of the output images",
                choices=["webp", "jpg", "png"],
                default="jpg",
        ),
        output_quality: int = Input(
                description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
                default=100,
                ge=0,
                le=100,
        ),
    ) -> List[Path]:
        start_time = time.time()
        if not seed:
            seed = random.randint(0, 100000)
            print(f"No seed provided. Generating a random seed. seed={seed}")

        image = Image.open(image)
        infer_width, infer_height = image.size
        if max(image.size) > MAX_IMAGE_SIZE:
            scale_factor = MAX_IMAGE_SIZE / max(image.size)
            infer_width = int(image.size[0] * scale_factor)
            infer_height = int(image.size[1] * scale_factor)
            print(f"Image is too large, resizing to {infer_width}x{infer_height}")
            image = image.resize((infer_width, infer_height), Image.LANCZOS)

        self.pipeline.unload_lora_weights()
        if use_hyper_lora:
            self.load_hyper_lora()

        """Run a single prediction on the model"""
        outputs = self.pipeline(
            control_image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=infer_width,
            height=infer_height,
            num_images_per_prompt=num_outputs,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        output_paths = []
        for i, image in enumerate(outputs.images):
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != "png":
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return output_paths
