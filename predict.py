# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import sys
sys.path.extend(['/IP-Adapter'])
import torch
import shutil
from PIL import Image
from typing import List
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel,StableDiffusionPipeline
from ip_adapter import IPAdapterPlus
from control_net_utils import CONTROLNET_MAPPING

base_model_path = "digiplay/Juggernaut_final"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/IP-Adapter/models/image_encoder/"
ip_ckpt = "/IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
device = "cuda"

MODEL_CACHE = "model-cache"
VAE_CACHE = "vae-cache"
CONTROL_CACHE = "control-cache"
POSE_CACHE = "pose-cache"
CONTROL_TYPE = "pose"

def load_image(path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MAPPING[CONTROL_TYPE]["model_id"],
            torch_dtype=torch.float16,
            cache_dir=POSE_CACHE,
        ).to(device)

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(
            vae_model_path,
            cache_dir=VAE_CACHE
        ).to(dtype=torch.float16)
        # load SD pipeline

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            cache_dir=MODEL_CACHE
        ).to(device)

#         self.pipe = StableDiffusionPipeline.from_pretrained(
#             base_model_path,
#             torch_dtype=torch.float16,
#             scheduler=noise_scheduler,
#             vae=vae,
#             feature_extractor=None,
#             safety_checker=None,
#             cache_dir=MODEL_CACHE
#         )

    def predict(
        self,
        ip_image: Path = Input(
             description="Input face image",
             default=None
        ),
        ip_guidance_scale: float = Input(
            description="ip_guidance_scale b/w 0 to 1",
            ge=0.0,
            le=1.0,
            default=0.5,
        ),
        prompt: str = Input(
            description="Prompt",
            default="photo of a beautiful girl wearing casual shirt in a garden"
        ),
        negative_prompt: str = Input(
                    description="negative_prompt",
                    default="nipple, cleavage, nudity,nsfw,sexual ,explicit, revealing ,suggestive ,provocative ,lingerie , signature, watermark, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, " \
                                                            "bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn " \
                                                            "hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, " \
                                                            "out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, , mangled, " \
                                                            "old, surreal, long face, out of frame"
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        width: int = Input(
            description="Width of the output image",
            ge=500,
            le=1025,
            default=512,
        ),
        height: int = Input(
            description="Height of the output image",
            ge=500,
            le=1025,
            default=512,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        guidance_scale: float = Input(
            description="guidance_scale",
              ge=0.0,default=7.5
        ),
        control_net_image: Path = Input(
             description="Enter image to be used with controlnet",
             default=None
        ),
        controlnet_conditioning_scale: float = Input(
                description="ip_guidance_scale b/w 0 to 1",
                ge=0.0,
                le=1.0,
                default=1.0,
            ),

    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        ip_image = Image.open(ip_image)
        ip_image.resize((256, 256))

        # Load pose image
        control_net_image = Image.open(control_net_image).resize((width, height))
        openpose_image = CONTROLNET_MAPPING[CONTROL_TYPE]["hinter"](control_net_image)

        ip_model = IPAdapterPlus(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            seed=seed,
            num_samples=num_outputs,
            num_inference_steps=num_inference_steps,

            pil_image=ip_image,
            scale=ip_guidance_scale,

            image=openpose_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        output_paths = []
        for i, _ in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            images[i].save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
