#!/usr/bin/env python
import os
import sys
import torch
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel,StableDiffusionPipeline


# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import base_model_path, vae_model_path
from predict import CONTROL_NAME, OPENPOSE_NAME, CONTROL_CACHE, POSE_CACHE, MODEL_CACHE, VAE_CACHE
from predict import  CONTROL_LOCAL, POSE_LOCAL, MODEL_LOCAL, VAE_LOCAL

# Make cache folders
if not os.path.exists(CONTROL_CACHE):
    os.makedirs(CONTROL_CACHE)
#
# if not os.path.exists(POSE_CACHE):
#     os.makedirs(POSE_CACHE)
#
# if not os.path.exists(MODEL_CACHE):
#     os.makedirs(MODEL_CACHE)
#
# if not os.path.exists(VAE_CACHE):
#     os.makedirs(VAE_CACHE)

openpose = OpenposeDetector.from_pretrained(
    CONTROL_NAME,
    cache_dir=CONTROL_CACHE,
)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

controlnet = ControlNetModel.from_pretrained(
    OPENPOSE_NAME,
    torch_dtype=torch.float16
)
# controlnet.save_pretrained(POSE_LOCAL)


vae = AutoencoderKL.from_pretrained(
    vae_model_path
)

vae.save_pretrained(VAE_LOCAL)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None,
)
pipe.save_pretrained(MODEL_LOCAL, safe_serialization=True)