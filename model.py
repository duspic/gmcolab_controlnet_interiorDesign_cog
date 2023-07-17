from __future__ import annotations

import gc

import numpy as np
import PIL.Image
import torch
import cv2
import einops
from transformers import AutoModel
from annotator.util import resize_image, HWC3
from diffusers import (ControlNetModel, DiffusionPipeline,
                       StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler, DEISMultistepScheduler,
                       EulerAncestralDiscreteScheduler)

A_PROMPT = "best quality,masterpiece,realistic,living room,Modern minimalist Nordic style,Soft light,Pure picture,Bright colors,Symmetrical composition"
N_PROMPT = "text,word,cropped,low quality,watermark,signature,blurry,soft,soft line,curved line,sketch,ugly,logo,pixelated,lowres,"


CONTROLNET_MODEL_IDS = {
    'Openpose': 'lllyasviel/control_v11p_sd15_openpose',
    'canny': 'lllyasviel/control_v11p_sd15_canny',
    'MLSD': 'lllyasviel/control_v11p_sd15_mlsd',
    'scribble': 'lllyasviel/control_v11p_sd15_scribble',
    'softedge': 'lllyasviel/control_v11p_sd15_softedge',
    'segmentation': 'lllyasviel/control_v11p_sd15_seg',
    'depth': 'lllyasviel/control_v11f1p_sd15_depth',
    'NormalBae': 'lllyasviel/control_v11p_sd15_normalbae',
    'lineart': 'lllyasviel/control_v11p_sd15_lineart',
    'lineart_anime': 'lllyasviel/control_v11p_sd15s2_lineart_anime',
    'shuffle': 'lllyasviel/control_v11e_sd15_shuffle',
    'ip2p': 'lllyasviel/control_v11e_sd15_ip2p',
    'inpaint': 'lllyasviel/control_v11e_sd15_inpaint',
}

config_dict = {
    'ducnapa/InteriorDesignSuperMixV2': 'lllyasviel/control_v11p_sd15_canny',
    'darkstorm2150/Protogen_x5.8_Official_Release': 'lllyasviel/control_v11p_sd15_canny',
    'wavymulder/modelshoot': 'lllyasviel/control_v11p_sd15_canny',
    'stabilityai/stable-diffusion-2-1-base': 'thepowefuldeez/sd21-controlnet-canny',
    'DGSpitzer/Cyberpunk-Anime-Diffusion': 'lllyasviel/control_v11p_sd15_canny',
    # 'thibaud/controlnet-sd21'
    # 'shgao/edit-anything-v0-4-sd21'
}


class Model:
    def __init__(self,
                 base_model_id: str = 'ducnapa/InteriorDesignSuperMixV2',
                 task_name: str = 'canny'):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.base_model_id = 'ducnapa/InteriorDesignSuperMixV2'
        self.task_name = ''
        self.pipe = self.load_pipe(base_model_id, task_name)
        # self.preprocessor = Preprocessor()

    def load_pipe(self, base_model_id: str, task_name) -> DiffusionPipeline:
        if base_model_id == self.base_model_id and task_name == self.task_name and hasattr(
                self, 'pipe') and self.pipe is not None:
            return self.pipe

        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(config_dict[base_model_id])
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            safety_checker=None,
            controlnet=controlnet,
            # torch_dtype=torch.float16
            )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config)
        # if self.device.type == 'cuda':
            # pipe.enable_xformers_memory_efficient_attention()
        pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.base_model_id = base_model_id
        self.task_name = task_name
        return pipe

    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id, self.task_name)
        except Exception:
            self.pipe = self.load_pipe(self.base_model_id, self.task_name)
        return self.base_model_id

    def load_controlnet_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        if self.pipe is not None and hasattr(self.pipe, 'controlnet'):
            del self.pipe.controlnet
        torch.cuda.empty_cache()
        gc.collect()
        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(model_id,
                                                     torch_dtype=torch.float16)
        # controlnet = AutoModel.from_pretrained(model_id)
        controlnet.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet
        self.task_name = task_name

    def get_prompt(self, prompt: str, additional_prompt: str) -> str:
        if not prompt:
            prompt = additional_prompt
        else:
            prompt = f'{prompt}, {additional_prompt}'
        return prompt

    @torch.autocast('cuda')
    def run_pipe(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int64).max)
        generator = torch.Generator().manual_seed(seed)
        return self.pipe(prompt=prompt,
                         negative_prompt=negative_prompt,
                         guidance_scale=guidance_scale,
                         num_images_per_prompt=num_images,
                         num_inference_steps=num_steps,
                         generator=generator,
                         image=control_image).images

    @torch.inference_mode()
    def process_canny(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str = A_PROMPT,
        negative_prompt: str = N_PROMPT,
        num_images: int = 1,
        image_resolution: int = 512,
        num_steps: int = 30,
        guidance_scale: float = 7.9,
        seed: int = -1,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> list[PIL.Image.Image]:
        img = resize_image(HWC3(image), image_resolution)
        H, W, C = img.shape

        detected_map = cv2.Canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_images)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        self.load_controlnet_weight('canny')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return results
