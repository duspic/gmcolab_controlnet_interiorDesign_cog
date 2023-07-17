# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from model import Model
from annotator.util import resize_image, HWC3
import cv2
import os
from PIL import Image
from typing import List
import numpy as np

MODEL = "ducnapa/InteriorDesignSuperMixV2"
A_PROMPT = "best quality,masterpiece,realistic,living room,Modern minimalist Nordic style,Soft light,Pure picture,Bright colors,Symmetrical composition"
N_PROMPT = "text,word,cropped,low quality,watermark,signature,blurry,soft,soft line,curved line,sketch,ugly,logo,pixelated,lowres,"

class Predictor(BasePredictor):
    def setup(self):
       self.model = Model(base_model_id=MODEL, task_name='canny')

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="choose a color scheme",
                            choices=["RED", "ORANGE", "GREEN", "BEIGE"],
                            default="RED"
                        ),
        #num_samples: str = '1',
        #image_resolution: str = '512',
        #low_threshold: int = 100,
        #high_threshold: int = 200,
        #ddim_steps: int = 30,
        #scale: float = 7,
        #seed: int = -1,
        #eta: float = 0.0,
        #a_prompt: str ="best quality, extremely detailed",
        #n_prompt: str = N_PROMPT,
        #detect_resolution: int = 512,
        # bg_threshold: float = Input(description="Background Threshold (only applicable when model type is 'normal')", default=0.0, ge=0.0, le=1.0), # only applicable when model type is 'normal'
        # value_threshold: float = Input(description="Value Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=2.0), # only applicable when model type is 'MLSD'
        # distance_threshold: float = Input(description="Distance Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=20.0), # only applicable when model type is 'MLSD'
    ) -> List[Path]:
        """Run a single prediction on the model"""

        num_samples = '1'
        image_resolution = '512'
        low_threshold: int = 100,
        high_threshold: int = 200,
        ddim_steps: int = 30,
        scale: float = 7,
        seed: int = -1,
        a_prompt: str = A_PROMPT,
        n_prompt: str = N_PROMPT,
        input_image = Image.open(image)
        input_image = np.array(input_image)        

        outputs = self.model.process_canny(
            input_image,
            prompt,
            a_prompt,
            n_prompt,
            int(num_samples),
            image_resolution,
            ddim_steps,
            scale,
            seed,
            low_threshold,
            high_threshold
        )


        # outputs = [Image.fromarray(output) for output in outputs]

        all_files = os.listdir("tmp/")
        existing_images = [filename for filename in all_files if filename.startswith("output_") and filename.endswith(".png")]
        num_existing_images = len(existing_images)

        outputs = [output.save(f"tmp/output_{num_existing_images+i}.png") for i, output in enumerate(outputs)]
        return [Path(f"tmp/output_{num_existing_images+i}.png") for i in range(len(outputs))]
