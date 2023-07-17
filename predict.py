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
        dominant_color: str = Input(description="choose a color scheme",
                            choices=["RED", "ORANGE", "GREEN", "BEIGE"],
                            default="RED"
                        ),
    
    ) -> List[Path]:
        """Run a single prediction on the model"""

        input_image = Image.open(image)
        input_image = np.array(input_image)        

        prompt = dominant_color
        outputs = self.model.process_canny(
            input_image,
            prompt,
            #a_prompt,
            #n_prompt,
            #int(num_samples),
            #image_resolution,
            #ddim_steps,
            #scale,
            #seed,
            #low_threshold,
            #high_threshold
        )


        # outputs = [Image.fromarray(output) for output in outputs]

        all_files = os.listdir("tmp/")
        existing_images = [filename for filename in all_files if filename.startswith("output_") and filename.endswith(".png")]
        num_existing_images = len(existing_images)

        outputs = [output.save(f"tmp/output_{num_existing_images+i}.png") for i, output in enumerate(outputs)]
        return [Path(f"tmp/output_{num_existing_images+i}.png") for i in range(len(outputs))]
