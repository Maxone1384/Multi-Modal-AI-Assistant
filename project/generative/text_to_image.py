from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")  # اگه GPU داری

def text_to_image(prompt, save_path="generated.png"):
    image = pipe(prompt).images[0]
    image.save(save_path)
    return save_path
