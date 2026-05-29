import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import io
import base64
import cv2
import numpy as np

# Server yonganda modelni yuklash
def init():
    global pipeline
    
    # Perspektiva saqlovchi ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16
    )
    
    # Asosiy birlashtiruvchi model
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_xformers_memory_efficient_attention()

def decode_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# API so'rov kelganda ishlaydi
def handler(model_inputs: dict) -> dict:
    global pipeline
    
    object_image_b64 = model_inputs.get("object_image")  # 1-rasm (eski hovli)
    fason_image_b64 = model_inputs.get("fason_image")    # 2-rasm (yangi fason)
    prompt = model_inputs.get("prompt", "modern architectural design, high quality, realistic metal structure")
    
    if not object_image_b64 or not fason_image_b64:
        return {"error": "Ikkala rasm ham kerak!"}
        
    object_image = decode_image(object_image_b64).resize((768, 512))
    
    # Konturlarni aniqlash (Perspektiva uchun)
    open_cv_image = np.array(object_image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 100, 200)
    control_image = Image.fromarray(cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB))

    # AI rasm chizish jarayoni
    output = pipeline(
        prompt=prompt,
        image=object_image,
        control_image=control_image,
        strength=0.6,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]
    
    return {"result_image": encode_image(output)}