import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import io
import base64
import cv2
import numpy as np

# 1. Server yonganda modelni xotiraga yuklash (Bir marta ishlaydi)
def init():
    global pipeline
    
    # Perspektiva va chiziqlarni ($perspective$, $angle$) saqlash uchun ControlNet Canny modeli
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16
    )
    
    # Img2Img + ControlNet pipeline (Eski joy rasmi ustiga yangi narsani chizish uchun)
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # Server xotirasini tejash uchun
    pipeline.enable_xformers_memory_efficient_attention()

def decode_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 2. API orqali so'rov kelganda ishlaydigan funksiya
def handler(model_inputs: dict) -> dict:
    global pipeline
    
    # Foydalanuvchi bot yoki ilovadan yuborgan ma'lumotlar
    object_image_b64 = model_inputs.get("object_image")  # Eski hovli/joy rasmi
    fason_image_b64 = model_inputs.get("fason_image")    # Yangi naves/darvoza rasmi
    prompt = model_inputs.get("prompt", "modern architectural design, high quality, realistic structure")
    
    if not object_image_b64 or not fason_image_b64:
        return {"error": "Ikkala rasm ham yuborilishi shart!"}
        
    # Rasmlarni neyrotarmoq formatiga keltirish
    object_image = decode_image(object_image_b64).resize((768, 512))
    
    # ControlNet uchun eski rasm burchaklarini (Canny Edge) aniqlash
    open_cv_image = np.array(object_image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 100, 200)
    control_image = Image.fromarray(cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB))

    # AI orqali rasmlarni birlashtirish (Inference)
    output = pipeline(
        prompt=prompt,
        image=object_image,              # Qayerga joylanadi
        control_image=control_image,    # Geometriya va proporsiya ($proportion$)
        strength=0.6,                    # O'zgarish darajasi (60%)
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]
    
    # Natijani qaytarish
    result_b64 = encode_image(output)
    return {"result_image": result_b64}