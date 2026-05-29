import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import cv2
import numpy as np

print("1. AI modellari yuklanyapti, kuting (Bu bir necha daqiqa vaqt olishi mumkin)...")

# Perspektiva va burchakni ($perspective$, $angle$) aniqlovchi ControlNet Canny modeli
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=torch.float16
)

# Rasmlarni birlashtiruvchi asosiy neyrotarmoq pipeline
pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu") # Agar GPU bo'lsa cuda, yo'qsa cpu

pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

print("2. Modellari muvaffaqiyatli yuklandi!")

# --- TEST JARAYONI ---

# 1-rasm: Eski obyekt (hovli/naves o'rni)
# Papkangizda 'hovli.jpg' degan rasm bo'lishi kerak!
object_image = Image.open("hovli.jpg").convert("RGB").resize((768, 512))

# ControlNet uchun konturlarni (liniyalarni) aniqlash
open_cv_image = np.array(object_image)
gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny_image = cv2.Canny(blur, 100, 200)
control_image = Image.fromarray(cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB))

# Prompt - AIga buyruq berish (fason rasmidagi elementlarni tasvirlash kerak)
prompt = "modern metal canopy, luxury high quality structure, professional welding, shadows, realistic architecture"

print("3. AI rasmlarni birlashtirmoqda, ozgina kuting...")

# AI jarayoni (Inference)
output = pipeline(
    prompt=prompt,
    image=object_image,             # Eski joyning o'zi
    control_image=control_image,    # Joyning burchak va chiziqlari ($proportion$)
    strength=0.6,                   # O'zgarish kuchi (60%)
    guidance_scale=7.5,
    num_inference_steps=30
).images[0]

# Natijani kompyuterga saqlash
output.save("natija_local.jpg")
print("4. Tayyor! Natija 'natija_local.jpg' nomi bilan saqlandi.")