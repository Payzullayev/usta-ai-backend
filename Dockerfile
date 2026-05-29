FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Kerakli tizim paketlarini o'rnatish (OpenCV ishlashi uchun)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Loyiha fayllarini nusxalash
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model yuklanishini tekshirish va serverni tayyorlash
EXPOSE 8000
CMD ["python", "-u", "app.py"]