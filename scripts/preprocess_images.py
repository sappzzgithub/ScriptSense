# scripts/preprocess_images.py
from PIL import Image
import os

traits = ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]
base_input_dir = "/Users/sakshizanjad/Desktop/ScriptSense--1/Dataset/train"
base_output_dir = "/Users/sakshizanjad/Desktop/ScriptSense--1/Dataset/processed_samples"

for trait in traits:
    source_dir = os.path.join(base_input_dir, trait)
    dest_dir = os.path.join(base_output_dir, trait)
    os.makedirs(dest_dir, exist_ok=True)

    for img_file in os.listdir(source_dir):
        if img_file.endswith((".jpg", ".jpeg", ".png")):  # safe check
            img_path = os.path.join(source_dir, img_file)
            img = Image.open(img_path).convert("L")  # grayscale
            img = img.resize((128, 128)) #resize
            img.save(os.path.join(dest_dir, img_file)) #saves the image in processed samples folder

    print(f"✅ Processed {trait} samples!")