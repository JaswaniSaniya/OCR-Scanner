import os
import cv2
import base64
import easyocr
import requests
import matplotlib.pyplot as plt
from PIL import Image
from mistralai import Mistral
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# -------------------- Setup -------------------- #

# Load EasyOCR model (runs once)
easy_reader = easyocr.Reader(['en'])

# Load Qwen2-VL model and processor
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
qwen_processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")


# -------------------- Utilities -------------------- #

def display_image(path):
    """Display image using matplotlib."""
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


def extract_easyocr_text(image_path):
    """Extract and print text using EasyOCR."""
    print(f"\n[EasyOCR] Reading: {image_path}")
    result = easy_reader.readtext(image_path)
    print('------ Detected Text ------')
    for _, text, _ in result:
        print(text)


def extract_qwen_text(image_path):
    """Extract and print text using Qwen2-VL OCR model."""
    print(f"\n[Qwen2-VL] Reading: {image_path}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Extract text from this image."}
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cpu")
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print('------ Detected Text ------')
    for line in output_text[0].split('\n'):
        print(line)


def encode_image_to_base64(image_path):
    """Convert image to base64 for API call."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"[Error encoding image] {e}")
        return None


def extract_mistral_text(image_path):
    """Extract and print text using Mistral OCR API."""
    print(f"\n[Mistral OCR] Reading: {image_path}")
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return
    try:
        api_key = os.environ["MISTRAL_API_KEY"]
        client = Mistral(api_key=api_key)

        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
            include_image_base64=True
        )

        print('------ Detected Text ------')
        print(response["text"])  # Adjust depending on exact API response

    except KeyError:
        print("Error: MISTRAL_API_KEY environment variable not set.")
    except Exception as e:
        print(f"[Mistral OCR Error] {e}")


# -------------------- Run All OCRs -------------------- #

def run_ocr_demo(image_path):
    display_image(image_path)
    extract_easyocr_text(image_path)
    extract_qwen_text(image_path)
    extract_mistral_text(image_path)


if __name__ == "__main__":
    # Modify these paths as needed
    images_to_test = [
        "/content/zones Info.jpg",
        "/content/ocr_long_text2.jpg",
        "/content/OCR_text1.png"
    ]

    for img_path in images_to_test:
        run_ocr_demo(img_path)
