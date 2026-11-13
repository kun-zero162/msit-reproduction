"""
data_generation.py: Module sinh dữ liệu tinh chỉnh AGTD và CTTD sử dụng Qwen qua OpenRouter API.
(Dựa trên quy trình In-context Learning của khung MSIT - ACL 2025).
"""

import os
import json
import time
import base64
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional
from PIL import Image

# Thiết lập API Key từ biến môi trường
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
MODEL_NAME = "qwen/qwen3.8-27b:free"

class OpenRouterDataGenerator:
    def __init__(self, model_name: str = MODEL_NAME, delay_between_requests: float = 6.0):
        """
        delay_between_requests: Thời gian nghỉ (giây) để tránh bị lỗi Rate Limit của gói Free.
        """
        self.model_name = model_name
        self.delay = delay_between_requests
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/MSIT-Project", # Khuyến nghị từ OpenRouter
            "X-Title": "MSIT Data Generator" # Khuyến nghị từ OpenRouter
        }

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Chuyển đổi ảnh PIL sang chuỗi Base64 để gửi qua API OpenRouter."""
        buffered = BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _clean_json_output(self, text: str) -> str:
        """Làm sạch block markdown (```json ... ```) thường xuất hiện trong output của Qwen."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
            
        return text.strip()

    def _safe_generate(self, prompt: str, image: Optional[Image.Image] = None, is_json: bool = False) -> str:
        """Hàm gọi HTTP Request trực tiếp, hỗ trợ tham số 'reasoning' của OpenRouter."""
        content = [{"type": "text", "text": prompt}]
        
        if image is not None:
            base64_image = self._encode_image_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "reasoning": {"enabled": True} # Bật chế độ suy luận theo tài liệu của OpenRouter
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(self.delay)
                response = requests.post(
                    url=self.api_url,
                    headers=self.headers,
                    data=json.dumps(payload)
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    # Trích xuất nội dung từ message
                    result = response_data['choices'][0]['message'].get('content', '')
                    
                    if not result:
                        # Trong một số trường hợp, output có thể nằm trong 'reasoning_details' nếu model chỉ suy luận
                        result = response_data['choices'][0]['message'].get('reasoning_details', '')
                        
                    if is_json:
                        result = self._clean_json_output(result)
                        
                    return result.strip()
                else:
                    print(f"[Lỗi HTTP {response.status_code}] {response.text}")
                    time.sleep(self.delay * (attempt + 2))
                
            except Exception as e:
                print(f"[Lỗi API] Thử lại lần {attempt + 1}/{max_retries} - Lỗi: {e}")
                time.sleep(self.delay * (attempt + 2))
        
        return ""

    def generate_agtd_sample(self, title: str, bullet_point: str, image_path: str) -> Dict[str, Any]:
        print(f"Đang xử lý AGTD cho sản phẩm: {title[:50]}...")
        
        # 1. Prompt trích xuất từ chữ (Text)
        text_prompt = (
            "You are an expert in e-commerce. Extract product attributes and their values from the following text.\n"
            f"Title: {title}\nBullet points: {bullet_point}\n\n"
            "Output strictly in JSON format where keys are attribute names and values are attribute values. "
            "Ensure all JSON keys are in lowercase snake_case format (e.g., compatible_phone_model)."
        )
        text_attributes = self._safe_generate(text_prompt, is_json=True)
        
        # 2. Prompt trích xuất từ ảnh (Image)
        image = None
        image_attributes = "{}"
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                img_prompt = (
                    "Extract visible product attributes (e.g., color, shape, packaging type) directly from this image.\n"
                    "Output strictly in JSON format where keys are attribute names and values are attribute values. "
                    "Ensure all JSON keys are in lowercase snake_case format."
                )
                image_attributes = self._safe_generate(img_prompt, image=image, is_json=True)
            except Exception as e:
                print(f"Lỗi đọc ảnh {image_path}: {e}")

        # 3. Hợp nhất (Merge)
        merge_prompt = (
            "Merge the following two JSON objects containing product attributes. "
            "Remove duplicates and keep the most descriptive values.\n"
            f"Text attributes: {text_attributes}\n"
            f"Image attributes: {image_attributes}\n\n"
            "Output the merged result strictly in JSON format. Do not include any other text."
        )
        merged_attributes = self._safe_generate(merge_prompt, is_json=True)
        
        try:
            parsed_attributes = json.loads(merged_attributes)
        except:
            parsed_attributes = {}

        return {
            "title": title,
            "bullet_point": bullet_point,
            "image_path": image_path,
            "attributes": parsed_attributes
        }

    def generate_cttd_sample(self, title: str, bullet_point: str, image_path: str, target_attribute: str) -> Dict[str, Any]:
        print(f"Đang xử lý CTTD cho thuộc tính '{target_attribute}'...")
        
        prompt = (
            f"Evaluate if the product has the attribute: '{target_attribute}'.\n"
            f"Title: {title}\nBullet points: {bullet_point}\n\n"
            "Provide your reasoning strictly in the following 5 steps:\n"
            "Step 1: Determine the type of product based on pictures and text information.\n"
            "Step 2: Analyze the meaning of the attribute and judge whether it matches the product type based on common sense.\n"
            "Step 3: Check if the product exhibits this attribute from the image.\n"
            "Step 4: Check if the product exhibits this attribute from the text description and verify exact value if present.\n"
            "Step 5: Summarize the conclusions and draw the final decision (yes or no).\n"
        )
        
        image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None
        cot_response = self._safe_generate(prompt, image=image)
        
        if not cot_response:
            decision = "no"
        else:
            decision = "yes" if "yes" in cot_response.lower().split("\n")[-1] else "no"

        return {
            "title": title,
            "bullet_point": bullet_point,
            "image_path": image_path,
            "target_attribute": target_attribute,
            "cot_response": cot_response,
            "decision": decision
        }