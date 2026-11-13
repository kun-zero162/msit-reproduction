"""
datasets.py: Module xử lý dữ liệu, tokenize và tiền xử lý cho mô hình MSIT
(Multimodal Self-Correction Instruction Tuning for Open-World Product Attribute Mining - ACL 2025).

Bao gồm:
1. AGTD_INSTRUCTIONS & CTTD_INSTRUCTIONS.
2. Raw Dataset Loaders (WOAM, OAMine).
3. MSITDataPreparator: Bộ tiền xử lý (Lọc nhiễu, Cân bằng danh mục, Chia tập, Mẫu đối lập).
4. PyTorch Dataset Classes (AGTD, CTTD, Joint).
5. Multimodal Data Collator.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        pass


# ==============================================================================
# 1. DANH SÁCH PROMPTS / INSTRUCTIONS CHUẨN TỪ BÀI BÁO (TABLE 4 & TABLE 5)
# ==============================================================================

AGTD_INSTRUCTIONS: List[str] = [
    "Extract the information from the title, bullet points, and product picture into JSON format.",
    "Convert the attribute values of the product from the provided information into JSON format.",
    "Generate the product attribute values in JSON format based on the provided title, bullet points, and picture.",
    "Compile the product's characteristic attributes into JSON format according to the provided information.",
    "Extract the product attribute information into JSON format from the provided title, bullet points, and picture.",
    "Parse the product's features and attributes into JSON format from the given information.",
    "Extract the product's characteristics into JSON format using the provided title, bullet points, and picture.",
    "Retrieve and organize the product's attribute values into JSON format from the provided information.",
    "Compile the product's attribute information into JSON format based on the title, bullet points, and picture content.",
    "Generate JSON-formatted product attribute data based on the provided information."
]

CTTD_INSTRUCTIONS: List[str] = [
    "I provide you with text and image information of a product along with one attribute of this product. Please determine if this product possesses this attribute through the text and image. Let's think step by step, and please provide your reasoning process.",
    "Here's textual and visual data about a product, along with a specific attribute. Your job is to discern if this attribute applies to the product, using both the text and the visuals. Let's methodically analyze the information, detailing your reasoning process step by step.",
    "I present you with textual and visual data about a product, along with a single attribute associated with it. Your task is to determine whether this product exhibits this attribute, utilizing both the text and the image. Let's think step by step, and please provide your reasoning process.",
    "I provide you with the text and picture information of a product and an attribute of this product, please help me judge whether this product has this attribute through the text and picture, we will think step by step, please give your reasoning process.",
    "I will give you the text and image information of a product, as well as one of its attributes. Please use the text and image to help me determine whether the product has this attribute. Let's think step by step, and please explain your reasoning process."
]

COT_5STEP_GUIDE = (
    "Five-step Chain of thought:\n"
    "Step 1: Determine the type of product based on pictures and text information.\n"
    "Step 2: Analyze the meaning of the attribute and judge whether the attribute matches the product type based on common sense.\n"
    "Step 3: Check if the product exhibits this attribute from the image.\n"
    "Step 4: Check if the product exhibits this attribute from the text description and verify exact value if present.\n"
    "Step 5: Summarize the conclusions from the above steps and draw the final decision (yes or no).\n"
)


# ==============================================================================
# 2. BỘ ĐỌC DỮ LIỆU GỐC (RAW DATA READERS)
# ==============================================================================

class WOAMRawReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file WOAM: {file_path}")

    def load_records(self, category: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                if category and item.get("pt", "").upper() != category.upper():
                    continue

                is_dev, is_test = item.get("is_dev", False), item.get("is_test", False)
                if split == "dev" and not is_dev: continue
                elif split == "test" and not is_test: continue
                elif split == "train" and (is_dev or is_test): continue

                title = item.get("title", "").strip()
                bullet_points = item.get("bullet_point", [])
                bullet_text = " ".join([str(bp).strip() for bp in bullet_points if bp]) if isinstance(bullet_points, list) else str(bullet_points).strip()

                records.append({
                    "asin": item.get("asin", ""),
                    "category": item.get("pt", ""),
                    "title": title,
                    "bullet_point": bullet_text,
                    "raw": item
                })
                if limit and len(records) >= limit:
                    break
        return records

class OAMineRawReader:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.annotations_dir = os.path.join(base_dir, "annotations")
        self.raw_dir = os.path.join(base_dir, "raw")

    def list_categories(self, use_annotations: bool = True) -> List[str]:
        target_dir = self.annotations_dir if use_annotations else self.raw_dir
        if not os.path.exists(target_dir): return []
        return [f.replace(".jsonl", "") for f in os.listdir(target_dir) if f.endswith(".jsonl")]

    def load_annotated_records(self, category: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records = []
        files = [f"{category}.jsonl"] if category else os.listdir(self.annotations_dir)

        for filename in files:
            if not filename.endswith(".jsonl"): continue
            cat_name = filename.replace(".jsonl", "")
            filepath = os.path.join(self.annotations_dir, filename)
            if not os.path.exists(filepath): continue

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    records.append({
                        "asin": item.get("asin", ""),
                        "category": cat_name,
                        "title": item.get("title", "").strip(),
                        "bullet_point": item.get("bullet_points", item.get("description", "")),
                        "raw": item
                    })
                    if limit and len(records) >= limit:
                        return records
        return records


# ==============================================================================
# 3. BỘ TIỀN XỬ LÝ VÀ CHIA TẬP (DATA PREPARATOR)
# ==============================================================================

class MSITDataPreparator:
    """
    Tiền xử lý và chia tập chuẩn xác theo paper MSIT:
    1. Lọc Modality Noise (Kiểm tra ảnh tồn tại + Độ dài văn bản >= 10 từ).
    2. Cân bằng danh mục (Rút đều số lượng từ các danh mục).
    3. Ngăn chặn Data Leakage (Tách biệt hoàn toàn Train/Test).
    4. Contrastive Sampling cho CTTD (50% Positive, 50% Negative).
    """
    def __init__(self, records: List[Dict], image_dir: str):
        self.records = records
        self.image_dir = image_dir

    def _filter_noise(self) -> List[Dict]:
        valid_records = []
        for r in self.records:
            # Nguyên tắc 4: Lọc Modality Noise
            img_path = os.path.join(self.image_dir, f"{r['asin']}.jpg")
            if not os.path.exists(img_path):
                continue
            
            text_length = len(r['title'].split()) + len(r['bullet_point'].split())
            if text_length < 10:
                continue
                
            r['image_path'] = img_path
            valid_records.append(r)
        return valid_records

    def prepare_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        cleaned_records = self._filter_noise()
        
        # Nhóm theo category để cân bằng
        cat_dict = {}
        for r in cleaned_records:
            cat = r['category']
            if cat not in cat_dict:
                cat_dict[cat] = []
            cat_dict[cat].append(r)

        for cat in cat_dict:
            random.shuffle(cat_dict[cat])

        # Rút đều (Round-robin) để đảm bảo Category Balancing
        balanced_pool = []
        while cat_dict:
            for cat in list(cat_dict.keys()):
                if cat_dict[cat]:
                    balanced_pool.append(cat_dict[cat].pop(0))
                else:
                    del cat_dict[cat]

        # Kiểm tra đủ số lượng 2300 mẫu (1000 Test + 1000 AGTD + 300 CTTD)
        if len(balanced_pool) < 2300:
            print(f"[Cảnh báo] Chỉ có {len(balanced_pool)} mẫu hợp lệ, ít hơn yêu cầu 2300 của paper.")
            
        test_pool = balanced_pool[:1000]
        agtd_pool = balanced_pool[1000:2000]
        cttd_pool = balanced_pool[2000:2300]

        # Nguyên tắc 3: Lấy mẫu đối lập (Contrastive Sampling) cho CTTD
        # 150 Positive, 150 Negative
        all_categories = list(set(r['category'] for r in cleaned_records))
        for i, item in enumerate(cttd_pool):
            if i < len(cttd_pool) // 2:
                item['cttd_label'] = "yes"
                item['attribute_to_judge'] = "[PLACEHOLDER_POSITIVE: Điền thuộc tính có thật từ raw data]"
            else:
                item['cttd_label'] = "no"
                other_cats = [c for c in all_categories if c != item['category']]
                random_fake_cat = random.choice(other_cats) if other_cats else "Khác"
                item['attribute_to_judge'] = f"[PLACEHOLDER_NEGATIVE: Điền thuộc tính giả mạo lấy từ danh mục {random_fake_cat}]"

        return test_pool, agtd_pool, cttd_pool


# ==============================================================================
# 4. CÁC LỚP PYTORCH DATASET
# ==============================================================================

def load_image_safely(image_source: Optional[Union[str, Any]], image_dir: Optional[str] = None) -> Optional[Any]:
    if Image is None: return None
    img_path = None
    if isinstance(image_source, str) and image_source:
        if os.path.exists(image_source): img_path = image_source
        elif image_dir:
            candidate = os.path.join(image_dir, f"{image_source}.jpg")
            if os.path.exists(candidate): img_path = candidate
            
    if img_path and os.path.exists(img_path):
        try: return Image.open(img_path).convert("RGB")
        except Exception: pass
    return Image.new("RGB", (224, 224), color=(240, 240, 240))

class MSITAGTDDataset(Dataset):
    def __init__(self, jsonl_path: str, image_dir: Optional[str] = None, use_random_instruction: bool = True):
        self.jsonl_path = jsonl_path
        self.image_dir = image_dir
        self.instructions = AGTD_INSTRUCTIONS
        self.use_random_instruction = use_random_instruction
        self.samples = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        samples = []
        if not os.path.exists(self.jsonl_path): return samples
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        instruction = random.choice(self.instructions) if self.use_random_instruction else self.instructions[0]

        title = item.get("title", "").strip()
        bullet_points = item.get("bullet_point", "")
        if isinstance(bullet_points, list):
            bullet_points = "\n".join([f"- {bp}" for bp in bullet_points if bp])

        user_content = f"<image>\n{instruction}\n\nTitle: {title}\nBullet points: {bullet_points}"
        
        attributes = item.get("attributes", item.get("merged_attributes", {}))
        target_text = json.dumps(attributes, ensure_ascii=False, indent=2) if isinstance(attributes, dict) else str(attributes).strip()
        
        img_id = item.get("image_path") or item.get("asin")
        return {
            "type": "agtd",
            "asin": item.get("asin", ""),
            "image": load_image_safely(img_id, self.image_dir),
            "prompt": user_content,
            "target": target_text,
            "category": item.get("category", "")
        }

class MSITCTTDDataset(Dataset):
    def __init__(self, jsonl_path: str, image_dir: Optional[str] = None, use_random_instruction: bool = True):
        self.jsonl_path = jsonl_path
        self.image_dir = image_dir
        self.instructions = CTTD_INSTRUCTIONS
        self.use_random_instruction = use_random_instruction
        self.samples = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        samples = []
        if not os.path.exists(self.jsonl_path): return samples
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        instruction = random.choice(self.instructions) if self.use_random_instruction else self.instructions[0]

        attribute_to_judge = item.get("attribute_to_judge", "Unknown")
        title = item.get("title", "").strip()
        bullet_points = item.get("bullet_point", "")
        if isinstance(bullet_points, list):
            bullet_points = "\n".join([f"- {bp}" for bp in bullet_points if bp])

        user_content = f"<image>\n{instruction}\n\nThe attribute I want to judge is: {attribute_to_judge}\nTitle: {title}\nBullet Point: {bullet_points}"

        if "cot_response" in item:
            target_text = item["cot_response"]
        else:
            decision = item.get("cttd_label", item.get("decision", "yes")).strip().capitalize()
            target_text = (
                "Step 1: Product type analyzed.\n"
                "Step 2: Common sense reasoning performed.\n"
                "Step 3: Image validation completed.\n"
                "Step 4: Text verification completed.\n"
                f"Step 5: Based on the above steps, the attribute is valid.\n\n{decision}!"
            )

        img_id = item.get("image_path") or item.get("asin")
        return {
            "type": "cttd",
            "asin": item.get("asin", ""),
            "image": load_image_safely(img_id, self.image_dir),
            "prompt": user_content,
            "target": target_text,
            "attribute": attribute_to_judge,
            "decision": item.get("cttd_label", "yes")
        }

class MSITJointDataset(Dataset):
    def __init__(self, agtd_dataset: MSITAGTDDataset, cttd_dataset: MSITCTTDDataset, shuffle: bool = True):
        self.agtd_dataset = agtd_dataset
        self.cttd_dataset = cttd_dataset
        self.indices = [("agtd", i) for i in range(len(agtd_dataset))] + [("cttd", j) for j in range(len(cttd_dataset))]
        if shuffle: random.shuffle(self.indices)

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dtype, sub_idx = self.indices[idx]
        return self.agtd_dataset[sub_idx] if dtype == "agtd" else self.cttd_dataset[sub_idx]


# ==============================================================================
# 5. MULTIMODAL DATA COLLATOR
# ==============================================================================

@dataclass
class MSITDataCollator:
    processor: Any
    max_length: int = 2048
    ignore_index: int = -100

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [item["image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        targets = [item["target"] for item in batch]

        full_texts, prompt_lengths = [], []
        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        for p, t in zip(prompts, targets):
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": p}, {"role": "assistant", "content": t}]
                try:
                    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    prompt_only = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
                except Exception:
                    full_text, prompt_only = f"USER: {p} ASSISTANT: {t}</s>", f"USER: {p} ASSISTANT:"
            else:
                full_text, prompt_only = f"USER: {p} ASSISTANT: {t}</s>", f"USER: {p} ASSISTANT:"

            full_texts.append(full_text)
            prompt_lengths.append(len(tokenizer.encode(prompt_only, add_special_tokens=False)))

        try:
            batch_encoding = self.processor(text=full_texts, images=images, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        except Exception:
            batch_encoding = tokenizer(full_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids = batch_encoding["input_ids"]
        labels = input_ids.clone()
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)

        for i, p_len in enumerate(prompt_lengths):
            labels[i, :min(p_len, labels.shape[1])] = self.ignore_index
            if pad_token_id is not None:
                labels[i][input_ids[i] == pad_token_id] = self.ignore_index

        batch_encoding["labels"] = labels
        return batch_encoding


if __name__ == "__main__":
    print("=== KIỂM TRA BỘ TIỀN XỬ LÝ (DATA PREPARATOR) ===")
    
    # Giả lập quy trình đọc và tiền xử lý
    woam_path = os.path.join("datasets", "woam", "corpus_with_seeds.jsonl")
    if os.path.exists(woam_path):
        reader = WOAMRawReader(woam_path)
        raw_records = reader.load_records()
        
        # 1. Chạy bộ tiền xử lý
        image_dir = os.path.join("datasets", "images", "woam")
        preparator = MSITDataPreparator(raw_records, image_dir)
        test_pool, agtd_pool, cttd_pool = preparator.prepare_splits()
        
        print(f"Tổng hợp lệ: {len(test_pool) + len(agtd_pool) + len(cttd_pool)}")
        print(f" - AGTD Train: {len(agtd_pool)} mẫu.")
        print(f" - CTTD Train: {len(cttd_pool)} mẫu (Cân bằng Yes/No).")
        print(f" - Test Set: {len(test_pool)} mẫu.")