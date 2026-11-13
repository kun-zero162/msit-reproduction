"""
build_sft_data.py: Tích hợp MSITDataPreparator và Qwen qua OpenRouter API để sinh nhãn SFT.
Lưu dữ liệu thành 3 tập: agtd_train.jsonl, cttd_train.jsonl, và test_set.jsonl.
"""

import os
import json
from datasets import WOAMRawReader, MSITDataPreparator
from data_generation import OpenRouterDataGenerator

def save_test_set(test_pool: list, output_file: str):
    """Lưu trữ riêng 1.000 mẫu Test Set để dùng cho pha Evaluation sau này."""
    with open(output_file, "w", encoding="utf-8") as f:
        for item in test_pool:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Đã lưu {len(test_pool)} mẫu vào {output_file} (Zero Data Leakage).")

def generate_and_save_sft_labels(pool: list, output_file: str, mode: str, generator: OpenRouterDataGenerator):
    """
    Duyệt qua danh sách dữ liệu, gọi Gemini API sinh nhãn và ghi nối (append) vào file.
    Hỗ trợ tự động bỏ qua các mẫu đã sinh nếu script bị gián đoạn.
    """
    # Tính năng Resume: Kiểm tra số lượng mẫu đã sinh trong file đích
    processed_asins = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_asins.add(json.loads(line).get("asin", ""))
    
    print(f"\nBắt đầu sinh nhãn {mode.upper()}... (Đã có sẵn {len(processed_asins)}/{len(pool)} mẫu)")

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, item in enumerate(pool):
            asin = item.get("asin", "")
            if asin in processed_asins:
                continue
            
            title = item.get("title", "")
            bp = item.get("bullet_point", "")
            img_path = item.get("image_path", "")

            print(f"[{idx+1}/{len(pool)}] Đang gọi API cho ASIN: {asin}")
            
            try:
                if mode == "agtd":
                    # Sinh JSON thuộc tính
                    api_result = generator.generate_agtd_sample(title, bp, img_path)
                    item["attributes"] = api_result.get("attributes", {})
                
                elif mode == "cttd":
                    # Sinh chuỗi suy luận 5 bước
                    attr = item.get("attribute_to_judge", "")
                    api_result = generator.generate_cttd_sample(title, bp, img_path, attr)
                    item["cot_response"] = api_result.get("cot_response", "")
                    # Ghi đè decision thực tế từ API để đối chiếu với cttd_label (yes/no) đã gán
                    item["decision"] = api_result.get("decision", item.get("cttd_label"))

                # Ghi ngay vào file để tránh mất dữ liệu nếu API lỗi giữa chừng
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"[Lỗi] Bỏ qua ASIN {asin} do lỗi API: {e}")

if __name__ == "__main__":
    # 1. Đọc dữ liệu Raw từ WOAM (hoặc OAMine tùy bạn chỉ định)
    raw_file = os.path.join("datasets", "woam", "corpus_with_seeds.jsonl")
    image_dir = os.path.join("datasets", "images", "woam")
    
    reader = WOAMRawReader(raw_file)
    raw_records = reader.load_records()
    
    # 2. Chạy bộ tiền xử lý để lọc nhiễu và chia tập
    print("Đang tiền xử lý và chia tập dữ liệu...")
    preparator = MSITDataPreparator(raw_records, image_dir)
    test_pool, agtd_pool, cttd_pool = preparator.prepare_splits()
    
    # 3. Khởi tạo Qwen qua OpenRouter API
    generator = OpenRouterDataGenerator()
    
    # 4. Lưu Test Set để đảm bảo không bị lẫn lộn vào tập huấn luyện
    save_test_set(test_pool, "test_set.jsonl")
    
    # 5. Gọi API sinh nhãn SFT cho 1.300 mẫu
    generate_and_save_sft_labels(agtd_pool[:3], "agtd_train.jsonl", mode="agtd", generator=generator)
    generate_and_save_sft_labels(cttd_pool[:3], "cttd_train.jsonl", mode="cttd", generator=generator)
    
    print("\nQuá trình sinh nhãn SFT đã hoàn tất. Các tệp .jsonl đã sẵn sàng để nạp vào mô hình.")