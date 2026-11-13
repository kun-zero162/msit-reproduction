"""
image_collection.py: Module tự động thu thập hình ảnh sản phẩm từ Amazon dựa trên mã ASIN.
Áp dụng cơ chế cân bằng danh mục (Category Balancing) để phục vụ cho các bài test Domain Adaptation.
"""

import os
import time
import random
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Nhập các class đọc dữ liệu từ module datasets.py
from datasets import WOAMRawReader, OAMineRawReader

BASE_OUTPUT_DIR = "datasets/images"

def setup_driver():
    """Thiết lập Selenium WebDriver với các cấu hình anti-bot cơ bản."""
    chrome_options = Options()
    
    # Bật dòng dưới đây nếu muốn chạy ngầm không hiện giao diện UI
    # chrome_options.add_argument("--headless") 
    
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--disable-extensions")
    
    # Giả mạo User-Agent của người dùng thực
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    chrome_options.add_argument(f"user-agent={user_agent}")
    
    # Ẩn cờ Automation của WebDriver
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Can thiệp vào DOM để xóa thuộc tính webdriver
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def download_image(img_url, save_dir, asin):
    """Tải hình ảnh thông qua thư viện requests và lưu cục bộ."""
    try:
        # Cung cấp User-Agent cho requests để tránh lỗi 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(img_url, headers=headers, stream=True, timeout=10)
        
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{asin}.jpg")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"[Thành công] Đã lưu: {file_path}")
            return True
        else:
            print(f"[Lỗi mạng] Không thể tải ảnh ASIN {asin}. Mã lỗi: {response.status_code}")
            return False
    except Exception as e:
        print(f"[Ngoại lệ] Lỗi tải ảnh ASIN {asin}: {str(e)}")
        return False

def extract_asins_by_category():
    """Trích xuất và gom nhóm ASIN theo cấu trúc: Dataset -> Category -> List[ASIN]"""
    dataset_dict = {
        "woam": {},
        "oamine": {}
    }
    
    # 1. Đọc WOAM (Yêu cầu gom vào 4 category)
    woam_path = os.path.join("datasets", "woam", "corpus_with_seeds.jsonl")
    if os.path.exists(woam_path):
        woam_reader = WOAMRawReader(woam_path)
        woam_records = woam_reader.load_records()
        for r in woam_records:
            cat = r.get("category", "UNKNOWN").upper()
            asin = r.get("asin")
            if asin and cat:
                if cat not in dataset_dict["woam"]:
                    dataset_dict["woam"][cat] = set()
                dataset_dict["woam"][cat].add(asin)

    # 2. Đọc OAMine từ tập RAW (Thay vì Annotations)
    oamine_dir = os.path.join("datasets", "oamine")
    if os.path.exists(oamine_dir):
        oam_reader = OAMineRawReader(oamine_dir)
        
        # SỬA LỖI CHÍNH TẠI ĐÂY: Quét toàn bộ thư mục raw để lấy ASIN
        if hasattr(oam_reader, 'load_all_raw_records'):
            oamine_records = oam_reader.load_all_raw_records()
        else:
            # Fallback nếu OAMineRawReader chưa được cập nhật trong scope này
            oamine_records = oam_reader.load_annotated_records() 
            print("[CẢNH BÁO] Chưa tìm thấy hàm load_all_raw_records. Đang cào theo Annotations.")

        for r in oamine_records:
            cat = r.get("category", "unknown").lower()
            asin = r.get("asin")
            if asin and cat:
                if cat not in dataset_dict["oamine"]:
                    dataset_dict["oamine"][cat] = set()
                dataset_dict["oamine"][cat].add(asin)

    # Chuyển đổi Set thành List để phục vụ cho việc xáo trộn và duyệt vòng lặp
    for ds_name in dataset_dict:
        for cat_name in dataset_dict[ds_name]:
            dataset_dict[ds_name][cat_name] = list(dataset_dict[ds_name][cat_name])

    return dataset_dict

def scrape_amazon_images(dataset_dict, base_dir=BASE_OUTPUT_DIR):
    """
    Cào ảnh ngẫu nhiên với cơ chế cân bằng danh mục.
    - WOAM: 800 mẫu mỗi danh mục.
    - OAMine: 300 mẫu mỗi danh mục.
    """
    driver = setup_driver()
    wait = WebDriverWait(driver, 10)

    # Cấu hình giới hạn (Quota) cho từng danh mục của mỗi dataset
    category_quotas = {
        "woam": 800,
        "oamine": 300
    }

    try:
        for dataset_name, categories in dataset_dict.items():
            if not categories:
                continue
            
            # Số lượng cần đạt cho 1 danh mục thuộc dataset này
            target_count = category_quotas.get(dataset_name, 300)
            
            # Tạo thư mục riêng cho dataset
            dataset_dir = os.path.join(base_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            for category_name, asin_list in categories.items():
                print(f"\n{'='*50}")
                print(f"DATASET: {dataset_name.upper()} | CATEGORY: {category_name.upper()}")
                print(f"MỤC TIÊU: {target_count} ảnh / TỔNG ASIN ĐANG CÓ: {len(asin_list)}")
                print(f"{'='*50}")
                
                # Trộn ngẫu nhiên danh sách ASIN để đảm bảo tính phân phối đều
                random.shuffle(asin_list)
                
                success_count = 0
                
                # BƯỚC 1: Đếm số lượng ảnh của danh mục này đã tải trước đó (Tính năng Resume)
                for asin in asin_list:
                    if os.path.exists(os.path.join(dataset_dir, f"{asin}.jpg")):
                        success_count += 1
                
                if success_count >= target_count:
                    print(f"[HOÀN THÀNH] Category '{category_name}' đã đạt {success_count}/{target_count} ảnh. Bỏ qua.")
                    continue
                else:
                    print(f"[*] Tiến trình hiện tại: Đã có sẵn {success_count}/{target_count} ảnh. Bắt đầu cào thêm...")

                # BƯỚC 2: Cào dữ liệu cho đến khi đủ Quota
                for idx, asin in enumerate(asin_list):
                    # Dừng cào nếu đã đủ chỉ tiêu
                    if success_count >= target_count:
                        print(f"\n[XONG CATEGORY] Đã đạt chỉ tiêu {target_count} ảnh cho {category_name}!")
                        break
                        
                    # Bỏ qua vòng lặp Web driver nếu file ảnh đã tồn tại cục bộ
                    if os.path.exists(os.path.join(dataset_dir, f"{asin}.jpg")):
                        continue

                    url = f"https://www.amazon.com/dp/{asin}"
                    print(f"\n[{success_count+1}/{target_count}] Đang thử ASIN: {asin} (Từ dòng thứ {idx+1}/{len(asin_list)})")
                    driver.get(url)
                    
                    # Thời gian chờ ngẫu nhiên để mô phỏng người dùng thực, giảm nguy cơ bị chặn
                    time.sleep(random.uniform(2.5, 4.5)) 

                    try:
                        img_element = wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "#landingImage, #imgBlkFront"))
                        )
                        
                        img_url = img_element.get_attribute("data-old-hires")
                        if not img_url or img_url.strip() == "":
                            img_url = img_element.get_attribute("src")
                        
                        if img_url:
                            if download_image(img_url, dataset_dir, asin):
                                success_count += 1
                        else:
                            print(f"[Bỏ qua] URL rỗng cho ASIN {asin}")
                            
                    except Exception:
                        print(f"[Bỏ qua] Lỗi trích xuất ASIN {asin} (Hết hàng/CAPTCHA/Sai UI).")

    finally:
        driver.quit()

if __name__ == "__main__":
    classified_asins = extract_asins_by_category()
    
    print("=== TỔNG QUAN ASIN THU THẬP ĐƯỢC ===")
    for ds, cats in classified_asins.items():
        print(f"Dataset {ds.upper()}:")
        for cat, asins in cats.items():
            print(f"  - {cat}: {len(asins)} mã ASIN")
            
    print("\nKhởi động quá trình tải ảnh với cơ chế cân bằng danh mục...")
    scrape_amazon_images(classified_asins, base_dir=BASE_OUTPUT_DIR)