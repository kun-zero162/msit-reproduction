# Open-World Attribute Mining for E-Commerce Products with Multimodal Self-Correction Instruction Tuning (MSIT)

> **Tái hiện và thực nghiệm bài báo khoa học:**  
> *Open-World Attribute Mining for E-Commerce Products with Multimodal Self-Correction Instruction Tuning (MSIT)* — **ACL 2025 · Volume 1: Long Papers**

---

## 👥 Thông tin đồ án

- **Môn học:** Học máy với dữ liệu Đồ thị (Graph Machine Learning)
- **Đơn vị:** Khoa Công nghệ Thông tin – Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (FIT@HCMUS)
- **Thành viên thực hiện:**
  - **Nguyễn Hoàng Việt** — MSHV: `25C11028`
  - **Nguyễn Ngọc Hồng Lĩnh** — MSHV: `25C11011`

---

## 📌 1. Đặt vấn đề (Problem Statement)

Khai phá thuộc tính sản phẩm (**Attribute Mining - AM**) đóng vai trò cốt lõi trong thương mại điện tử nhằm chuẩn hóa danh mục sản phẩm, cải thiện công cụ tìm kiếm, hệ thống gợi ý và hỗ trợ người dùng đưa ra quyết định mua hàng. 

Bài toán **Open-World Attribute Mining (Open-World AM)** đặt ra thách thức: mô hình phải tự động phát hiện và trích xuất các thuộc tính mới mà không bị giới hạn trong schema hay tập nhãn định sẵn.

### 🚨 Hai hạn chế cốt lõi của các mô hình AM hiện tại:
1. **Lãng phí dữ liệu thị giác (Underutilizing Multimodal Info):** Các mô hình truyền thống chủ yếu phân tích tiêu đề và mô tả (text), bỏ qua các thông số kỹ thuật, hình dạng, quy cách đóng gói in trực tiếp trên bao bì sản phẩm qua hình ảnh (modality bias).
2. **Thiếu suy luận logic (Absence of Explicit Reasoning):** Các mô hình sinh trực tiếp (generative models) không trải qua quy trình suy luận kiểm chứng dẫn đến hiện tượng sinh ảo giác (*hallucination*), trích xuất sai hoặc tạo ra các thuộc tính vô nghĩa (như các lời quảng cáo thổi phồng - "Marketing Claims").

---

## ✨ 2. Đóng góp chính của MSIT

MSIT giải quyết trọn vẹn hai hạn chế trên với các điểm đột phá:

1. **Tiên phong khai thác Multimodal LLM (MLLM) cho Open-World AM:** Kết hợp triệt để cả hai kênh ảnh và văn bản.
2. **Khung huấn luyện với 2 bộ dữ liệu tinh chỉnh:** 
   - **AGTD (Attribute Generation Tuning Data):** Dạy mô hình trích xuất cặp *Attribute – Value*.
   - **CTTD (Chain-of-Thought Tuning Data):** Dạy mô hình tư duy tự sửa lỗi (*Self-Correction*).
3. **Chuỗi suy luận tự sửa lỗi 5 bước (5-Step CoT):** Ép mô hình kiểm chứng logic từng bước trước khi đưa ra quyết định, loại bỏ các thuộc tính sai lệch.
4. **Mở rộng dữ liệu đơn phương thức sang đa phương thức:** Bổ sung ảnh sản phẩm crawl từ Amazon cho 2 bộ benchmark kinh điển **WOAM** và **OAMine**. MSIT đạt hiệu năng vượt trội so với GPT-4 và các mô hình SOTA.

---

## 🏗️ 3. Kiến trúc hệ thống MSIT

Kiến trúc tổng thể của MSIT gồm 3 thành phần cốt lõi:

```
+-----------------------------------------------------------------------------------+
|                                 MSIT FRAMEWORK                                    |
+-----------------------------------------------------------------------------------+
| 1. AGTD Construction:                                                             |
|    Image ----[GPT-4/VLM]----> Image Attributes \                                  |
|                                                 +--> [Merge API] --> AGTD Pairs   |
|    Text  ----[GPT-4/VLM]----> Text Attributes  /                                  |
|                                                                                   |
| 2. CTTD Construction (5-Step CoT + Contrastive Negatives):                       |
|    Narrow Type -> Common-sense -> Image Validation -> Text Verification -> Decision|
|                                                                                   |
| 3. Attribute Inference Pipeline:                                                  |
|    [Batch Generation] --> [Word2Vec Sim Filtering] --> [Sequential 5-Step CoT]    |
+-----------------------------------------------------------------------------------+
```

### 🛠️ Giai đoạn 1: Xây dựng dữ liệu tinh chỉnh AGTD (AGTD Construction)
- Trích xuất **độc lập** thuộc tính từ ảnh và từ văn bản (2 lượt gọi API riêng biệt) trước khi hợp nhất bằng lệnh gọi thứ 3.
- *Lý do thuật toán:* Tránh hiện tượng **"modality bias"** khi đưa đồng thời ảnh và text vào cùng lúc (GPT-4 thường thiên vị đọc text và bỏ qua ảnh).

### 🧠 Giai đoạn 2: Chuỗi tự sửa lỗi 5 bước (5-Step CoT in CTTD)
Quy trình CoT ép mô hình kiểm chứng chéo thông tin qua 5 bước:
1. **Bước 1 (Product Type Narrowing):** Xác định danh mục/loại sản phẩm từ ảnh và text để thu hẹp không gian thuộc tính.
2. **Bước 2 (Common-sense Reasoning):** Đối chiếu tri thức nền: Liệu thuộc tính này có hợp lý với loại sản phẩm đó không?
3. **Bước 3 (Image-based Validation):** Dùng ảnh để xác nhận hoặc suy đoán giá trị thô ban đầu của thuộc tính.
4. **Bước 4 (Text-based Verification):** Dùng văn bản để kiểm chứng và trích xuất giá trị chính xác.
5. **Bước 5 (Final Decision):** Đưa ra quyết định cuối cùng (`Yes` / `No`) và giá trị chuẩn xác.

> **Contrastive Tuning:** Bổ sung các mẫu âm (negative samples) lấy từ sản phẩm khác loại nhằm cân bằng dữ liệu, tránh việc mô hình bị thiên lệch (overfitting) luôn trả lời "Yes".

### ⚙️ Giai đoạn 3: Quy trình suy luận thực tế (3-Stage Inference Pipeline)
1. **Stage 1 - Batch Attribute Generation:** MLLM đã fine-tune sinh loạt thuộc tính thô từ ảnh và văn bản.
2. **Stage 2 - Filtering Repeated Attributes:** Lọc thuộc tính trùng lặp ngữ nghĩa dựa trên quy tắc và độ tương đồng Cosine qua Word2Vec (`word2vec-google-news-300`).
3. **Stage 3 - Sequential Inference with 5-Step CoT:** Chạy quy trình 5 bước CoT cho từng thuộc tính còn lại để loại bỏ dương tính giả (false positives).

---

## 📊 4. Thiết lập thực nghiệm & Kết quả

### Cài đặt thực nghiệm:
- **Tập dữ liệu:**
  - **WOAM:** 4 danh mục sản phẩm, trung bình ~16.5 thuộc tính/loại.
  - **OAMine:** 100 danh mục sản phẩm, 1,943 sản phẩm test.
  - Ảnh crawl từ Amazon tương ứng với ASIN.
  - Tập huấn luyện: 1,000 mẫu AGTD + 300 mẫu CTTD; Tập kiểm thử: 1,000 mẫu.
- **Mô hình nền tảng:** LLaVA-7B, Qwen-VL (Qwen2.5-VL-7B-Instruct), InternLM.
- **Phương pháp tinh chỉnh:** LoRA (PEFT), Optimizer Adam, learning rate $3 \times 10^{-4}$, 10 epochs.
- **Cấu hình Decoding:** Top_p sampling ($p=0.7$, temperature=0.2).

### Kết quả nổi bật:
- **Vượt trội SOTA & GPT-4:** Trên WOAM & OAMine, Exact Match Precision của MSIT-LLaVA-7B đạt **35.34%** (tăng mạnh từ 10.61% của LLaVA-7B gốc), bỏ xa GPT-4 (15.51%).
- **Hiệu quả tự sửa lỗi:** Cơ chế Self-Correction sửa đúng tới **98.6%** các phán đoán âm tính ban đầu.
- **Domain Adaptation:** Thử nghiệm Cross-Validation trên các danh mục hoàn toàn mới (*Sofa, Phone, Tea, Vitamin*), MSIT dẫn đầu tuyệt đối với Similar Match Precision đạt **69.21%** và Exact Match Precision đạt **40.96%**.
- **Sức mạnh kênh Ảnh:** Kênh ảnh đạt Exact Match Precision 40.96% (so với GPT-4 là 23.81%), cung cấp số lượng thuộc tính phong phú in trên bao bì mà text bỏ sót.

---

## 📂 5. Cấu trúc thư mục dự án

```plaintext
msit-reproduction/
├── 25C11028_25C11011.pptx      # Slide thuyết trình báo cáo đồ án
├── README.md                   # Tài liệu giới thiệu và hướng dẫn đồ án
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── .gitignore                  # Cấu hình bỏ qua file venv, pycache, checkpoints
├── build_sft_data.py           # Tiền xử lý và tạo dataset SFT chuẩn format
├── data_generation.py          # Pipeline sinh dữ liệu AGTD và CTTD từ LLM/VLM
├── datasets.py                 # Xử lý và đọc dữ liệu WOAM / OAMine
├── image_collection.py         # Module crawl/tải ảnh sản phẩm từ Amazon
├── fine-tune-msit-open-world.ipynb # Notebook huấn luyện LoRA cho VLM
├── msit-open-world.ipynb       # Notebook suy luận và đánh giá toàn diện pipeline
├── msit_pipeline_results.jsonl # Kết quả chạy pipeline chi tiết
├── datasets/                   # Dữ liệu phục vụ nghiên cứu
│   ├── images/                 # Ảnh sản phẩm Amazon đã thu thập
│   ├── oamine/                 # Dữ liệu gốc và nhãn OAMine
│   ├── sft_datasets/           # Dữ liệu SFT sau khi đóng gói
│   └── woam/                   # Dữ liệu benchmark WOAM
└── results/                    # Checkpoints, AGTD/CTTD splits và logs thực nghiệm
```

---

## 🚀 6. Hướng dẫn cài đặt & Thực thi

### 6.1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10+ và môi trường ảo:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1
# (Linux/macOS)
# source .venv/bin/activate

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 6.2. Thu thập dữ liệu ảnh
Tải ảnh sản phẩm Amazon tương ứng với tập dữ liệu:
```bash
python image_collection.py
```

### 6.3. Sinh dữ liệu tinh chỉnh (AGTD & CTTD)
Sinh dữ liệu trích xuất thuộc tính độc lập và chuỗi CoT 5 bước:
```bash
python data_generation.py
python build_sft_data.py
```

### 6.4. Huấn luyện mô hình (Fine-tuning)
Mở và thực thi notebook `fine-tune-msit-open-world.ipynb` trên môi trường hỗ trợ GPU (Google Colab / Kaggle / Server GPU riêng) để fine-tune LoRA với Qwen2.5-VL hoặc LLaVA.

### 6.5. Đánh giá & Suy luận (Inference Pipeline)
Chạy notebook `msit-open-world.ipynb` để chạy kiểm thử quy trình 3 giai đoạn:
1. Batch Attribute Generation
2. Filtering Repeated Attributes (Word2Vec)
3. Sequential 5-step CoT Inference

---

## 📚 7. Tài liệu tham khảo

1. **MSIT Paper:** *Open-World Attribute Mining for E-Commerce Products with Multimodal Self-Correction Instruction Tuning*, ACL 2025 (Volume 1: Long Papers).
2. **WOAM Benchmark:** *Open-World Attribute Mining from E-Commerce Product Text*.
3. **OAMine Benchmark:** *Open-World Product Attribute Mining*.
4. **Qwen2.5-VL / LLaVA:** Vision-Language Foundation Models.
