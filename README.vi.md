# LocalTabula: Truy vấn dữ liệu bảng bằng ngôn ngữ tự nhiên
[![License: GPL-3.0](https://img.shields.io/badge/license-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)[![English README](https://img.shields.io/badge/README-English-blue)](README.md)

**LocalTabula** là một ứng dụng Streamlit biến bảng tính của bạn thành cuộc trò chuyện trực tiếp trên máy. Kéo thả file Excel hoặc dán link CSV từ Google Sheet công khai, rồi đặt câu hỏi bằng tiếng Anh hay tiếng Việt—không cần chờ đám mây, không lo lộ dữ liệu. Tất cả diễn ra ngay trên máy bạn, dùng SQLite và Qdrant lưu trữ vector để tìm kiếm thông minh.

---

## Tại Sao Phải Chạy Local?

Có thể ChatGPT hay Claude cho kết quả SQL rất tốt, và nhiều dịch vụ cloud sẵn sàng đáp ứng hiệu suất cao. Nhưng nếu bạn quan tâm tới:

* **Bảo mật & Tuân thủ**
* **Ngân sách cố định**
* **Tùy biến triệt để**

thì chạy local vẫn là lựa chọn duy nhất đảm bảo dữ liệu nằm trong tầm kiểm soát và chi phí không “lố” mất kiểm soát.

1. **Chủ Quyền Dữ Liệu & Tuân Thủ**

   Dữ liệu ở lại trong hệ thống, không chui ra internet. GDPR, HIPAA hay chính sách nội bộ cứ yên tâm.

2. **Chi Phí Định Trước**

   Không lo thanh toán theo token hay hoá đơn bất ngờ. Một lần đầu tư phần cứng, sau đó chạy thả ga miễn phí.

3. **Độ Trễ & Độ Ổn Định**

   Phản hồi nhanh, ổn định, không phụ thuộc mạng hay uptime của dịch vụ bên ngoài. Có thể chạy hoàn toàn trên môi trường offline.

4. **Chạy Trên Phần Cứng Khiêm Tốn**

   LocalTabula hỗ trợ GPU từ 4 GB VRAM: bạn offload Gemma3-4B 4-bit lên GPU, còn model SQL 1.3B để CPU, là đã có inference mạnh mẽ mà không cần datacenter.

5. **Chỉnh Độ Chính Xác Với Model Nhỏ**

   Với pip-sql-1.3b, bạn dễ dàng đạt \~90% chính xác trên các câu hỏi cơ bản, nhưng khi vào phần hỏi lắt léo, độ chính xác có thể rớt còn \~30%. LocalTabula giải quyết bằng pipeline đa giai đoạn cùng prompt engineering—template thoải mái sửa, few-shot, retry loop, feedback prompt—để “vỗ” model nhỏ thành công cụ mạnh mẽ.

6. **Tùy Biến & Mở Rộng**

   Thay prompt, swap model, thêm logic retry, hoặc xây RAG/agent lên trên—bạn nắm hoàn toàn roadmap, không bị khoá bên vendor.

---

**Tóm Lại:** Nếu ưu tiên của bạn là bảo mật, chi phí cố định, tùy biến sâu, và biến model nhẹ thành “ngon lành,” thì local-first không chỉ là lựa chọn—mà là bắt buộc. LocalTabula giúp bạn khởi động nhanh, ngay cả trên phần cứng khiêm tốn.

---

## Bạn Có Thể Làm Gì Với LocalTabula

* **Chuyển data & Lập Index:** Nhập file csv hoặc google sheet, làm sạch tên cột, tạo cơ sở SQLite, tự động chọn cột text để embedding.
* **Truy Vấn Bằng Ngôn Ngữ Tự Nhiên:** Hỏi “Doanh thu Q1 theo vùng?” hay “Tìm sản phẩm giống X.”
* **Chuyển Hướng Thông Minh:** Ứng dụng tự quyết SQL hay semantic search, kèm schema và sample rows để LLM có ngữ cảnh.
* **Kiểm Tra & Tùy Chỉnh:** Mở rộng câu lệnh SQL, xem trước kết quả thô hoặc embeddings, re-index khi cần.
* **Offline hoặc API:** Giai đoạn dev dùng OpenRouter, deploy thì chạy offline với GGUF (có thể bật GPU).

---

## Hoạt Động Bên Trong: Pipeline 5 Giai Đoạn

1. **Tiền Xử Lý & Chuẩn Hoá**
   Biến câu hỏi của bạn thành English rõ ràng để local model (hoặc model tùy chỉnh) hiểu tốt nhất. Muốn bật/tắt, chỉnh `prompts.yaml`.

2. **Chọn Luồng & Chuẩn Bị Schema**
   Xác định SQL hay semantic, gắn schema và vài sample rows để LLM có ngữ cảnh.

3. **Tinh Chỉnh Prompt (Tùy Chọn)**
   LLM “đánh bóng” câu hỏi thành prompt SQL chuẩn. Dành cho người không chuyên—nếu bạn giỏi SQL, có thể skip.

4. **Thực Thi & Dự Phòng**

   * **SQL Mode:** Sinh, kiểm tra, chạy SQL trên SQLite. Lỗi thì retry 1 lần.
   * **Semantic Mode:** Embed câu hỏi, tìm top-k kết quả Qdrant, trả snippets.
     Nếu route chính trả trống, route kia sẽ chạy thay.

5. **Tổng Hợp Kết Quả**
   Đưa kết quả thô vào LLM để sinh câu trả lời tự nhiên, không chỉ là bảng khô. Muốn raw ouput? Chỉnh `generate_final_summary` trong `prompts.yaml`.

---

## Những “Vít” Bạn Có Thể Vặn

Tất cả nằm trong **`.env`**, **`config/prompts.yaml`**, và **`utils.py`**:

| Giai Đoạn               | File / Hàm                                | Có Thể Chỉnh                                      |
| ----------------------- | ----------------------------------------- | ------------------------------------------------- |
| Chuẩn Hoá Ngôn Ngữ      | `prompts.yaml`                            | Thay logic dịch, tắt chức năng                    |
| Chọn DB                 | `select_database_id` / `prompts.yaml`     | Thay sample size, template prompt, fallback logic |
| Tinh Chỉnh & Định Tuyến | `refine_and_select` / `prompts.yaml`      | Few-shot, temperature, flag force-mode            |
| Sinh SQL                | `generate_sql_*` / `aux_models`           | Swap model NL→SQL, sửa ví dụ, retry logic         |
| Thực Thi SQL            | `utils._execute_sql_query`                | Đổi path DB, pragma, timeout                      |
| Tìm Kiếm Ngữ Nghĩa      | `init_qdrant_client` / `aux_models`       | Swap embedding, top-k, metric                     |
| Tổng Hợp                | `generate_final_summary` / `prompts.yaml` | Sửa tone, độ chi tiết, hoặc tắt hẳn               |

---

## Cách Bắt Đầu

1. **Clone & Kích Hoạt**

   ```bash
   git clone <repo-url> && cd <repo>  
   python -m venv venv && source venv/bin/activate  
   ```

2. **Cài Đặt Thư Viện**

   ```bash
   pip install -r requirements.txt
   ```

   ⚠️ **Sử dụng GPU**

   Bạn phải build `llama-cpp-python` với flag CMake phù hợp—không đơn giản đâu. Xem README trong thư mục **config** để biết chi tiết.

3. **Cấu Hình**
   Sao chép `.env.example` thành `.env` và thiết lập:

   * `DEVELOPMENT_MODE` (true cho OpenRouter, false cho local GGUF)
   * Đường dẫn/repo ID model GGUF
   * API key (nếu dùng OpenRouter)

4. **Chạy Ứng Dụng**

   ```bash
   streamlit run app.py
   ```

   Mở `http://localhost:8501` và bắt đầu “chat” với dữ liệu của bạn!

---

## **Work in Progress (2025.05.13)**

1. **Giao Diện & Bảng Cấu Hình**
   Pipeline lõi đã ổn định, nhưng giao diện vẫn còn cơ bản. Sẽ có trang settings để chỉnh prompt, retry, max tokens…

2. **Trình Chạy SQL Thủ Công**
   Thêm console SQL tương tác để bạn tự viết và chạy truy vấn song song với pipeline tự động.

3. **Hỗ Trợ Google Colab**
   Có notebook `main.ipynb` thử nghiệm, dùng ngrok để expose Streamlit. Colab chạy CUDA 12.5 nên cấu hình GPU hơi lằng nhằng—nếu dùng Colab, khuyến nghị model llama-3.1 8B để mượt mà hơn.
