# KỊCH BẢN QUAY VIDEO DEMO — Multi-Agent LLM System
# Thời lượng: 60–75 giây
# Yêu cầu: Multi-turn ≥ 3 agents khác nhau

---

## TỔNG QUAN KỊCH BẢN

| Agent | Intent | Số turn | Thời gian |
|-------|--------|---------|-----------|
| **ConsultantAgent** | consultant | Turn 1 | ~12s |
| **OrderAgent** | order | Turn 2 | ~12s |
| **FAQAgent** | faq | Turn 3 | ~12s |
| **OrderAgent** | order | Turn 4 | ~10s |
| **FAQAgent** | faq | Turn 5 | ~10s |
| **Outro** | — | — | ~6s |
| **Tổng** | **4 agents** | **5 turns** | **~62s** |

---

## TRƯỚC KHI QUAY — Checklist

- [ ] Ollama đang chạy (`ollama serve`)
- [ ] Neo4j đang chạy
- [ ] Redis đang chạy
- [ ] API server đang chạy (`uvicorn app.main:app --reload --port 8000`)
- [ ] Postman collection đã import
- [ ] Mở Swagger UI: http://localhost:8000/docs
- [ ] Mở terminal chạy lệnh curl để demo (backup nếu Postman lỗi)
- [ ] Đặt camera/record màn hình

**Công cụ quay:** Loom, OBS, QuickTime Screen Recording, hoặc bất kỳ tool nào ghi được màn hình + mic.

---

## KỊCH BẢN CHI TIẾT

---

### [0:00 – 0:05] INTRO (5 giây)

**MÀN HÌNH:** Mở terminal đang chạy server + Swagger UI http://localhost:8000/docs

**SPEAK:**
> "Xin chào, tôi là Nguyễn Hữu Việt. Hôm nay tôi sẽ demo hệ thống Multi-Agent LLM cho bài test AI NLP Round 2 — một robot tư vấn F&B cho chuỗi cửa hàng Highlands Coffee, chạy hoàn toàn local không dùng cloud API."

**ACTION:** Nhìn thẳng vào camera, giới thiệu bản thân rõ ràng.

---

### [0:05 – 0:08] KIẾN TRÚC HỆ THỐNG (3 giây)

**MÀN HÌNH:** Mở file kiến trúc hoặc draw.io diagram trên màn hình

**SPEAK (nhanh):**
> "Hệ thống gồm 4 thành phần: Router phân loại intent, 4 agents chuyên biệt, Graph RAG trên Neo4j + ChromaDB, và caching nhiều tầng. Hãy xem nó hoạt động thực tế."

---

### [0:08 – 0:20] TURN 1 — Consultant Agent (12 giây)

**MÀN HÌNH:** Postman — gửi request "Có gì ngon không?"

**ACTION:** Gõ vào Postman body:
```json
{"query": "Có gì ngon không? Gợi ý cho tôi món gì đi", "session_id": "demo-001"}
```

**SPEAK:**
> "Tôi bắt đầu với một câu hỏi tư vấn. Hệ thống nhận diện intent là **consultant**, chuyển sang Consultant Agent. Agent truy vấn RAG để gợi ý món phù hợp."

**CHỜ** response hiện ra (~10-15s trên M1 Max). Khi câu trả lời hiện:
> "Agent trả lời: ..."

**MÀN HÌNH:** Zoom vào response JSON hiển thị `"intent": "consultant"` + câu trả lời.

---

### [0:20 – 0:32] TURN 2 — Order Agent (12 giây)

**MÀN HÌNH:** Cùng Postman session `demo-001` — gửi request tiếp

**ACTION:** Gõ:
```json
{"query": "Cho anh 1 ly cà phê sữa đá size M, thêm 1 bánh croissants", "session_id": "demo-001"}
```

**SPEAK:**
> "Tôi tiếp tục trong cùng session. Lần này tôi đặt hàng. Router nhận diện intent là **order**, chuyển sang Order Agent. Agent truy xuất menu từ Neo4j để lấy giá và thông tin món."

**CHỜ** response (~10-15s).

**MÀN HÌNH:** Zoom vào `"intent": "order"` + `"sources"` chứa menu items.

---

### [0:32 – 0:44] TURN 3 — FAQ Agent (12 giây)

**MÀN HÌNH:** Cùng session `demo-001`

**ACTION:** Gõ:
```json
{"query": "Wifi tên gì vậy?", "session_id": "demo-001"}
```

**SPEAK:**
> "Tôi hỏi thông tin cửa hàng — intent **faq**. FAQ Agent truy vấn Neo4j graph để lấy thông tin chính sách và store policy."

**CHỜ** response.

**MÀN HÌNH:** Zoom `"intent": "faq"` + nội dung trả lời wifi.

**ĐIỂM QUAN TRỌNG:** Cùng session `demo-001` — hệ thống nhớ lịch sử 4 câu trước. Đây là proof của multi-turn session management.

---

### [0:44 – 0:54] TURN 4 — Order Agent (10 giây)

**MÀN HÌNH:** Cùng session `demo-001`

**ACTION:** Gõ:
```json
{"query": "Tính tiền giúp tôi", "session_id": "demo-001"}
```

**SPEAK:**
> "Cuối cùng tôi yêu cầu tính tiền — vẫn trong cùng session. Order Agent nhớ các món đã gọi ở Turn 2: cà phê sữa đá size M và bánh croissants. Đây chính là ví dụ về **multi-turn conversation** — hệ thống duy trì context xuyên suốt."

**CHỜ** response.

**MÀN HÌNH:** Zoom vào câu trả lời tính tiền.

---

### [0:54 – 0:62] OUTRO (8 giây)

**MÀN HÌNH:** Quay lại terminal hoặc Swagger UI

**SPEAK:**
> "Hệ thống đã xử lý đúng 4 intent: consultant, order, faq, order — qua nhiều turns trong cùng session. Điểm nổi bật: hybrid Graph RAG, semantic cache với hit rate 83%, cascade routing đảm bảo hệ thống không fail, và streaming token-by-token qua SSE. Cảm ơn mọi người đã theo dõi."

**ACTION:** Mỉm cười, gật đầu. Kết thúc.

---

## CÂU LỆNH BACKUP (nếu Postman gặp lỗi)

Chạy trực tiếp trong terminal để demo nếu Postman không hoạt động:

```bash
# Turn 1: Consultant
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Có gì ngon không?","session_id":"demo-001"}'

# Turn 2: Order
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Cho anh 1 ly cà phê sữa đá size M, thêm 1 bánh croissants","session_id":"demo-001"}'

# Turn 3: FAQ
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Wifi tên gì vậy?","session_id":"demo-001"}'

# Turn 4: Order (tính tiền)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Tính tiền giúp tôi","session_id":"demo-001"}'
```

---

## MẸO QUAY

1. **Quay ở 1080p** — hình rõ ràng để reviewer đọc JSON response
2. **Tăng font terminal** lên 18-20px trước khi quay
3. **Đặt mic gần** — giọng nói rõ ràng
4. **Đợi response đầy đủ** — không cắt khi server đang streaming
5. **Giữ yên 2-3s** sau mỗi response — để reviewer đọc
6. **Chuẩn bị sẵn 2-3 take** — lần 1 có thể lúng túng
7. **Thời gian chờ mỗi turn** trên M1 Max Ollama: ~12-20s — bình thường, đừng lo

---

## RUBRIC CHECKLIST

| Yêu cầu | Đạt? |
|----------|-------|
| ≥ 60 giây | ✅ ~62 giây |
| Multi-turn conversation | ✅ 4-5 turns |
| ≥ 3 agents khác nhau | ✅ 3 agents: Consultant, Order, FAQ |
| Demo end-to-end | ✅ Router → Agent → RAG → Response |
