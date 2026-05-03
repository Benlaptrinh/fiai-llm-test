# FI-AI Multi-Agent LLM — Hướng dẫn chạy

## 1. Cài đặt môi trường

### 1.1 Clone & Setup Python

```bash
git clone https://github.com/Benlaptrinh/fiai-llm-test.git
cd fiai-llm-test

# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Cài dependencies
pip install -r requirements.txt
```

### 1.2 Ollama

```bash
# Cài Ollama
brew install ollama        # macOS
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Pull model qwen2.5:7b (6.1GB, cần ~10GB RAM)
ollama pull qwen2.5:7b

# Pull model qwen2.5:1.5b (để test SLM router)
ollama pull qwen2.5:1.5b

# Khởi động Ollama server (chạy nền)
ollama serve
```

### 1.3 Neo4j (Docker)

```bash
docker run \
  --name fiai-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5-community
```

Sau khi Neo4j khởi động:
- Truy cập http://localhost:7474 (Neo4j Browser)
- Đăng nhập: `neo4j` / `password123`
- Chạy ingest data:

```bash
python3 -c "
from app.ingest import run_ingest
run_ingest()
print('Done!')
"
```

### 1.4 Redis (Docker)

```bash
docker run \
  --name fiai-redis \
  -p 6379:6379 \
  redis:7-alpine
```

### 1.5 ChromaDB (Tự động)

ChromaDB được khởi tạo tự động khi server start lần đầu. Không cần setup thủ công.

---

## 2. Chạy API Server

### 2.1 Copy và chỉnh .env

```bash
cp .env.example .env
# Chỉnh các giá trị nếu cần (mặc định đã OK cho local dev)
```

### 2.2 Khởi động

```bash
# Development (auto-reload khi code thay đổi)
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0

# Production
uvicorn app.main:app --port 8000 --host 0.0.0.0
```

**Server chạy tại:** http://localhost:8000

**API docs tự động:** http://localhost:8000/docs (Swagger UI)

---

## 3. Test nhanh

```bash
# Health check
curl http://localhost:8000/health

# Chat (FAQ)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Wifi tên gì vậy?","session_id":"test-001"}'

# Chat (Order)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Cho anh 1 ly cà phê sữa đá","session_id":"test-002"}'

# Chat (Consultant)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Có gì ngon không? Gợi ý cho tôi","session_id":"test-003"}'

# Chat (Ignore)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"haha hello","session_id":"test-004"}'

# Streaming
curl -N http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query":"Mấy giờ đóng cửa?","session_id":"stream-001"}'

# Cache stats
curl http://localhost:8000/cache/stats

# Invalidate cache
curl -X POST http://localhost:8000/cache/invalidate
```

---

## 4. Import Postman Collection

### 4.1 Import JSON

1. Mở Postman
2. Click **Import** → Chọn file `postman/FIAI_MultiAgent_LLM.postman_collection.json`
3. Collection "FI-AI Multi-Agent LLM F&B Assistant" sẽ xuất hiện

### 4.2 Thiết lập Environment

1. Click **Environments** (biểu tượng bánh răng góc phải trên)
2. Tạo environment mới: `FI-AI Local`
3. Thêm variable:

| Variable | Initial Value | Current Value |
|----------|---------------|---------------|
| `baseUrl` | `http://localhost:8000` | `http://localhost:8000` |

4. Chọn environment `FI-AI Local` làm active

### 4.3 Test theo thứ tự

**Thứ tự chạy đề xuất:**

```
1. Health Check                      → Verify server is up
2. Chat — Non-Streaming             → First test, check response shape
3. Chat — Order Intent              → intent=order
4. Chat — Consultant Intent         → intent=consultant
5. Chat — FAQ Intent                → intent=faq
6. Chat — Ignore Intent             → intent=ignore
7. Chat — Guardrail Block           → intent=guardrail
8. Chat — English Query             → multilingual test
9. Chat — Multi-Turn               → Same session_id, test history
10. Chat — Cache Hit Test           → Send same query twice → cached=true
11. Cache — Stats                   → Check hit/miss rate
12. Cache — Invalidate              → Clear cache
13. Chat — Streaming (SSE)          → Token-by-token response
14. Chat — Streaming Multi-Turn     → 3-agent demo (use same session)
```

### 4.4 Đọc SSE stream trong Postman

Với request **Chat — Streaming (SSE)**:
- Postman sẽ hiển thị raw SSE output
- Hoặc chuyển qua tab **Bulk Edit** để xem từng event
- Mỗi event có dạng:
  ```
  data: {"type":"metadata","intent":"faq","cached":false,"sources":[...]}
  data: {"type":"token","content":"Wi"}
  data: {"type":"token","content":"fi"}
  ...
  data: {"type":"done"}
  ```

---

## 5. Multi-Turn Demo (3 Agents)

Để demo multi-turn ≥ 3 agents (yêu cầu video ≥ 60s):

```bash
# Turn 1: Order Agent
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Cho anh 1 ly cà phê sữa đá size M","session_id":"demo-001"}'

# Turn 2: FAQ Agent (same session)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Wifi tên gì vậy?","session_id":"demo-001"}'

# Turn 3: Consultant Agent (same session)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Có gì ngon cho buổi chiều không?","session_id":"demo-001"}'

# Turn 4: Order Agent — add more (same session)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Thêm 1 bánh croissants nữa","session_id":"demo-001"}'

# Turn 5: FAQ Agent (same session)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Giờ mấy đóng cửa?","session_id":"demo-001"}'
```

Mỗi turn tiếp theo sẽ nhớ lịch sử hội thoại trước đó.

---

## 6. Troubleshooting

### Ollama not running
```bash
# Kiểm tra
curl http://localhost:11434/api/tags

# Khởi động lại
pkill ollama
ollama serve
```

### Neo4j connection refused
```bash
# Kiểm tra container đang chạy
docker ps | grep neo4j

# Restart nếu cần
docker restart fiai-neo4j
```

### Redis connection refused
```bash
docker ps | grep redis
docker restart fiai-redis
```

### Model not found (Ollama)
```bash
ollama list
ollama pull qwen2.5:7b
```

### Port 8000 đã bị占用
```bash
# Tìm process
lsof -ti:8000

# Kill và restart
kill -9 $(lsof -ti:8000)
uvicorn app.main:app --reload --port 8000
```

### Cache hit rate thấp
```bash
# Invalidate cache trước
curl -X POST http://localhost:8000/cache/invalidate

# Check stats
curl http://localhost:8000/cache/stats
```

---

## 7. Docker Compose (Production)

Nếu muốn chạy toàn bộ bằng Docker:

```bash
docker-compose up
```

---

## 8. Project Structure

```
fiai-llm-test/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── router_agent.py      # Intent classification (cascade)
│   ├── agents/              # Order, Consultant, FAQ, Ignore agents
│   ├── rag.py              # RAG store (ChromaDB + Neo4j)
│   ├── cache.py            # Multi-layer caching
│   ├── session.py          # Session management
│   ├── guardrails.py      # Harmful content filter
│   ├── ingest.py           # Data ingestion pipeline
│   └── config.py           # Environment configuration
├── data/
│   ├── menu.csv
│   ├── faq.csv
│   └── docs.txt
├── models/
│   └── router_sft/         # Fine-tuned LoRA router
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── benchmark_router_sft.py
│   ├── benchmark_reranker_b12.py
│   └── eval_full.py
├── postman/
│   └── FIAI_MultiAgent_LLM.postman_collection.json
├── report/
│   └── technical_report.pdf
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── RUN.md
```
