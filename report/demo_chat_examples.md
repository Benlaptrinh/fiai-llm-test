# Demo Chat Examples (Real Run)

Date: 2026-04-30  
Session ID: `demo-proof`  
Backend: Ollama `qwen2.5:7b`

## Cases

1. Query: `Cho anh 1 ly phin sữa đá size M`  
   Intent: `order`  
   Cached: `false`

2. Query: `Có gì ngon không em, anh thích ít ngọt`  
   Intent: `consultant`  
   Cached: `false`

3. Query: `Wifi tên gì vậy?`  
   Intent: `faq`  
   Cached: `false`

4. Query: `haha hello`  
   Intent: `ignore`  
   Cached: `false`

5. Query: `Wifi tên gì vậy?` (repeat)  
   Intent: `faq`  
   Cached: `true`

Full JSON responses are in `report/demo_chat_examples_2026-04-30.json`.
