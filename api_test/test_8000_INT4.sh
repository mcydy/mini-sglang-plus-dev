curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/jeff/work/demo/Qwen3-4B-Instruct-2507",
    "messages": [{"role":"user","content":"教我moe"}],
    "max_tokens": 64
  }'
  
