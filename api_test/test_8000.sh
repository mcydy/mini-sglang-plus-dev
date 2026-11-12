curl http://127.0.0.1:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
       "model": "OLMoE-1B-7B-0924",
       "messages": [{"role":"user","content":"花儿为什么这么红"}],
       "max_tokens": 64
     }'
