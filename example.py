import torch
from minisgl.llm import LLM
from minisgl.core import SamplingParams

llm = LLM(
    model_path="/home/jeff/work/demo/Qwen3-4B-Instruct-2507",
    dtype=torch.bfloat16,
    attention_backend="fi",
)

prompts = [
    "请自我介绍一句。",
    "1+1 等于几？",
]

sp = SamplingParams(temperature=0.7, max_tokens=128)

outputs = llm.generate(prompts, sp)
for o in outputs:
    print(">>", o["text"])
