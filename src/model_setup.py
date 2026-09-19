import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

target_name = "Qwen/Qwen2.5-1.5B-Instruct"
draft_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(target_name)
target_model = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=torch.float16, device_map="cuda")
draft_model = AutoModelForCausalLM.from_pretrained(draft_name, torch_dtype=torch.float16, device_map="cuda")

target_model.eval()
draft_model.eval()

greedy_config = GenerationConfig(
    do_sample=False,
    num_beams=1,
    temperature=None,
    top_p=None,
    top_k=None,
    repetition_penalty=1.0
)
