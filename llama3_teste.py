import transformers
import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Teste o pipeline
output = pipeline("Hello, how are you?")
print(output)