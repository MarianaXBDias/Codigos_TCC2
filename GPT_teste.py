#importar bibliotecas
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

#configurar o modelo
def setup_llama(model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
    #Define onde o modelo será rodado
    #Checa a disponibilidade de GPU p/ rodar o modelo
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    #inicializar o tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    #Verifica se possui tokens especiais e define caso não existam
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    #carregando o modelo
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32, # precisão de float16 caso não utiliza GPU
        device_map="auto", #dividir entre CPU e GPU
        load_in_8bit=device == "cuda", #reduzir o tamanho do modelo se usar GPU
    )
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    #entrada
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    
    #saida
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


model, tokenizer = setup_llama()

#teste
prompt = "Once upon a time"

generated_text = generate_text(model, tokenizer, prompt)
print(f"Generated text:\n{generated_text}")