import torch
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI()

# 1. SETUP PATHS (Matches your folder structure)
adapter_path = "./model" 
base_model_name = "mistralai/Mistral-7B-v0.1"
offload_folder = "./offload_folder"
os.makedirs(offload_folder, exist_ok=True)

# 2. LOAD MODEL (Reusing your stable Windows config)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

print("⏳ Loading Fine-Tuned Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder=offload_folder
)
model = PeftModel.from_pretrained(base_model, adapter_path, offload_folder=offload_folder)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model Ready for Custom Frontend!")

# 3. API ENDPOINT FOR CHAT
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message")
    
    prompt = f"<s>[INST] {user_message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3)
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("[/INST]")[-1].strip()
    return {"response": response}

# 4. SERVE YOUR HTML FILES
# This looks into your 'static' folder for index.html, script.js, and style.css
app.mount("/", StaticFiles(directory="static", html=True), name="static")