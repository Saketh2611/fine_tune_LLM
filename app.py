import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

# --------------------------------------------------------------------
# 1. SETUP PATHS & OFFLOAD FOLDER
# --------------------------------------------------------------------
adapter_path = "./model" 
base_model_name = "mistralai/Mistral-7B-v0.1"

# Create offload folder to prevent "KeyError"
offload_folder = "./offload_folder"
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder)

print("⏳ Loading model with detailed offload configuration...")

# --------------------------------------------------------------------
# 2. CONFIGURE 4-BIT LOAD
# --------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

# --------------------------------------------------------------------
# 3. LOAD BASE MODEL
# --------------------------------------------------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder=offload_folder,
    low_cpu_mem_usage=True
)

# --------------------------------------------------------------------
# 4. LOAD ADAPTERS
# --------------------------------------------------------------------
model = PeftModel.from_pretrained(
    base_model, 
    adapter_path,
    offload_folder=offload_folder,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded successfully!")

# --------------------------------------------------------------------
# 5. CHAT FUNCTION
# --------------------------------------------------------------------
def ask_medical_bot(message, history):
    prompt = f"<s>[INST] {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()
    return response

# --------------------------------------------------------------------
# 6. LAUNCH APP
# --------------------------------------------------------------------
# [FIXED] Removed 'theme="soft"' to prevent the TypeError
demo = gr.ChatInterface(
    fn=ask_medical_bot,
    title="🏥 Medical AI Assistant",
    description="Fine-tuned Mistral 7B (Running Locally with Disk Offload)"
)

if __name__ == "__main__":
    demo.launch()
