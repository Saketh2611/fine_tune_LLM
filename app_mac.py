import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# --------------------------------------------------------------------
# 1. SETUP PATHS
# --------------------------------------------------------------------
adapter_path = "./model"  # Ensure your adapter files are in this folder
base_model_name = "mistralai/Mistral-7B-v0.1"

print("🍎 Loading model on Apple Silicon (MPS)...")

# --------------------------------------------------------------------
# 2. LOAD BASE MODEL (Mac Optimized)
# --------------------------------------------------------------------
# We use torch.float16 which fits on 16GB Macs. 
# If your friend has 8GB RAM, this step might crash.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="mps",  # <--- This forces it to use the Mac GPU
)

# --------------------------------------------------------------------
# 3. APPLY YOUR ADAPTERS
# --------------------------------------------------------------------
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload() # Merges them for faster speed

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded successfully on Mac!")

# --------------------------------------------------------------------
# 4. CHAT FUNCTION
# --------------------------------------------------------------------
def ask_medical_bot(message, history):
    prompt = f"<s>[INST] {message} [/INST]"
    
    # Move input to Mac GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()
    return response

# --------------------------------------------------------------------
# 5. LAUNCH APP
# --------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=ask_medical_bot,
    title="🏥 Medical AI (Mac Version)",
    description="Running locally on Apple Silicon",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
