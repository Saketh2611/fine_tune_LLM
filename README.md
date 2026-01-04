# 🏥 Medical AI Assistant (Fine-Tuned Mistral 7B)

A full stack specialized medical chatbot powered by a **Fine-Tuned Mistral 7B** model deployed with Docker. This application uses **LoRA (Low-Rank Adaptation)** adapters trained on medical flashcards to answer health-related queries with high accuracy.

It is engineered to run on **consumer hardware** by utilizing **4-bit quantization** (Windows) and **MPS acceleration** (Mac).

---
The project supports:
- 🧪 Local experimentation using Jupyter Notebook  
- 🌐 Web-based UI (HTML/CSS/JS)  
- 🐳 Production-ready Docker deployment  

---

## 🚀 Features

* **Specialized Knowledge:** Fine-tuned on the `medalpaca/medical_meadow_medical_flashcards` dataset.
* **Efficient Inference:**
    * **Windows:** Uses **QLoRA (4-bit)** with Disk Offloading to run on 6GB+ VRAM cards.
    * **Mac:** Uses **Metal Performance Shaders (MPS)** for hardware acceleration on M1/M2/M3 chips.
* **Hardware Resilience:** Implements offloading techniques to prevent "Out of Memory" crashes.
* **User Interface:** Clean, browser-based chat interface built with **Gradio**.

---

## 📂 Project Structure

Ensure your project folder looks exactly like this.

```text
medical-ai-assistant/
├── model/
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── app.py
├── app_mac.py
├── main.py
├── Fine_Tune_LLM.ipynb
├── jupiter-notebook.ipynb
├── Dockerfile
├── requirements.txt
└── README.md                 
```

---
## Model Details

- Base Model: Mistral-7B  
- Fine-Tuning: LoRA / PEFT  
- Domain: Medical Question Answering  

---

## 🛠️ Local Development (Jupyter Notebook)

Install dependencies:
pip install -r requirements.txt

Run:
jupyter notebook

Use notebooks for model testing and experimentation.

---

## 🌐 Running Without Docker

python main.py

Backend runs at:
http://localhost:8000

Open static/index.html in your browser.

---

## 🐳 Docker Deployment

Build image:
docker build -t medical-ai-app .

Run container:
docker run -p 8000:8000 medical-ai-app

---

## 📡 API Endpoint

POST /chat

Request:
{
  "message": "What are the symptoms of diabetes?"
}

Response:
{
  "response": "Common symptoms of diabetes include..."
}

---

## 💻 The Application Code (`app.py`)

Choose the code block that matches your operating system.

### **Option A: Windows (Nvidia GPU)**
*Features: 4-bit Quantization, Disk Offloading, Crash Prevention.*

```python
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
```

### **Option B: Mac (Apple Silicon M1/M2)**
*Features: MPS Acceleration (Metal), FP16 Precision.*

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# 1. SETUP
adapter_path = "./model" 
base_model_name = "mistralai/Mistral-7B-v0.1"

print("🍎 Loading model on Apple Silicon...")

# 2. LOAD BASE MODEL (Native Mac Support)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="mps",  # Uses Mac GPU
)

# 3. LOAD ADAPTERS
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload() # Optimization safe on Mac
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4. CHAT FUNCTION
def ask_medical_bot(message, history):
    prompt = f"<s>[INST] {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.3)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# 5. LAUNCH
gr.ChatInterface(ask_medical_bot, title="🏥 Medical AI (Mac)").launch()
```

---

## 🐛 Troubleshooting Journey (Common Issues)

During development, we solved these critical issues. Check here if you get stuck.

| Error | Platform | Solution |
| :--- | :--- | :--- |
| **`OSError: [Errno 28] No space left on device`** | Win/Mac | The C: drive is full. Delete `%userprofile%\.cache\huggingface` to free up 20GB. |
| **`ValueError: Some modules dispatched on CPU/disk...`** | Windows | You need to enable offloading. Use the **Windows Code (Option A)** provided above. |
| **`KeyError: 'base_model.model.model...'`** | Windows | A bug in `peft`. Fixed by adding `offload_folder="./offload_folder"` to the code. |
| **`AssertionError: Torch not compiled with CUDA`** | Windows | You have the CPU version of PyTorch. Reinstall using the command in "Installation" above. |
| **App crashes silently / 0% Load** | Windows | Increase your Virtual Memory (Pagefile) to 25GB in System Settings. |
| **`mps` device not found** | Mac | Ensure you are on a Mac with an M-Series chip (M1/M2/M3) and MacOS 12+. |

---

## 📜 Credits
* **Base Model:** Mistral AI (Mistral-7B-v0.1)
* **Dataset:** MedAlpaca (Medical Meadow Flashcards)
* **Frameworks:** Hugging Face Transformers, PEFT, Gradio.
