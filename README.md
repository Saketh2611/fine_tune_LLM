# 🏥 Medical AI Assistant (Fine-Tuned Mistral 7B)

A specialized medical chatbot powered by a **Fine-Tuned Mistral 7B** model. This application uses **LoRA (Low-Rank Adaptation)** adapters trained on medical flashcards to answer health-related queries with high accuracy.

It is engineered to run on **consumer hardware** by utilizing **4-bit quantization** (Windows) and **MPS acceleration** (Mac).

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
medical-ai-app/
│
├── model/                     # <--- PASTE YOUR GOOGLE DRIVE FILES HERE
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer.model
│   └── special_tokens_map.json
│
├── app.py                     # Main application script (Choose version below)
├── requirements.txt           # List of dependencies
└── README.md                  # This file
```

---

## 🛠️ Installation

### 1. Prerequisites
* **Python 3.10+** installed.
* **Hardware:**
    * *Windows:* Nvidia GPU with at least 6GB VRAM.
    * *Mac:* M1/M2/M3 chip with 16GB RAM (Recommended).

### 2. Setup Virtual Environment
**Windows:**
```bash
python -m venv ai_env
ai_env\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv ai_env
source ai_env/bin/activate
```

### 3. Install Dependencies
Create a file named `requirements.txt`:
```text
torch
transformers
peft
gradio
accelerate
protobuf
sentencepiece
scipy
bitsandbytes; sys_platform == "win32" or sys_platform == "linux"
```

Run the install command:
```bash
pip install -r requirements.txt
```

**⚠️ WINDOWS USERS ONLY:**
You must install the **GPU-enabled** version of PyTorch manually, or it will crash. Run this:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

---

## 💻 The Application Code (`app.py`)

Choose the code block that matches your operating system.

### **Option A: Windows (Nvidia GPU)**
*Features: 4-bit Quantization, Disk Offloading, Crash Prevention.*

```python
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

# 1. SETUP PATHS
adapter_path = "./model" 
base_model_name = "mistralai/Mistral-7B-v0.1"

# [FIX] Create offload folder to prevent "KeyError"
offload_folder = "./offload_folder"
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder)

print("⏳ Loading model (Windows Optimized)...")

# 2. CONFIGURE 4-BIT LOAD (Prevents OOM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

# 3. LOAD BASE MODEL
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder=offload_folder,
    low_cpu_mem_usage=True
)

# 4. LOAD ADAPTERS
model = PeftModel.from_pretrained(
    base_model, 
    adapter_path,
    offload_folder=offload_folder,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 5. CHAT FUNCTION
def ask_medical_bot(message, history):
    prompt = f"<s>[INST] {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.3)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# 6. LAUNCH
gr.ChatInterface(ask_medical_bot, title="🏥 Medical AI (Windows)").launch()
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
