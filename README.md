# 🏥 Medical AI Assistant (Fine-Tuned Mistral 7B)

A specialized medical chatbot powered by a **Fine-Tuned Mistral 7B** model. This application uses **LoRA (Low-Rank Adaptation)** adapters trained on medical flashcards to answer health-related queries with high accuracy.

The interface is built with **Gradio**, offering a clean, ChatGPT-like user experience that runs locally on your machine.

---

## 🚀 Features

* **Specialized Knowledge:** Fine-tuned on the `medalpaca/medical_meadow_medical_flashcards` dataset.
* **Efficient Inference:** Supports **4-bit quantization** (QLoRA) to run on consumer GPUs (Nvidia RTX 3050/3060/4060).
* **Cross-Platform:** Optimized instructions for both **Windows (CUDA)** and **Mac (MPS/Apple Silicon)**.
* **User Interface:** Web-based chat interface using Gradio.

---

## 📂 Project Structure

Ensure your project folder looks exactly like this before running:

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
├── app.py                     # Main application script (Code provided below)
├── requirements.txt           # List of dependencies
└── README.md                  # This file
```

---

## 🛠️ Installation

### 1. Prerequisites
* **Python 3.10+** installed.
* **Git** (optional, for cloning).
* **Hardware:**
    * *Windows:* Nvidia GPU with at least 6GB VRAM (Recommended).
    * *Mac:* M1/M2/M3 chip with 16GB RAM (Recommended).

### 2. Setup Virtual Environment
It is recommended to use a virtual environment to keep your system clean.

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
Create a file named `requirements.txt` with the following content:

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

Then run the installation command:
```bash
pip install -r requirements.txt
```

> **⚠️ Windows User Note:** If you get a `bitsandbytes` error later, install the Windows-specific version manually by running:
> `pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl`

---

## 💻 The Application Code (`app.py`)

Create a file named `app.py` and paste the code below. Choose the version that matches your OS.

### **Option A: For Windows (Nvidia GPU)**
*Uses 4-bit quantization for speed and low memory usage.*

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# 1. SETUP
adapter_path = "./model" 
base_model_name = "mistralai/Mistral-7B-v0.1"

# 2. LOAD MODEL (4-bit Mode)
print("⏳ Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # Keeps model small (5GB)
)

model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. CHAT FUNCTION
def ask_medical_bot(message, history):
    inputs = tokenizer(f"<s>[INST] {message} [/INST]", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.3)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# 4. LAUNCH UI
gr.ChatInterface(ask_medical_bot, title="🏥 Medical AI (Windows)").launch()
```

### **Option B: For Mac (Apple Silicon)**
*Uses 16-bit precision with MPS acceleration.*

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# 1. SETUP
adapter_path = "./model" 
base_model_name = "mistralai/Mistral-7B-v0.1"

# 2. LOAD MODEL (Mac Mode)
print("🍎 Loading model on Apple Silicon...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="mps",  # Uses Mac GPU
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload() # Optimize for Mac
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. CHAT FUNCTION
def ask_medical_bot(message, history):
    inputs = tokenizer(f"<s>[INST] {message} [/INST]", return_tensors="pt").to("mps")
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.3)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# 4. LAUNCH UI
gr.ChatInterface(ask_medical_bot, title="🏥 Medical AI (Mac)").launch()
```

---

## 🏃‍♂️ How to Run

1.  Make sure your virtual environment is active.
2.  Run the script:
    ```bash
    python app.py
    ```
    *(Note: On Mac, use `python3 app.py`)*
3.  Wait for the model to load (1-2 minutes).
4.  Click the link that appears in the terminal (e.g., `http://127.0.0.1:7860`).

---

## 🐛 Troubleshooting

| Error | Solution |
| :--- | :--- |
| **OSError: [Errno 28] No space left on device** | Your disk is full. Delete the cache at `%userprofile%\.cache\huggingface\hub` and free up 20GB. |
| **RuntimeError: CUDA out of memory** | Close other apps. Ensure `load_in_4bit=True` is ON (Windows). Reduce `max_new_tokens`. |
| **ModuleNotFoundError: bitsandbytes** | Run the manual install command mentioned in the **Installation** section. |
| **App crashes on Mac** | You likely have 8GB RAM. This model requires 16GB. Use Google Colab instead. |

---

## 📜 Credits
* **Base Model:** Mistral AI (Mistral-7B-v0.1)
* **Dataset:** MedAlpaca (Medical Meadow Flashcards)
* **Frameworks:** Hugging Face Transformers, PEFT, Gradio.
