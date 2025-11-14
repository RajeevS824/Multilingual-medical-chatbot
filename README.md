
# 🩺 Multilingual Medical Support Chatbot
---

## 📌 Project Overview

**Multilingual Medical Support Chatbot** is an **AI-powered healthcare assistant** designed to understand queries in **any language**. It leverages a **fine-tuned Flan-T5 model** for medically accurate responses and translates answers back to the user’s original language via an intuitive **Streamlit interface**. The primary goal is to **enhance healthcare accessibility**, offering multilingual support, symptom guidance, and basic health awareness information.

---

    https://huggingface.co/spaces/Rajeev8248/medical-multilingual-chatbot


## **Project Structure**

```
multilingual-medical-chatbot/
├── app.py
├── utils.py
├── fine_tune.py
├── requirements.txt
├── README.md
├── .gitignore
├── hospital.jpg
└── assets/
```

## 🛠️ Workflow
---
### 1. 🧹 Data Collection & Preprocessing

* Gathered medical QA datasets: **MedQuAD, PubMedQA, HealthCareMagic QA**.
* Cleaned and normalized text: removed special characters, HTML tags, and extra spaces.
* Tokenized and prepared data for **seq2seq model fine-tuning**.

### 2. 🤖 Model Fine-Tuning

* Selected **Flan-T5** as the base model.
* Fine-tuned on medical QA datasets to improve domain-specific responses.
* Implemented **prompt engineering** to structure responses with causes, symptoms, diagnosis, treatment, and recommendations.

### 3. 🌐 Multilingual Translation

* Detected query language using **Google Translate API**.
* Translated non-English queries to English for model input.
* Translated model responses back to the user’s original language.

### 4. 📈 Streamlit Dashboard / UI

* Built an interactive **chat interface** with:

  * Query input box
  * Language detection
  * Smart responses in original language
  * English version (optional)
  * Buttons for “Get Response” and “Clear text”
* Styled UI with **custom background, fonts, and button design**.

---

## **1️⃣ app.py**

```python
# app.py
# ==========================================================
# 🩺 Multilingual Medical Support Chatbot with Streamlit
# ==========================================================

import streamlit as st
from utils import (
    load_model_and_tokenizer,
    clean_text,
    detect_language,
    translate_text,
    generate_medical_response,
)

# ==========================================================
# Page Config
# ==========================================================
st.set_page_config(page_title="🩺 Multilingual Medical Assistant", layout="centered")

# ==========================================================
# Load Model & Tokenizer
# ==========================================================
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer, model, device = load_model_and_tokenizer()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ==========================================================
# Header
# ==========================================================
st.markdown("<h1 style='text-align:center;color:#1A5276;'>MULTILINGUAL MEDICAL ASSISTANT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#7F8C8D;'>How can I help you?</p>", unsafe_allow_html=True)

# ==========================================================
# Chat Logic
# ==========================================================
def get_chatbot_response(query):
    """Reusable chatbot logic"""
    cleaned = clean_text(query)
    lang = detect_language(cleaned)
    text_en = translate_text(cleaned, src=lang, dest="en") if lang != "en" else cleaned

    prompt = (
        "You are a professional medical assistant. "
        "Provide detailed explanation including causes, symptoms, diagnosis, treatments, and recommendations. "
        "Do not provide emergency advice; recommend consulting a doctor if needed. "
        f"Question: {text_en}\nAnswer:"
    )

    response_en = generate_medical_response(prompt, tokenizer, model, device)
    final_response = translate_text(response_en, src="en", dest=lang) if lang != "en" else response_en
    return final_response, response_en, lang

# ==========================================================
# Streamlit UI
# ==========================================================
query = st.text_area("Enter your health-related query here...")

col1, col2 = st.columns(2)
with col1:
    if st.button("🩺 Get Response"):
        if not query.strip():
            st.error("Please enter a query first.")
        else:
            with st.spinner("Generating response..."):
                final_response, response_en, lang = get_chatbot_response(query)
            st.info("### 🧠 Response")
            st.write(final_response)
            if lang != "en":
                st.divider()
                st.info("### English Response")
                st.write(response_en)

with col2:
    if st.button("Clear"):
        query = ""
        st.session_state["query"] = ""

# Disclaimer
st.markdown("""
<p style="font-size:15px; color:black; text-align:center;">
⚠️ <strong>Disclaimer:</strong> This chatbot is for educational purposes only. Not a substitute for professional medical advice.
</p>
""", unsafe_allow_html=True)
```

---

## **2️⃣ utils.py**

```python
# utils.py
# ==========================================================
# Helper functions for Multilingual Medical Chatbot
# ==========================================================

import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from googletrans import Translator

_TRANSLATOR = Translator()
MODEL_NAME = "MBZUAI/LaMini-Flan-T5-783M"

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model, device

def clean_text(text: str) -> str:
    """Normalize text"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def detect_language(text: str) -> str:
    """Detect input language"""
    try:
        return _TRANSLATOR.detect(text).lang
    except:
        return "en"

def translate_text(text: str, src: str = None, dest: str = "en") -> str:
    """Translate text between languages"""
    try:
        if src:
            return _TRANSLATOR.translate(text, src=src, dest=dest).text
        return _TRANSLATOR.translate(text, dest=dest).text
    except:
        return text

def generate_medical_response(prompt: str, tokenizer, model, device, max_new_tokens: int = 400) -> str:
    """Generate medical response"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if len(response.split()) < 25:
        follow_up_prompt = f"{response}\n\nPlease elaborate with causes, symptoms, diagnosis, and treatments."
        inputs = tokenizer(follow_up_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response
```

---

## **3️⃣ fine_tune.py**

```python
# =======================================================
#  Medical QA Fine-Tuning Script (Stable GPU Version C)
# =======================================================

# 1️⃣ Install required packages
# !pip install -q transformers datasets accelerate sentencepiece

# 2️⃣ Imports
import os, torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)

# Disable tokenizer multiprocessing warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 3️⃣ Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Device in use:", device)

# =======================================================
# 4️⃣ Load datasets
# =======================================================
datasets_list = []

print("⏳ Loading datasets...")
medquad = load_dataset("lavita/MedQuAD", split="train")
datasets_list.append(medquad)

pubmedqa = load_dataset("pubmed_qa", "pqa_labeled", split="train")
datasets_list.append(pubmedqa)

hc_magic = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
datasets_list.append(hc_magic)
print("✅ Datasets loaded successfully.")

# =======================================================
# 5️⃣ Standardize columns
# =======================================================
def standardize_columns_safe(example):
    instr, resp = None, None
    if 'question' in example and 'answer' in example:
        instr, resp = example["question"], example["answer"]
    elif 'question' in example and 'final_decision' in example:
        instr, resp = example["question"], example["final_decision"]
    elif 'instruction' in example and 'output' in example:
        instr, resp = example["instruction"], example["output"]
    return {"instruction": str(instr) if instr else None, 
            "response": str(resp) if resp else None}

combined_dataset = concatenate_datasets(datasets_list)
combined_dataset = combined_dataset.map(standardize_columns_safe)
combined_dataset = combined_dataset.filter(lambda x: x["instruction"] and x["response"])

# Remove duplicates
seen = set()
def remove_duplicates(example):
    if example["instruction"] in seen:
        return False
    seen.add(example["instruction"])
    return True
combined_dataset = combined_dataset.filter(remove_duplicates)

# Shuffle & subset
combined_dataset = combined_dataset.shuffle(seed=42)
max_examples = min(8000, len(combined_dataset)) 
combined_dataset = combined_dataset.select(range(max_examples))
print(f"✅ Dataset ready with {len(combined_dataset)} samples")
print("Sample:", combined_dataset[0])

# =======================================================
# 6️⃣ Load model & tokenizer
# =======================================================
model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()

# =======================================================
# 7️⃣ Preprocessing
# =======================================================
max_input_length = 256 
max_target_length = 192 

def preprocess_function(examples):
    inputs = [str(x) for x in examples["instruction"]]
    targets = [str(x) for x in examples["response"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = combined_dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format(type="torch")
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
print("✅ Tokenization complete")

# =======================================================
# 8️⃣ Training arguments (Stable settings)
# =======================================================
training_args = Seq2SeqTrainingArguments(
    output_dir="./medical_flan_t5",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    predict_with_generate=True,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=2000,
    logging_strategy="steps",
    logging_steps=500,
    num_train_epochs=2,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=False,                 # ⚠️ Disable FP16 → prevents NaN loss
    bf16=False,                 # Disable mixed precision for safety
    push_to_hub=False,
    report_to=[],
    max_grad_norm=1.0,
    dataloader_num_workers=2,
)

# =======================================================
# 9️⃣ Trainer
# =======================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# =======================================================
# 🔟 Train the model
# =======================================================
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete.")

# =======================================================
# 11️⃣ Save final model
# =======================================================
final_model_dir = "./medical_flan_t5_final"
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"✅ Model saved at {final_model_dir}")  
```

---

## **4️⃣ requirements.txt**

```
streamlit
transformers
torch
googletrans==4.0.0-rc1
datasets
sentencepiece
huggingface_hub
regex
```

---

## **5️⃣ .gitignore**

```gitignore
# Virtual environment
venv/
.env

# Python cache
__pycache__/
*.pyc

# IDE files
.vscode/
.idea/
*.DS_Store

# Logs
*.log

# Hugging Face cache
~/.cache/huggingface/

# Jupyter notebook checkpoints
.ipynb_checkpoints/
```

---

## 🌍 Real-Life Use Cases
* **Patient Support:** Automates answers to health questions.
* **Public Health Campaigns:** Delivers vaccination, hygiene, and disease prevention info in local languages.
* **Hospital Assistance:** Guides patients for appointments, departments, and FAQs.
* **Symptom Pre-Screening:** Collects preliminary symptoms before doctor consultation.
* **Medical Education:** Explains medical terms and prescriptions in simple language.
* **Remote Healthcare Assistance:** Provides guidance where medical resources are scarce.

---

## 🌟 Future Enhancements
* Integrate **commercial translation APIs** for higher reliability (Azure Translator, AWS Translate).
* Implement **context memory** for multi-turn conversations.
* Add **voice input/output** for accessibility.
* Incorporate **more medical datasets** and expand multilingual support.
* Deploy with **scalable backend** using Hugging Face Spaces or AWS Lambda.
* Add **analytics dashboard** to track usage, queries, and model performance.
* Implement **user feedback loop** to improve responses over time.

---

## 📊 Evaluation Metrics
* **Response Accuracy** – correctness of medical guidance.
* **Language Detection Accuracy** – detects user language correctly.
* **Translation Fidelity** – preserves meaning during language conversion.
* **Scalability & Performance** – handles multiple users and large queries.
---

## ⚙️ Tech Stack

| Category        | Tools & Libraries                               |
| --------------- | ----------------------------------------------- |
| **Programming** | Python                                          |
| **Libraries**   | Transformers, Torch, Pandas, Regex, Googletrans |
| **Frontend**    | Streamlit                                       |
| **Deployment**  | Streamlit Cloud / Hugging Face Spaces           |

---

## 📁 Dataset Used

* MedQuAD: [https://huggingface.co/datasets/lavita/MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD)
* PubMedQA: [https://huggingface.co/datasets/qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)
* HealthCareMagic QA: [https://huggingface.co/datasets/HealthCareMagic](https://huggingface.co/datasets/HealthCareMagic)

---

## 🚀 How to Run the Project

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py
```

---

## ⚠️ Disclaimer

This chatbot is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---


