
# 🩺 Multilingual Medical Support Chatbot
## 📌 Project Overview 
**Multilingual Medical Support Chatbot** is an **AI-driven healthcare assistant** that understands user queries in any language, provides medically accurate responses using a **fine-tuned Flan-T5 model**, and translates answers back to the user’s language via a **Streamlit interface**. The goal is to **improve healthcare accessibility** by providing multilingual support, symptom guidance, and basic health awareness content.
---

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
# fine_tune.py
# ==========================================================
# Fine-tuning script for medical QA datasets
# ==========================================================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import argparse

def preprocess_examples(batch, tokenizer, max_input_length=512, max_target_length=256):
    inputs = [q + "\n\nAnswer:" for q in batch["question"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=max_input_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["answer"], truncation=True, padding="max_length", max_length=max_target_length)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--dataset", default="lavita/MedQuAD")  
    parser.add_argument("--output_dir", default="./medical-flan-t5")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized = dataset.map(lambda x: preprocess_examples(x, tokenizer), batched=True, remove_columns=dataset["train"].column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", None),
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
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


