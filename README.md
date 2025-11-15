
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
# 🩺 Multilingual Medical Support Chatbot with Custom UI(app.py)
# ==========================================================
import streamlit as st
import os
from pathlib import Path
import base64
from utils import (
    load_model_and_tokenizer,
    clean_text,
    detect_language,
    translate_text,
    generate_medical_response,
    is_medical_query
)

# ==========================================================
# 1️⃣ Page Config
# ==========================================================
st.set_page_config(page_title="🩺 Multilingual Medical Assistant", layout="centered")

# Safe image path resolution
image_path = Path(__file__).parent / "hospital.jpg"

# Convert to base64
if image_path.exists():
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
else:
    encoded_image = ""

# Inject CSS with the background image
st.markdown(f"""
    <style>
        [class*="stAppViewContainer"] {{
            background: linear-gradient(rgba(210, 235, 250, 0.85), rgba(210, 235, 250, 0.85)), 
                        url(data:image/jpg;base64,{encoded_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        body, [class*="stMain"] {{
            color: #2C3E50 !important;
        }}
        .title {{
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #1A5276;
            margin-top: 10px;
        }}
        .subtitle {{
            text-align: center;
            font-size: 16px;
            color: #7F8C8D;
            margin-bottom: 25px;
        }}
        .chat-box {{
            background-color: rgba(210, 235, 250, 0.85);
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.05);
            padding: 25px;
            margin: 20px auto;
            width: 80%;
        }}
        .stTextArea textarea {{
            border-radius: 25px !important;
            padding: 10px 20px !important;
            font-size: 15px;
            border: 1px solid #D0D3D4 !important;
        }}
        .stButton button {{
            background-color: #3498DB !important;
            color: white !important;
            border-radius: 25px !important;
            padding: 8px 20px !important;
            font-size: 14px !important;
            border: none !important;
        }}
        .stButton button:hover {{
            background-color: #2E86C1 !important;
        }}
    </style>
""", unsafe_allow_html=True)



# ==========================================================
# 🏥 Header
# ==========================================================
st.markdown('<div class="title">MULTILINGUAL MEDICAL ASSISTANT</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">How can I help you ?</div>', unsafe_allow_html=True)

# ==========================================================
# 2️⃣ Load Model & Tokenizer
# ==========================================================
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer, model, device = load_model_and_tokenizer()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ==========================================================
# 3️⃣ Initialize chat history
# ==========================================================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ==========================================================
# 4️⃣ Function to clear text & chat history
# ==========================================================
def clear_text():
    st.session_state["query"] = ""
    st.session_state["chat_history"] = []

# ==========================================================
# 💬 Chat Section 
# ==========================================================

with st.container():
    st.markdown(
        """
        <div class="chat-box" style="text-align: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/387/387561.png" width="70" style="border-radius: 50%; margin-bottom: 10px;">
            <p style="font-weight: 600; color: #2C3E50; font-size: 16px;">Hello! I'm here to assist you.</p>
            <p style="color: #7F8C8D; font-size: 15px; margin-bottom: 15px;">Please type your query below 👇</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    for entry in st.session_state["chat_history"]:
        user_msg = entry.get("user")
        bot_msg = entry.get("bot")
        
        if user_msg:
            st.markdown(
        f"""
        <div style="
            display: flex; justify-content: flex-end; margin: 10px 0;
        ">
            <div style="
                background-color: rgba(52, 152, 219, 0.15); 
                color: #2C3E50; 
                padding: 12px 18px; 
                border-radius: 20px 20px 0px 20px; 
                max-width: 90%;
                word-wrap: break-word;
                font-size: 15px;
            ">
                👤 You: {user_msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

        if bot_msg:
            st.markdown(
        f"""
        <div style="
            display: flex; justify-content: flex-start; margin: 10px 0;
        ">
            <div style="
                background-color: rgba(46, 204, 113, 0.15); 
                color: #2C3E50; 
                padding: 12px 18px; 
                border-radius: 20px 20px 20px 0px; 
                max-width: 90%;
                word-wrap: break-word;
                font-size: 15px;
            ">
                🩺 Assistant: {bot_msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# 6️⃣ User Input
# ==========================================================
query = st.text_area(
    " ",
    value=st.session_state.get("query", ""),
    placeholder="Enter your health-related query here...",
    key="query"
)


# ==========================================================
# Action Buttons
# ==========================================================

st.markdown("""
    <style>
        /* Flex container for buttons */
        .button-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 80%;
            margin: 20px auto;
        }
        /* Common button styling */
        .stButton > button {
            border-radius: 25px !important;
            padding: 10px 25px !important;
            font-size: 15px !important;
            border: none !important;
            cursor: pointer !important;
        }
        /* Submit button (left) */
        .submit-btn button {
            background-color: #3498DB !important;
            color: white !important;
        }
        .submit-btn button:hover {
            background-color: #2E86C1 !important;
        }
        /* Clear button (right) */
        .clear-btn button {
            background-color: #E74C3C !important;
            color: white !important;
        }
        .clear-btn button:hover {
            background-color: #C0392B !important;
        }
    </style>
""", unsafe_allow_html=True)

# Create the two buttons in the same visual row
st.markdown('<div class="button-row">', unsafe_allow_html=True)

col_submit, col_clear = st.columns([1, 1], gap="large")
with col_submit:
    submit_clicked = st.button("🩺 Get Response", key="submit", use_container_width=True)
with col_clear:
    clear_clicked = st.button("Clear", key="clear", use_container_width=True, on_click=clear_text)

st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
#  Chatbot Logic 
# ==========================================================
detected_lang = "en"
response_en = ""
final_response = "🫀 Please ask a health or medically related question."

if submit_clicked:
    if not query.strip():
        st.error("Please enter a query first.")
    else:
        cleaned = clean_text(query)
        detected_lang = detect_language(cleaned)

        # 🔹 Step 1: Translate non-English input to English for keyword detection
        if detected_lang != "en":
            cleaned_en = translate_text(cleaned, src=detected_lang, dest="en")
        else:
            cleaned_en = cleaned

        # 🔹 Step 2: Check if translated text is medical
        if not is_medical_query(cleaned_en):
            final_response = "🫀 Please ask a health or medically related question."
        else:
            try:
                # 🔹 Step 3: Prepare the medical prompt
                prompt = (
                    "You are a helpful and professional medical assistant. "
                    "Provide a short, clear, and accurate answer to the user's question. "
                    "If relevant, briefly mention key symptoms and treatments in 3 sentences. "
                    "Focus only on health-related questions. "
                    "If the query is not medical, respond politely asking for a medical question. "
                    "Do not give emergency advice; recommend consulting a doctor if needed.\n"
                    "Do not introduce yourself; go straight to the answer.\n"
                    f"Question: {cleaned_en}\nAnswer:"
                )

                # 🔹 Step 4: Generate response with model
                with st.spinner("💬 Generating medical response... Please wait"):
                    response_en = generate_medical_response(
                        prompt, tokenizer, model, device, max_new_tokens=180
                    )

                # 🔹 Step 5: Translate response back to original language if needed
                final_response = (
                    translate_text(response_en, src="en", dest=detected_lang)
                    if detected_lang != "en"
                    else response_en
                )

            except Exception as e:
                final_response = "Sorry, I cannot provide a response now. Please consult a medical professional."

        # 🔹 Step 6: Save chat history and display
        st.session_state["chat_history"].append({
            "user": query,
            "bot": final_response
        })

        st.info("### 🧠 Smart Response")
        st.write(final_response)

        # 🔹 Step 7: Show English response for reference
        if detected_lang != "en":
            st.divider()
            st.info("### English Response")
            st.write(response_en)

# ⚠️ Disclaimer
st.markdown("""
<p style="font-size:15px; color:black; text-align:center; margin: 30px 40px 20px 40px;">
⚠️ <strong>Disclaimer:</strong> This chatbot is for educational and informational purposes only.<br> 
It is not a substitute for professional diagnosis.</p>""",
unsafe_allow_html=True)


# venv\Scripts\activate
# ==========================================================
# Run Command (for reference)
# ==========================================================
# python -m streamlit run app.py
```

---

## **2️⃣ utils.py**

```python
# ==========================================================
# utils.py — Helper Functions for Multilingual Medical Chatbot
# ==========================================================
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,MarianMTModel, MarianTokenizer
import torch
from googletrans import Translator

_TRANSLATOR = Translator()

# Tuned Model
# MODEL_NAME = "./medical_flan_t5_final_8"
MODEL_NAME = "./medical_flan_t5_final"

# ==========================================================
# 🩺 Medical Vocabulary 
# ==========================================================
def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load model and tokenizer from Hugging Face."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model, device

def is_medical_query(text: str) -> bool:
    """Check if the query contains at least one medical-related word."""
    words = set(re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split())
    return len(MEDICAL_KEYWORDS.intersection(words)) > 0

def clean_text(text: str) -> str:
    """Basic cleaning for text normalization."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def detect_language(text: str) -> str:
    """Detect input language using Google Translate."""
    try:
        return _TRANSLATOR.detect(text).lang
    except Exception:
        return "en"

# Load Marian models dynamically (only when needed)
_MARIAN_MODELS = {}

def load_marian_model(src_lang: str, dest_lang: str):
    """Load MarianMT model for the given language pair (with caching)."""
    key = f"{src_lang}-{dest_lang}"
    if key in _MARIAN_MODELS:
        return _MARIAN_MODELS[key]

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _MARIAN_MODELS[key] = (tokenizer, model)
    except Exception as e:
        print(f"[ERROR] Failed to load Marian model '{model_name}': {e}")
        _MARIAN_MODELS[key] = None
    return _MARIAN_MODELS[key]


def marian_translate(text: str, src: str, dest: str = "en") -> str:
    """Offline translation using MarianMT."""
    pair = load_marian_model(src, dest)
    if pair is None:
        return text  # if model not found, fallback to raw text
    tokenizer, model = pair
    batch = tokenizer([text], return_tensors="pt", padding=True)
    gen = model.generate(**batch, max_new_tokens=200)
    return tokenizer.decode(gen[0], skip_special_tokens=True)

def translate_text(text: str, src: str = None, dest: str = "en") -> str:
    """Translate text using Google Translate first, then fallback to MarianMT if it fails."""
    try:
        # Primary: Google Translate
        if src:
            return _TRANSLATOR.translate(text, src=src, dest=dest).text
        return _TRANSLATOR.translate(text, dest=dest).text
    except Exception as e:
        print(f"[WARN] Google Translate failed ({e}). Switching to MarianMT fallback...")

    # Fallback: MarianMT (offline)
    try:
        src_lang = src or detect_language(text)  # auto-detect source if missing
        src_lang = src_lang.split("-")[0]  # normalize (e.g., 'en-US' -> 'en')
        dest_lang = dest.split("-")[0]
        return marian_translate(text, src_lang, dest_lang)
    except Exception as e:
        print(f"[ERROR] MarianMT translation failed: {e}")
        return f"[Translation unavailable] {text}"


def generate_medical_response(prompt: str, tokenizer, model, device, max_new_tokens: int = 180) -> str:
    """Generate a detailed medical response with safe fallback and query-length check."""
    
    # Step 0: Check if user query is too short
    try:
        user_query = prompt.split("Question:")[1].split("Answer:")[0].strip()
        if len(user_query.split()) < 2:
            return "Please enter a more detailed medical question for better results."
        elif len(prompt.split()) < 2:
            return "Please enter a more detailed medical question for better results."

    except:
        pass

    try:
        # Step 1: Initial generation
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

        # Step 2: If response is too short, try elaborating
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

    except Exception as e:
        response = "Sorry, I cannot provide a response now. Please consult a medical professional."

    return response.strip() 


```

---

## **3️⃣ fine_tune.py**

```python
# =======================================================
#  Medical QA Fine-Tuning Script
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


