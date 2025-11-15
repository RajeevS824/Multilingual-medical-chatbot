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