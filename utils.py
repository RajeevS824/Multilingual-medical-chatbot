# ==========================================================
# utils.py — Helper Functions for Multilingual Medical Chatbot
# ==========================================================
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,MarianMTModel, MarianTokenizer
import torch
from googletrans import Translator

_TRANSLATOR = Translator()

# Tuned Model
MODEL_NAME = "./medical_flan_t5_final_8"
# ==========================================================
# 🩺 Medical Vocabulary 
# ==========================================================
MEDICAL_KEYWORDS = set("""health illness disease diabetes disorder infection injury condition symptom treatment diagnosis therapy prevention recovery examination patient doctor nurse hospital clinic medicine prescription dosage vaccination injection consultation appointment ward emergency operation surgeon specialist referral healthcare monitoring vital signs stethoscope thermometer pulse blood pressure oxygen respiration heartbeat temperature pain rest nutrition hydration hygiene checkup admission discharge laboratory report test x-ray MRI scan ultrasound ECG CT biopsy result chart record insurance consent procedure guideline protocol ICU OPD follow-up telemedicine pharmacist anesthesia sterilization mask gloves gown syringe drip infusion saline stretcher wheelchair sanitizer bandage wound dressing first aid rehabilitation care plan fever cough cold sore throat headache fatigue nausea vomiting dizziness weakness pain swelling rash chills diarrhea constipation breathlessness wheezing loss appetite insomnia anxiety stress depression confusion sweating shaking trembling blurred vision itching bleeding runny nose chest pain back pain cramps numbness tingling fainting restlessness palpitations sneezing sore eyes yellowing burning sensation indigestion bloating dry mouth thirst hunger dehydration stiffness cough phlegm sleepiness ringing ears irritability slow heartbeat rapid heartbeat swollen legs black stools ulcers abdomen tightness hallucinations forgetfulness tremor glands speech weakness limbs excessive sweating itchy skin surgery operation chemotherapy radiation physiotherapy dialysis antibiotic antiviral analgesic sedative anesthesia suturing transfusion oxygen therapy nebulization rehabilitation counseling psychotherapy speech therapy insulin therapy IV fluids saline infusion ointment inhaler nebulizer ventilator catheterization endoscopy biopsy x-ray CT MRI ultrasound stent pacemaker detoxification laser therapy radiology donation physiotherapy immobilization compression tooth extraction cataract removal skin graft orthopedic surgery laparoscopy electrotherapy herbal medicine acupuncture homeopathy ayurveda yoga therapy bacteria virus fungi parasite genetic hereditary environmental pollution allergy inflammation deficiency trauma injury stress hormonal imbalance poor nutrition dehydration poor hygiene obesity smoking alcohol toxins mutation immune system autoimmune antigen antibody enzyme metabolic disorder cancer tumor clot blockage ischemia necrosis contamination bacterial growth viral load PCR test blood count urinalysis liver kidney hormone lipid profile allergy test ECG echo biopsy cytology pathology radiology microbiology virology hematology toxicology diagnostic report observation interpretation diabetes hypertension asthma tuberculosis pneumonia malaria dengue typhoid influenza COVID-19 cancer heart disease stroke kidney failure liver disease hepatitis arthritis anemia thyroid disorder obesity depression anxiety schizophrenia migraine epilepsy ulcer gastritis appendicitis bronchitis sinusitis tonsillitis eczema psoriasis acne dermatitis cholera measles mumps chickenpox rabies tetanus polio HIV AIDS lupus sclerosis Parkinson Alzheimer glaucoma cataract leukemia lymphoma hepatitis cirrhosis gallstones kidney stones urinary infection COPD meningitisencephalitis gout fracture burns sepsis jaundice viral fever food poisoning malnutrition heart attack cardiac arrest angina tumorparalysis COVID influenza""".split())

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

