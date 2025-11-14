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
if submit_clicked:
    if not query.strip():
        st.error("Please enter a query first.")
    else:
        cleaned = clean_text(query)

        # 🩺 Step: Filter non-medical queries
        if not is_medical_query(cleaned):
            st.warning("⚕️ Please ask a health or medically related question.")
        else:
            with st.spinner("💬 Generating medical response... Please wait"):
                detected_lang = detect_language(cleaned)

                text_en = (
                    translate_text(cleaned, src=detected_lang, dest="en")
                    if detected_lang != "en"
                    else cleaned
                )

                prompt = (
                    "You are a helpful and professional medical assistant. "
                    "Provide a short, clear, and accurate answer to the user's question. "
                    "If relevant, briefly mention key symptoms and treatments in 4 sentences. "
                    "Focus only on health-related questions. "
                    "If the query is not medical, respond politely asking for a medical question. "
                    "Do not give emergency advice; recommend consulting a doctor if needed.\n"
                    "Do not introduce yourself; go straight to the answer.\n"
                    f"Question: {text_en}\nAnswer:"
                )

            response_en = generate_medical_response(
                prompt, tokenizer, model, device, max_new_tokens=180
            )

            final_response = (
                translate_text(response_en, src="en", dest=detected_lang)
                if detected_lang != "en"
                else response_en
            )

        st.session_state["chat_history"].append({
            "user": query,
            "bot": final_response
        })

        st.info("## 🧠 Smart Response")
        st.write(final_response)

        if detected_lang != "en":
            st.divider()
            st.info("## English Response")
            st.write(response_en)

# Disclaimer at bottom
st.markdown("""
<p style="
    font-size:15px; 
    color:black; 
    text-align:center; 
    margin: 30px 40px 20px 40px;
">
⚠️ <strong>Disclaimer:</strong> This chatbot is for educational and informational purposes only.<br> It is not a substitute for 
professional diagnosis.
</p>
""", unsafe_allow_html=True)

# venv\Scripts\activate
# ==========================================================
# Run Command (for reference)
# ==========================================================
# python -m streamlit run app.py

