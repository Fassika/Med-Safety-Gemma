import streamlit as st
import requests
import json
import sqlite3
import google.generativeai as genai
from PIL import Image
from pathlib import Path
import io
import base64
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="🩺 Med-GemMA Safety",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
GEMMA_MODEL_ID = "google/gemma-2-9b-it" 
DATA_REPO_ID = "FassikaF/medical-safety-app-data" 
DB_FILENAME = "ddi_database.db"

# --- CSS ---
st.markdown("""
    <style>
    .main-header {font-size: 2rem; color: #4285F4; font-weight: 700;} 
    .status-red {background-color: #ffebee; border-left: 5px solid #d32f2f; padding: 15px; border-radius: 5px; color: #b71c1c;}
    .status-yellow {background-color: #fff3e0; border-left: 5px solid #f57c00; padding: 15px; border-radius: 5px; color: #e65100;}
    .status-green {background-color: #e8f5e9; border-left: 5px solid #388e3c; padding: 15px; border-radius: 5px; color: #1b5e20;}
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def download_file_from_hf(repo_id: str, filename: str, dest_path: str = "."):
    local_path = Path(dest_path) / filename
    if local_path.exists(): return str(local_path)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return str(local_path)
    except: return None

db_path = download_file_from_hf(DATA_REPO_ID, DB_FILENAME)

def image_to_base64(pil_image):
    """Convert PIL image to base64 for OpenRouter fallback"""
    buffered = io.BytesIO()
    # Convert RGBA to RGB if necessary
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- 1. OpenRouter (The Brain & Backup Eyes) ---
def query_openrouter(model, messages, temperature=0.1):
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://med-gemma-safety.streamlit.app/",
        "X-Title": "Med-GemMA Safety"
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=45 # Increased timeout for vision models
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            # Print error to console for debugging
            print(f"Model {model} failed with status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception on {model}: {e}")
        return None

# --- 2. Google Native Vision (Primary Eyes) ---
def get_visual_description_native(image, audience):
    """Try Google Native first. Return None if it fails."""
    google_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_key: return None

    try:
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel('gemini-1.5-flash') 
        tone = "clinical" if audience == "Clinician" else "simple"
        prompt = f"Describe the medical symptom in this image in {tone} terms. Focus on visible dermatological or physical signs."
        response = model.generate_content([prompt, image])
        return response.text
    except Exception:
        return None

# --- 3. Hybrid Strategy Wrapper (Updated with New Models) ---
def get_visual_description_hybrid(pil_image, audience):
    # 1. Try Google Native (Best Points)
    desc = get_visual_description_native(pil_image, audience)
    if desc:
        return desc, "Google Native (Gemini Flash)"
    
    # 2. Try OpenRouter Fallbacks
    b64_img = image_to_base64(pil_image)
    tone = "clinical" if audience == "Clinician" else "simple"
    prompt = f"Describe the medical symptom in this image in {tone} terms. Focus on visible dermatological or physical signs."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }
    ]
    
    # The List of Models to Try (In order of reliability/preference)
    fallback_models = [
        "meta-llama/llama-3.2-11b-vision-instruct", # Best fallback (Fast, Open)
        "qwen/qwen2.5-vl-32b-instruct",            # Very Strong Vision
        "google/gemini-flash-1.5",                 # Google via OpenRouter
        "moonshotai/kimi-vl-a3b-thinking",         # Alternative
        "google/gemma-3-4b-it"                     # As requested (if available)
    ]
    
    for model in fallback_models:
        st.write(f"🔄 Trying model: `{model}`...") # Visual feedback for user
        desc = query_openrouter(model, messages)
        if desc:
            return desc, f"OpenRouter ({model})"
            
    return None, "All Providers Failed"

# --- Logic Modules ---
def extract_entities(text):
    if not text.strip(): return []
    prompt = f"""Extract "drugs" (list) and "symptoms" (list) from: "{text}". Return JSON."""
    messages = [{"role": "user", "content": prompt}]
    res = query_openrouter(GEMMA_MODEL_ID, messages, 0.0)
    try:
        data = json.loads(res.replace("```json", "").replace("```", "").strip())
        return data.get("drugs", []), data.get("symptoms", [])
    except: return [text], []

def query_ddi_db(d1, d2):
    if not db_path: return "Unknown"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT level FROM ddi_interactions WHERE (LOWER(drug1)=? AND LOWER(drug2)=?) OR (LOWER(drug1)=? AND LOWER(drug2)=?)", (d1.lower(), d2.lower(), d2.lower(), d1.lower()))
    res = c.fetchone()
    conn.close()
    return res[0] if res else "Unknown"

def analyze_symptom_causality(drugs, symptoms, visual_context, audience, language):
    role = "Empathetic Assistant" if audience == "Patient" else "Clinical Pharmacologist"
    prompt = f"""
    Role: {role}. Language: {language}.
    Drugs: {drugs}. Symptoms: {symptoms}.
    Visual Findings: {visual_context}
    
    Analyze if this is an Adverse Drug Reaction, Allergy, or Emergency.
    Return: Triage (EMERGENCY/WARNING/MONITOR), Assessment, Recommendation.
    """
    messages = [{"role": "system", "content": f"You are a {role}."}, {"role": "user", "content": prompt}]
    return query_openrouter(GEMMA_MODEL_ID, messages, 0.2)

def analyze_interaction_report(d1, d2, level, lang, audience):
    prompt = f"Create interaction report for {d1} & {d2}. Level: {level}. Audience: {audience}. Lang: {lang}."
    messages = [{"role": "user", "content": prompt}]
    return query_openrouter(GEMMA_MODEL_ID, messages, 0.2)

# --- UI ---
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.markdown("### Settings")
    target_audience = st.radio("Audience", ["Patient", "Clinician"])
    language = st.selectbox("Language", ["English", "Spanish", "French", "Arabic", "Amharic"])
    st.caption(f"Reasoning: **{GEMMA_MODEL_ID}**")

st.markdown('<div class="main-header">🩺 Med-GemMA Safety Hub</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["💊 Interaction Checker", "📸 Visual Symptom Analyzer"])

with tab1:
    col1, col2 = st.columns(2)
    with col1: d1 = st.text_input("First Medication", placeholder="e.g. Warfarin")
    with col2: d2 = st.text_input("Second Medication", placeholder="e.g. Aspirin")
    if st.button("Check", key="btn1"):
        if d1 and d2:
            with st.spinner("Analyzing..."):
                lvl = query_ddi_db(d1, d2)
                rep = analyze_interaction_report(d1, d2, lvl, language, target_audience)
                color = "red" if "major" in lvl.lower() else "green"
                st.markdown(f"**Risk:** :{color}[{lvl.upper()}]")
                st.success(rep)

with tab2:
    c1, c2 = st.columns([1,1])
    with c1:
        txt_drugs = st.text_area("Meds", placeholder="e.g. Penicillin")
        txt_feel = st.text_area("Feeling", placeholder="e.g. Rash")
    with c2:
        img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if img_file: st.image(img_file, width=200)

    if st.button("Analyze Safety", key="btn2"):
        if not txt_drugs: st.warning("Enter meds.")
        else:
            with st.status("Processing...", expanded=True) as status:
                st.write("🧠 Extracting entities...")
                drugs, _ = extract_entities(txt_drugs)
                
                v_ctx = "No image."
                if img_file:
                    st.write("👁️ Analyzing Image (Hybrid Mode)...")
                    img = Image.open(img_file)
                    v_desc, source = get_visual_description_hybrid(img, target_audience)
                    if v_desc:
                        v_ctx = v_desc
                        st.info(f"Visual Findings ({source}): {v_ctx}")
                    else:
                        st.error("Vision analysis failed on all providers.")
                
                st.write("⚕️ Clinical Synthesis...")
                ans = analyze_symptom_causality(drugs, [txt_feel] if txt_feel else [], v_ctx, target_audience, language)
                status.update(label="Done", state="complete")
                
                st.markdown("---")
                if ans:
                    color = "status-green"
                    if "EMERGENCY" in ans: color = "status-red"
                    elif "WARNING" in ans: color = "status-yellow"
                    st.markdown(f'<div class="{color}">{ans}</div>', unsafe_allow_html=True)
