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
    buffered = io.BytesIO()
    if pil_image.mode == 'RGBA': pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- 1. OpenRouter (Universal Gateway) ---
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
            timeout=55 
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"Model {model} failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception on {model}: {e}")
        return None

# --- 2. Google Native Vision (Primary Eyes) ---
def get_visual_description_native(image, audience):
    """
    Try Google Native first. 
    UPDATED: Iterates through the newest available models (2.0/2.5/Latest).
    """
    google_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_key: return None

    try:
        genai.configure(api_key=google_key)
        
        # Priority list: Try 3.0/2.5 preview first, then fall back to stable 2.0/Latest
        model_candidates = [
            "gemini-2.0-flash-exp",   # Cutting edge
            "gemini-2.0-flash",       # Stable 2.0
            "gemini-flash-latest",    # Always points to newest stable
            "gemini-1.5-flash-latest" # Old faithful fallback
        ]
        
        prompt = "Analyze this medical image. Describe ONLY the physical appearance (morphology, color, texture). DO NOT diagnose. Be precise."

        # Loop through candidates until one works
        for model_name in model_candidates:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, image])
                if response.text:
                    return response.text
            except Exception:
                continue # Try next model
        
        return None
    except Exception:
        return None

# --- 3. Translation / Localization Layer ---
def localize_content(text, target_language):
    """
    Translates content using OpenRouter.
    Uses Gemini 2.0 Flash or Llama 3.3 for best Amharic support.
    """
    if target_language == "English":
        return text

    prompt = f"""
    You are a professional medical translator.
    Translate the following text into {target_language}.
    
    RULES:
    1. Keep drug names in English (Latin script) like "Warfarin".
    2. Use natural, conversational {target_language}.
    3. Keep the formatting (bolding, headers).
    
    TEXT:
    {text}
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    # Updated to use the newer 2.0 models
    translation = query_openrouter("google/gemini-2.0-flash-exp:free", messages, 0.1)
    
    if not translation:
        # Fallback to Llama 3.3 (Excellent multilingual)
        translation = query_openrouter("meta-llama/llama-3.3-70b-instruct", messages, 0.1)
        
    return translation if translation else text

# --- 4. Hybrid Vision Strategy ---
def get_visual_description_hybrid(pil_image, audience, drugs_list):
    # 1. Try Google Native
    desc = get_visual_description_native(pil_image, audience)
    if desc:
        return desc, "Google Native (Gemini Flash 2.x)"
    
    # 2. Try OpenRouter Fallbacks
    b64_img = image_to_base64(pil_image)
    
    prompt = f"""
    You are a medical imaging assistant.
    The patient is taking: {', '.join(drugs_list)}.
    
    TASK: Provide a purely DESCRIPTIVE analysis of the visual symptoms.
    RESTRICTIONS:
    1. Do NOT diagnose the condition.
    2. Do NOT mention specific drugs or causes.
    3. Describe ONLY visible features: Color, Texture, Distribution.
    4. Ignore clothing and background.
    """
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }
    ]
    
    # UPDATED FALLBACK LIST (Newest Models)
    fallback_models = [
        "google/gemini-2.0-flash-exp:free",        # 1. Gemini 2.0 (Fast & Free on OR)
        "google/gemini-2.0-flash-001",             # 2. Gemini 2.0 Stable
        "rhymes-ai/ovis-1.6-gemma-2-9b",           # 3. OVIS (Gemma-based!)
        "qwen/qwen2.5-vl-72b-instruct",            # 4. Qwen 2.5 (SOTA Open Source)
        "meta-llama/llama-3.2-11b-vision-instruct" # 5. Llama 3.2 (Reliable)
    ]
    
    for model in fallback_models:
        st.write(f"🔄 Trying model: `{model}`...") 
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
    # STRICTER PERSONA DEFINITION
    if audience == "Patient":
        role = "Caring Triage Nurse"
        tone_instruction = """
        CRITICAL INSTRUCTION FOR PATIENT MODE:
        1. Speak in simple, everyday language (5th-grade reading level).
        2. DO NOT use words like "Erythema Multiforme", "Stevens-Johnson Syndrome", or "Pharmacovigilance".
        3. Instead of "Erythema", say "Redness". Instead of "Discontinue", say "Stop taking".
        4. Focus ONLY on: Is it dangerous? What should I do right now?
        5. Be empathetic but direct.
        """
    else:
        role = "Clinical Pharmacologist"
        tone_instruction = """
        Use professional medical terminology. Discuss Differential Diagnosis, Pharmacokinetics, and Clinical Management guidelines.
        """

    prompt = f"""
    Role: {role}.
    Drugs: {drugs}. Symptoms Reported: {symptoms}.
    
    **Visual Observation (Technician Report):** 
    "{visual_context}"
    
    **Task:**
    Review the visual report and symptoms.
    {tone_instruction}
    
    **Output Structure:**
    1. **Triage Status:** (EMERGENCY / WARNING / MONITOR)
    2. **What is happening?:** (Simple explanation for patient / Technical for doctor)
    3. **Next Steps:** (Actionable advice)
    """
    messages = [{"role": "system", "content": f"You are a {role}."}, {"role": "user", "content": prompt}]
    
    # 1. Generate English Reasoning
    english_analysis = query_openrouter(GEMMA_MODEL_ID, messages, 0.2)
    
    # 2. Localize via OpenRouter (Bypasses Google Native Block)
    if language != "English" and english_analysis:
        return localize_content(english_analysis, language)
    
    return english_analysis

def analyze_interaction_report(d1, d2, level, lang, audience):
    if audience == "Patient":
        prompt = f"Explain interaction between {d1} and {d2} (Level: {level}) to a patient. Simple language. Safety focus."
    else:
        prompt = f"Create clinical interaction report for {d1} & {d2}. Level: {level}. Include mechanism."
        
    messages = [{"role": "user", "content": prompt}]
    english_report = query_openrouter(GEMMA_MODEL_ID, messages, 0.2)
    
    if lang != "English" and english_report:
        return localize_content(english_report, lang)
    
    return english_report

# --- UI ---
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.markdown("### Settings")
    target_audience = st.radio("Audience", ["Patient", "Clinician"])
    language = st.selectbox("Language", ["English", "Amharic", "Spanish", "French", "Arabic"])
    st.caption(f"Reasoning: **{GEMMA_MODEL_ID}**")

st.markdown('<div class="main-header">🩺 Med-GemMA Safety Hub</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["💊 Interaction Checker", "📸 Visual Symptom Analyzer"])

with tab1:
    col1, col2 = st.columns(2)
    with col1: d1 = st.text_input("First Medication", placeholder="e.g. Warfarin")
    with col2: d2 = st.text_input("Second Medication", placeholder="e.g. Aspirin")
    if st.button("Check", key="btn1"):
        if d1 and d2:
            with st.spinner(f"Analyzing..."):
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
                drugs, symps = extract_entities(txt_drugs)
                if txt_feel: symps.append(txt_feel)
                
                v_ctx = "No image."
                if img_file:
                    st.write("👁️ Analyzing Image (Hybrid Mode)...")
                    img = Image.open(img_file)
                    v_desc, source = get_visual_description_hybrid(img, target_audience, drugs)
                    
                    if v_desc:
                        v_ctx = v_desc
                        st.info(f"Visual Findings ({source}): {v_ctx}")
                    else:
                        st.error("Vision analysis failed on all providers.")
                
                st.write(f"⚕️ Clinical Synthesis & Localization ({language})...")
                if not drugs: drugs = [txt_drugs]
                
                ans = analyze_symptom_causality(drugs, symps, v_ctx, target_audience, language)
                status.update(label="Done", state="complete")
                
                st.markdown("---")
                if ans:
                    color = "status-green"
                    if "EMERGENCY" in ans: color = "status-red"
                    elif "WARNING" in ans: color = "status-yellow"
                    st.markdown(f'<div class="{color}">{ans}</div>', unsafe_allow_html=True)
