import streamlit as st
import requests
import json
import sqlite3
import google.generativeai as genai
from PIL import Image
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="🩺 Med-GemMA Safety",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
# THE BRAIN: Gemma 2 via OpenRouter 
GEMMA_MODEL_ID = "google/gemma-2-9b-it" 

DATA_REPO_ID = "FassikaF/medical-safety-app-data" 
DB_FILENAME = "ddi_database.db"

# --- CSS / UI Styling ---
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
    if local_path.exists():
        return str(local_path)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return str(local_path)
    except Exception as e:
        return None

db_path = download_file_from_hf(DATA_REPO_ID, DB_FILENAME)

def query_openrouter(model, messages, temperature=0.1):
    """
    Used ONLY for Gemma 2 (Text/Reasoning)
    """
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("🚨 OpenRouter API Key Missing.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://med-gemma-safety.streamlit.app/",
        "X-Title": "Med-GemMA Safety"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1000
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Gemma API Error: {e}")
        return None

# --- NEW: Google Native Vision Logic ---
def get_visual_description_native(image, audience):
    """
    Uses Google's Native API (google-generativeai) for Vision.
    No OpenRouter dependency for this part.
    """
    google_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_key:
        st.error("🚨 GOOGLE_API_KEY missing in secrets. Please add it to use Vision features.")
        return None

    try:
        # Configure the library
        genai.configure(api_key=google_key)
        
        # Use the latest Flash model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        tone = "clinical and precise" if audience == "Clinician" else "simple and descriptive"
        prompt = f"Describe the medical symptom in this image in {tone} terms. Focus on visible dermatological or physical signs. Be concise."
        
        # Call Google directly
        response = model.generate_content([prompt, image])
        return response.text
        
    except Exception as e:
        st.error(f"Google Vision API Error: {e}")
        return None

# --- Logic Modules ---

def extract_entities(text):
    """Uses Gemma to extract clinical entities."""
    if not text.strip(): return []
    prompt = f"""
    Extract medical entities from the text: "{text}".
    Return a JSON object with two keys: "drugs" (list) and "symptoms" (list).
    Example: {{"drugs": ["Advil"], "symptoms": ["headache"]}}
    """
    messages = [{"role": "user", "content": prompt}]
    res = query_openrouter(GEMMA_MODEL_ID, messages, temperature=0.0)
    try:
        if res:
            res = res.replace("```json", "").replace("```", "").strip()
            data = json.loads(res)
            return data.get("drugs", []), data.get("symptoms", [])
        return [], []
    except:
        return [text], []

def query_ddi_db(drug1, drug2):
    if not db_path: return "Unknown"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    q = "SELECT level FROM ddi_interactions WHERE (LOWER(drug1)=? AND LOWER(drug2)=?) OR (LOWER(drug1)=? AND LOWER(drug2)=?)"
    c.execute(q, (drug1.lower(), drug2.lower(), drug2.lower(), drug1.lower()))
    res = c.fetchone()
    conn.close()
    return res[0] if res else "Unknown"

def analyze_symptom_causality(drugs, symptoms, visual_context, audience, language):
    """
    Uses Gemma 2 to reason about the relationship between drugs and symptoms.
    """
    
    visual_note = ""
    if visual_context:
        visual_note = f"**Visual Analysis Findings (from Gemini Flash):** {visual_context}"

    # Define Role-Based Instructions
    if audience == "Patient":
        role_desc = "Empathetic Medical Assistant"
        style_guide = "Use simple language (5th-grade level). Avoid jargon. Focus on 'What should I do?' and clear warning signs."
    else:
        role_desc = "Clinical Pharmacologist"
        style_guide = "Use professional medical terminology. Discuss Pharmacokinetics (PK), Pharmacodynamics (PD), differential diagnosis, and clinical management strategies."

    prompt = f"""
    **Role:** {role_desc} (Powered by Gemma 2).
    **Language:** {language}
    
    **Scenario:**
    - Current Drugs: {', '.join(drugs)}
    - Reported Symptoms: {', '.join(symptoms)}
    {visual_note}
    
    **Task:** 
    Analyze if the reported symptoms (or visual signs) are an adverse drug reaction (ADR), allergy, or emergency.
    
    **Style Instructions:** {style_guide}
    
    **Output Logic:**
    1. If signs suggest Stevens-Johnson Syndrome, Anaphylaxis, or severe toxicity -> RETURN "EMERGENCY".
    2. If signs are common side effects -> RETURN "MONITOR".
    3. If unknown/concerning -> RETURN "WARNING".
    
    **Required Format:**
    - **Triage:** [EMERGENCY / WARNING / MONITOR]
    - **Assessment:** (Explanation based on audience settings)
    - **Recommendation:** (Action plan based on audience settings)
    """
    
    messages = [
        {"role": "system", "content": f"You are a helpful {role_desc}."},
        {"role": "user", "content": prompt}
    ]
    return query_openrouter(GEMMA_MODEL_ID, messages, temperature=0.2)

def analyze_interaction_report(drug1, drug2, level, language, audience):
    """
    Generates interaction report tailored to audience.
    """
    if audience == "Patient":
        style = "simple, non-medical language. Explain what might feel wrong and when to call a doctor."
    else:
        style = "clinical language. Include mechanism of action (e.g., CYP450 inhibition), clinical significance, and monitoring parameters."

    prompt = f"""
    Create a drug interaction safety report for **{drug1}** and **{drug2}**. 
    
    **Data:**
    - Database Risk Level: {level}
    - Target Audience: {audience} ({style})
    - Output Language: {language}
    
    **Structure:**
    1. Summary
    2. Detailed Explanation (Mechanism or What to feel)
    3. Action Plan (Management or Next Steps)
    """
    messages = [{"role": "user", "content": prompt}]
    return query_openrouter(GEMMA_MODEL_ID, messages, temperature=0.2)

# --- Main App UI ---

with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.markdown("### Settings")
    
    target_audience = st.radio("Target Audience", ["Patient", "Clinician"], index=0)
    language = st.selectbox("Language", ["English", "Spanish", "French", "Arabic", "Amharic"])
    
    st.markdown("---")
    st.caption(f"Brain: **{GEMMA_MODEL_ID}**")
    st.caption("Eyes: **Google Gemini Flash 1.5** (Native)")

st.markdown('<div class="main-header">🩺 Med-GemMA Safety Hub</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💊 Interaction Checker", "📸 Visual Symptom Analyzer"])

# --- TAB 1: INTERACTION CHECKER ---
with tab1:
    st.markdown(f"#### Check drug combinations ({target_audience} Mode)")
    col1, col2 = st.columns(2)
    with col1: d1_input = st.text_input("First Medication", placeholder="e.g. Warfarin")
    with col2: d2_input = st.text_input("Second Medication", placeholder="e.g. Aspirin")
        
    if st.button("Check Interactions", key="btn_interact"):
        if d1_input and d2_input:
            with st.spinner("Consulting Gemma 2..."):
                db_level = query_ddi_db(d1_input, d2_input)
                report = analyze_interaction_report(d1_input, d2_input, db_level, language, target_audience)
                
                color = "green"
                if "major" in db_level.lower(): color = "red"
                elif "moderate" in db_level.lower(): color = "orange"
                
                st.markdown(f"**Risk Level:** :{color}[{db_level.upper()}]")
                st.success(report)

# --- TAB 2: VISUAL SYMPTOM ANALYZER ---
with tab2:
    st.markdown(f"#### 🛡️ Visual Side Effect Triage ({target_audience} Mode)")
    st.markdown("Upload a photo of your symptom (e.g., rash, swelling) and list your meds.")
    
    col_input, col_img = st.columns([1, 1])
    
    with col_input:
        st.subheader("1. What are you taking?")
        txt_drugs = st.text_area("Medication List", height=100, placeholder="e.g. Lamotrigine, Penicillin")
        st.subheader("2. Describe feeling (Optional)")
        txt_feel = st.text_area("Description", height=100, placeholder="e.g. Itchy, burning sensation")
    
    with col_img:
        st.subheader("3. Upload Photo (Optional)")
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if img_file:
            st.image(img_file, caption="Uploaded Symptom", width=250)

    if st.button("Analyze Safety", key="btn_visual"):
        if not txt_drugs:
            st.warning("Please enter your medications.")
        else:
            with st.status("Running Multimodal Analysis...", expanded=True) as status:
                
                # Step 1: Text Extraction (Gemma)
                st.write("🧠 Gemma: Extracting medication names...")
                drugs_list, _ = extract_entities(txt_drugs)
                
                # Step 2: Visual Processing (Google Native)
                visual_context = "No image provided."
                
                if img_file:
                    st.write("👁️ Gemini Flash (Native): Analyzing image patterns...")
                    try:
                        # Convert uploaded file to PIL Image for Google SDK
                        pil_image = Image.open(img_file)
                        v_desc = get_visual_description_native(pil_image, target_audience)
                        if v_desc:
                            visual_context = v_desc
                            st.info(f"Visual Findings: {visual_context}")
                        else:
                            st.warning("Vision analysis returned no data.")
                    except Exception as e:
                        st.error(f"Image processing failed: {e}")
                
                # Step 3: Synthesis (Gemma)
                st.write("⚕️ Gemma: Synthesizing clinical assessment...")
                symptoms_combined = [txt_feel] if txt_feel else []
                if not drugs_list: drugs_list = [txt_drugs]
                
                analysis = analyze_symptom_causality(drugs_list, symptoms_combined, visual_context, target_audience, language)
                
                status.update(label="Analysis Complete", state="complete")
                
                # Step 4: Display
                st.markdown("---")
                if analysis:
                    if "EMERGENCY" in analysis:
                        st.markdown(f'<div class="status-red"><h3>🚨 POSSIBLE EMERGENCY</h3>{analysis}</div>', unsafe_allow_html=True)
                    elif "WARNING" in analysis:
                         st.markdown(f'<div class="status-yellow"><h3>⚠️ WARNING - CONSULT DOCTOR</h3>{analysis}</div>', unsafe_allow_html=True)
                    else:
                         st.markdown(f'<div class="status-green"><h3>✅ MONITOR</h3>{analysis}</div>', unsafe_allow_html=True)
