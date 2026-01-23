import streamlit as st
import requests
import json
import sqlite3
import base64
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="🩺 Med-GemMA Safety",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
# THE BRAIN: Performs the safety reasoning
GEMMA_MODEL_ID = "google/gemma-2-9b-it" 
# THE EYE: Describes the image (Using Gemini Flash via OpenRouter for speed/vision)
VISION_MODEL_ID = "google/gemini-flash-1.5-8b"

DATA_REPO_ID = "FassikaF/medical-safety-app-data" 
DB_FILENAME = "ddi_database.db"

# --- CSS / UI Styling ---
st.markdown("""
    <style>
    .main-header {font-size: 2rem; color: #4285F4; font-weight: 700;} 
    .status-red {background-color: #ffebee; border-left: 5px solid #d32f2f; padding: 15px; border-radius: 5px; color: #b71c1c;}
    .status-yellow {background-color: #fff3e0; border-left: 5px solid #f57c00; padding: 15px; border-radius: 5px; color: #e65100;}
    .status-green {background-color: #e8f5e9; border-left: 5px solid #388e3c; padding: 15px; border-radius: 5px; color: #1b5e20;}
    .vision-box {border: 1px dashed #4285F4; padding: 10px; border-radius: 10px; background-color: #f8f9fa;}
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

def encode_image(uploaded_file):
    """Encodes streamlit uploaded file to base64 string."""
    if uploaded_file is None:
        return None
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def query_openrouter(model, messages, temperature=0.1):
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("🚨 API Key Missing.")
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
        st.error(f"API Error ({model}): {e}")
        return None

# --- Logic Modules ---

def get_visual_description(base64_image):
    """
    Uses a Vision Model (Gemini Flash) to translate the image into text.
    This text is then passed to Gemma 2 for reasoning.
    """
    prompt = "Describe the medical symptom in this image in clinical terms (e.g., 'erythematous rash', 'swollen edema'). Be precise but concise."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    # We use Gemini Flash as the 'Eye' because it's fast and in the Google ecosystem
    return query_openrouter(VISION_MODEL_ID, messages, temperature=0.1)

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
        res = res.replace("```json", "").replace("```", "").strip()
        data = json.loads(res)
        return data.get("drugs", []), data.get("symptoms", [])
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

def analyze_symptom_causality(drugs, symptoms, visual_context=None):
    """
    Uses Gemma 2 to reason about the relationship between drugs and symptoms.
    """
    
    visual_note = ""
    if visual_context:
        visual_note = f"**Visual Analysis of Symptom:** {visual_context}"

    prompt = f"""
    **Role:** Clinical Safety AI (Gemma 2).
    
    **Scenario:**
    - Current Drugs: {', '.join(drugs)}
    - Reported Symptoms: {', '.join(symptoms)}
    {visual_note}
    
    **Task:** 
    Determine if the reported symptoms (or visual signs) are a known side effect of the drugs, an allergic reaction, or a medical emergency.
    
    **Output Logic:**
    1. If signs suggest Stevens-Johnson Syndrome, Anaphylaxis, or severe toxicity -> RETURN "EMERGENCY".
    2. If signs are common, manageable side effects -> RETURN "MONITOR".
    3. If unknown or concerning -> RETURN "WARNING".
    
    **Format:**
    Provide a structured response:
    - **Triage:** [EMERGENCY / WARNING / MONITOR]
    - **Assessment:** Explain *why*. Link the visual signs to the drug mechanism if possible.
    - **Recommendation:** Actionable advice.
    """
    
    messages = [
        {"role": "system", "content": "You are a safe, evidence-based medical assistant powered by Google Gemma 2."},
        {"role": "user", "content": prompt}
    ]
    return query_openrouter(GEMMA_MODEL_ID, messages, temperature=0.2)

def analyze_interaction_report(drug1, drug2, level, language):
    prompt = f"Create a safety report for {drug1} and {drug2}. Level: {level}. Lang: {language}. Audience: Patient."
    messages = [{"role": "user", "content": prompt}]
    return query_openrouter(GEMMA_MODEL_ID, messages, temperature=0.2)

# --- Main App UI ---

with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.markdown("### Settings")
    language = st.selectbox("Language", ["English", "Spanish", "French", "Arabic"])
    st.caption(f"Reasoning: **{GEMMA_MODEL_ID}**")
    st.caption(f"Vision: **{VISION_MODEL_ID}**")

st.markdown('<div class="main-header">🩺 Med-GemMA Safety Hub</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💊 Interaction Checker", "📸 Visual Symptom Analyzer"])

# --- TAB 1: INTERACTION CHECKER ---
with tab1:
    st.markdown("#### Check drug combinations.")
    col1, col2 = st.columns(2)
    with col1: d1_input = st.text_input("First Medication", placeholder="e.g. Warfarin")
    with col2: d2_input = st.text_input("Second Medication", placeholder="e.g. Aspirin")
        
    if st.button("Check Interactions", key="btn_interact"):
        if d1_input and d2_input:
            with st.spinner("Consulting Gemma 2..."):
                db_level = query_ddi_db(d1_input, d2_input)
                report = analyze_interaction_report(d1_input, d2_input, db_level, language)
                color = "green"
                if "major" in db_level.lower(): color = "red"
                elif "moderate" in db_level.lower(): color = "orange"
                st.markdown(f"**Risk Level:** :{color}[{db_level.upper()}]")
                st.success(report)

# --- TAB 2: VISUAL SYMPTOM ANALYZER ---
with tab2:
    st.markdown("#### 🛡️ Visual Side Effect Triage")
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
                
                # Step 2: Visual Processing (Gemini Flash)
                visual_context = "No image provided."
                if img_file:
                    st.write("👁️ Vision Model: Analyzing image patterns...")
                    b64_img = encode_image(img_file)
                    visual_context = get_visual_description(b64_img)
                    st.info(f"Visual Findings: {visual_context}")
                
                # Step 3: Synthesis (Gemma)
                st.write("⚕️ Gemma: Synthesizing clinical assessment...")
                symptoms_combined = [txt_feel] if txt_feel else []
                analysis = analyze_symptom_causality(drugs_list, symptoms_combined, visual_context)
                
                status.update(label="Analysis Complete", state="complete")
                
                # Step 4: Display
                st.markdown("---")
                if "EMERGENCY" in analysis:
                    st.markdown(f'<div class="status-red"><h3>🚨 POSSIBLE EMERGENCY</h3>{analysis}</div>', unsafe_allow_html=True)
                elif "WARNING" in analysis:
                     st.markdown(f'<div class="status-yellow"><h3>⚠️ WARNING - CONSULT DOCTOR</h3>{analysis}</div>', unsafe_allow_html=True)
                else:
                     st.markdown(f'<div class="status-green"><h3>✅ MONITOR</h3>{analysis}</div>', unsafe_allow_html=True)
