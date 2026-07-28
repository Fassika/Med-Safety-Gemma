import streamlit as st
import requests
import json
import sqlite3
from PIL import Image
from pathlib import Path
import io
import base64
import itertools  

# --- New Official Google GenAI SDK ---
try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

# --- Page Configuration ---
st.set_page_config(
    page_title="🩺 Med-GemMA Safety",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Models & Database Configuration ---
PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.0-flash"
OPENROUTER_FALLBACK_MODEL = "google/gemma-2-9b-it"

DATA_REPO_ID = "FassikaF/medical-safety-app-data" 
DB_FILENAME = "ddi_database.db"

# --- CSS Styling ---
st.markdown("""
    <style>
    .main-header {font-size: 2rem; color: #4285F4; font-weight: 700;} 
    .status-red {background-color: #ffebee; border-left: 5px solid #d32f2f; padding: 15px; border-radius: 5px; color: #b71c1c; margin-top: 10px;}
    .status-yellow {background-color: #fff3e0; border-left: 5px solid #f57c00; padding: 15px; border-radius: 5px; color: #e65100; margin-top: 10px;}
    .status-green {background-color: #e8f5e9; border-left: 5px solid #388e3c; padding: 15px; border-radius: 5px; color: #1b5e20; margin-top: 10px;}
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
    except Exception as e:
        st.error(f"Failed to load database file: {e}")
        return None

db_path = download_file_from_hf(DATA_REPO_ID, DB_FILENAME)

def get_gemini_client():
    google_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_key or not HAS_GOOGLE_GENAI:
        return None
    return genai.Client(api_key=google_key)

# --- Unified Gemini Generation Logic ---
def query_gemini_native(prompt, system_instruction=None, image=None, json_mode=False, temperature=0.1):
    client = get_gemini_client()
    if not client:
        return None

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction
    )
    if json_mode:
        config.response_mime_type = "application/json"

    contents = []
    if image:
        contents.append(image)
    contents.append(prompt)

    # Try Primary Model then Fallback
    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            st.toast(f"Gemini model `{model_name}` notice: {e}")
            continue

    return None

# --- OpenRouter Fallback Logic ---
def query_openrouter(model, messages, temperature=0.1):
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://med-safety-gemma.streamlit.app/",
        "X-Title": "Med-GemMA Safety Hub"
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=40
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.toast(f"OpenRouter Fallback Error: {e}")
    return None

# --- 1. Entity Extraction ---
def extract_entities(text):
    if not text.strip(): return [], []
    
    prompt = f"""Extract medication names into 'drugs' array and patient symptoms into 'symptoms' array from: "{text}". Return JSON with keys "drugs" and "symptoms"."""
    
    # Try Native Gemini Structured Output first
    res = query_gemini_native(prompt, json_mode=True, temperature=0.0)
    
    # Fallback to OpenRouter if needed
    if not res:
        messages = [{"role": "user", "content": prompt}]
        res = query_openrouter(OPENROUTER_FALLBACK_MODEL, messages, 0.0)

    try:
        if res:
            res_clean = res.replace("```json", "").replace("```", "").strip()
            data = json.loads(res_clean)
            return data.get("drugs", []), data.get("symptoms", [])
    except Exception:
        pass
    
    return [text], []

# --- 2. Database Lookup ---
def query_ddi_db(d1, d2):
    if not db_path: return "Unknown"
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT level FROM ddi_interactions WHERE (LOWER(drug1)=? AND LOWER(drug2)=?) OR (LOWER(drug1)=? AND LOWER(drug2)=?)", (d1.lower(), d2.lower(), d2.lower(), d1.lower()))
        res = c.fetchone()
        conn.close()
        return res[0] if res else "Unknown"
    except Exception:
        return "Unknown"

# --- 3. Translation / Localization Layer ---
def localize_content(text, target_language):
    if target_language == "English" or not text: 
        return text

    prompt = f"""You are a professional medical translator.
Translate the following medical safety text into {target_language}.
RULES: Keep drug names in English. Use natural {target_language}. Keep markdown formatting.
TEXT:
{text}"""

    res = query_gemini_native(prompt, temperature=0.1)
    if not res:
        messages = [{"role": "user", "content": prompt}]
        res = query_openrouter("meta-llama/llama-3.3-70b-instruct", messages, 0.1)
        
    return res if res else text

# --- 4. Hybrid Vision Strategy ---
def get_visual_description_hybrid(pil_image, audience, drugs_list):
    prompt = f"""Analyze this clinical image for visual symptoms. Patient medications: {', '.join(drugs_list)}.
Describe ONLY the physical appearance (Morphology, Color, Distribution, Texture). DO NOT diagnose. Be objective and precise."""

    # 1. Direct Native Gemini Call
    desc = query_gemini_native(prompt, image=pil_image, temperature=0.1)
    if desc: 
        return desc, "Google Native (Gemini 2.5 Flash)"

    # 2. OpenRouter Multimodal Fallback
    buffered = io.BytesIO()
    if pil_image.mode == 'RGBA': pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format="JPEG")
    b64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
    ]}]
    
    fallback_models = [
        "google/gemini-2.0-flash-001",
        "qwen/qwen2.5-vl-72b-instruct",
        "meta-llama/llama-3.2-11b-vision-instruct"
    ]
    
    for model in fallback_models:
        st.write(f"🔄 Fallback Vision Attempt: `{model}`...") 
        desc = query_openrouter(model, messages)
        if desc: return desc, f"OpenRouter ({model})"
            
    return "No distinct visual lesions detected or analysis failed.", "Fallback Provider Failed"

# --- 5. Multi-Drug Interaction Analysis ---
def analyze_multidrug_interactions(drug_list, language, audience):
    pairs = list(itertools.combinations(drug_list, 2))
    findings = []
    for d1, d2 in pairs:
        level = query_ddi_db(d1, d2)
        findings.append(f"- {d1} + {d2}: {level}")
    
    findings_str = "\n".join(findings)
    
    if audience == "Patient":
        prompt = f"""You are a Medical Safety Assistant.
Medications being taken: {', '.join(drug_list)}.
Database Interactions Found:
{findings_str}

Explain these interactions in simple language. Outline safety precautions and potential side effects to monitor. Do not refuse to answer."""
    else:
        prompt = f"""You are a Clinical Pharmacologist.
Medication Regimen: {', '.join(drug_list)}
Database Interactions:
{findings_str}

Provide a clinical pharmacodynamics and pharmacokinetics (CYP450 pathways, clearance, QT risks) assessment. Provide recommendations."""

    report = query_gemini_native(prompt, temperature=0.2)
    if not report:
        messages = [{"role": "user", "content": prompt}]
        report = query_openrouter(OPENROUTER_FALLBACK_MODEL, messages, 0.2)

    return localize_content(report, language) if report else "Unable to generate interaction report."

# --- 6. Symptom Causality Analysis ---
def analyze_symptom_causality(drugs, symptoms, visual_context, audience, language):
    if audience == "Patient":
        role = "Caring Triage Nurse"
        tone = "Speak in plain, reassuring, everyday language (5th-grade level)."
    else:
        role = "Clinical Pharmacologist"
        tone = "Use precise medical terminology, differential diagnosis, and mechanism-of-action reasoning."

    system_instruction = f"You are an expert {role}."
    
    prompt = f"""{tone}

**Drugs Reported:** {', '.join(drugs)}
**Symptoms Reported:** {', '.join(symptoms)}
**Visual Description:** "{visual_context}"

Provide a structured evaluation with:
1. **Triage Status:** (Must be strictly one of: EMERGENCY / WARNING / MONITOR)
2. **Clinical Evaluation:** Direct evaluation of drug-induced adverse events or rash causality.
3. **Recommended Next Steps:** Practical precautions and action items for the user."""

    # 1. Primary Native Call
    report = query_gemini_native(prompt, system_instruction=system_instruction, temperature=0.2)
    
    # 2. Fallback Call
    if not report:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        report = query_openrouter(OPENROUTER_FALLBACK_MODEL, messages, 0.2)

    if report:
        return localize_content(report, language)
    return None

# --- UI Interface ---
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=40)
    st.markdown("### Settings")
    target_audience = st.radio("Audience Mode", ["Patient", "Clinician"])
    language = st.selectbox("Language", ["English", "Amharic", "Spanish", "French", "Arabic"])
    st.caption(f"Engine: **{PRIMARY_MODEL}** (Native)")

st.markdown('<div class="main-header">🩺 Med-GemMA Safety Hub</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["💊 Interaction Checker", "📸 Visual Symptom Analyzer"])

# --- TAB 1: POLYPHARMACY CHECKER ---
with tab1:
    st.markdown(f"#### Check Polypharmacy Regimens ({target_audience} Mode)")
    drug_input = st.text_area("Enter Medications (comma separated)", placeholder="e.g. Azithromycin, Chloroquine, Ibuprofen", height=80)
    
    if st.button("Check Interactions", key="btn1"):
        if drug_input:
            d_list = [d.strip() for d in drug_input.split(",") if d.strip()]
            if len(d_list) < 2:
                st.warning("Please enter at least two medications to evaluate interactions.")
            else:
                with st.spinner("Analyzing pharmacological interactions..."):
                    report = analyze_multidrug_interactions(d_list, language, target_audience)
                    st.markdown("### Safety Report")
                    st.info(report)

# --- TAB 2: VISUAL SYMPTOM ANALYZER ---
with tab2:
    c1, c2 = st.columns([1,1])
    with c1:
        txt_drugs = st.text_area("Meds Taken", placeholder="e.g. Penicillin")
        txt_feel = st.text_area("Reported Feeling/Symptoms", placeholder="e.g. Headache, itchy skin rash")
    with c2:
        img_file = st.file_uploader("Upload Symptom Image", type=["jpg","png","jpeg"])
        if img_file: 
            st.image(img_file, width=220)

    if st.button("Analyze Safety", key="btn2"):
        if not txt_drugs and not txt_feel: 
            st.warning("Please enter medications or symptoms.")
        else:
            with st.status("Processing Medical Safety Evaluation...", expanded=True) as status:
                st.write("🧠 Extracting entities...")
                drugs, symps = extract_entities(txt_drugs)
                if txt_feel: 
                    symps.append(txt_feel)
                
                v_ctx = "No visual image uploaded."
                if img_file:
                    st.write("👁️ Analyzing Visual Symptom Morphology...")
                    img = Image.open(img_file)
                    v_desc, source = get_visual_description_hybrid(img, target_audience, drugs)
                    if v_desc:
                        v_ctx = v_desc
                        st.info(f"**Visual Findings ({source}):**\n{v_ctx}")
                    else:
                        st.error("Vision analysis could not process the image.")
                
                st.write(f"⚕️ Synthesizing Clinical Assessment ({language})...")
                if not drugs: drugs = [txt_drugs]
                
                ans = analyze_symptom_causality(drugs, symps, v_ctx, target_audience, language)
                status.update(label="Analysis Completed", state="complete")
                
                st.markdown("---")
                if ans:
                    color = "status-green"
                    if "EMERGENCY" in ans: color = "status-red"
                    elif "WARNING" in ans: color = "status-yellow"
                    st.markdown(f'<div class="{color}">{ans}</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to generate clinical analysis. Please verify your API keys in Streamlit Secrets (`GOOGLE_API_KEY` or `OPENROUTER_API_KEY`).")
