import streamlit as st
import requests
import json
import os
import sqlite3
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="🩺 Med-GemMA Safety Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration & Constants ---
# CRITICAL: We are using Google's Gemma 2 model to meet competition criteria.
# We use the 9B parameter model for a balance of speed and medical reasoning.
GEMMA_MODEL_ID = "google/gemma-2-9b-it" 
DATA_REPO_ID = "FassikaF/medical-safety-app-data" 
DB_FILENAME = "ddi_database.db"

# --- CSS for UI Polish ---
st.markdown("""
    <style>
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .main-header {font-size: 2.5rem; color: #4285F4; font-weight: 700;} 
    .sub-header {font-size: 1.5rem; color: #333;}
    .risk-high {color: #d32f2f; font-weight: bold;}
    .risk-mod {color: #f57c00; font-weight: bold;}
    .risk-low {color: #388e3c; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def download_file_from_hf(repo_id: str, filename: str, dest_path: str = "."):
    """Downloads database from Hugging Face if not present."""
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
        st.error(f"Failed to download {filename}: {e}")
        return None

db_path = download_file_from_hf(DATA_REPO_ID, DB_FILENAME)

def query_openrouter_gemma(messages, temperature=0.1):
    """
    Wrapper for OpenRouter API specifically targeting Google Gemma 2.
    """
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("🚨 API Key Missing. Please set OPENROUTER_API_KEY in secrets.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://med-gemma-safety.streamlit.app/",
        "X-Title": "Med-GemMA Safety"
    }
    
    payload = {
        "model": GEMMA_MODEL_ID, # COMPETITION REQUIREMENT: Using Gemma
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

# --- Core Logic: Gemma-Powered Extraction & Analysis ---

def extract_medications_with_gemma(text):
    """
    Uses Gemma 2 to extract medication names. 
    Replaces the heavy 'd4data' NER pipeline for better performance and reasoning.
    """
    if not text.strip():
        return []
        
    prompt = f"""
    Analyze the following text and extract all pharmaceutical drug names, brand names, or active ingredients.
    Return ONLY a valid JSON list of strings. Do not add markdown formatting or explanation.
    
    Text: "{text}"
    """
    
    messages = [
        {"role": "system", "content": "You are a precise medical entity extractor. Output JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    response = query_openrouter_gemma(messages, temperature=0.0)
    
    try:
        # Clean potential markdown code blocks if Gemma adds them
        cleaned = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except:
        # Fallback if JSON fails
        return [w.strip() for w in response.split(',')]

def query_ddi_db(drug1, drug2):
    """Queries the local SQLite database for known interactions."""
    if not db_path: return None
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Check both permutations
    q = "SELECT level FROM ddi_interactions WHERE (LOWER(drug1)=? AND LOWER(drug2)=?) OR (LOWER(drug1)=? AND LOWER(drug2)=?)"
    c.execute(q, (drug1.lower(), drug2.lower(), drug2.lower(), drug1.lower()))
    res = c.fetchone()
    conn.close()
    return res[0] if res else "Unknown"

def generate_gemma_analysis(drug1, drug2, db_level, target_audience, language):
    """
    Generates the final report using Gemma 2, adapting tone based on audience.
    """
    
    role_instruction = ""
    if target_audience == "Patient":
        role_instruction = "Explain this in simple, non-medical language (5th-grade reading level). Focus on 'What should I do?' and warning signs."
    else:
        role_instruction = "Provide a clinical pharmacological consult suitable for a physician. Include mechanism of action and monitoring parameters."

    prompt = f"""
    Analyze the interaction between **{drug1}** and **{drug2}**.
    
    **Database Flag:** The interaction database lists the severity as: "{db_level}".
    
    **Instructions:**
    1. {role_instruction}
    2. Provide the response in {language}.
    3. Structure the response clearly (Summary, Risks, Action Steps).
    4. If the database level is 'Unknown', rely on your internal medical knowledge to assess the risk, but clearly state that this is AI-generated advice.
    """
    
    messages = [
        {"role": "system", "content": "You are an expert medical safety assistant powered by Google Gemma 2. You prioritize patient safety and evidence-based answers."},
        {"role": "user", "content": prompt}
    ]
    
    return query_openrouter_gemma(messages, temperature=0.2)

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.header("Settings")
    target_audience = st.radio("Target Audience", ["Clinician", "Patient"], help="Adjusts the complexity of the explanation.")
    language = st.selectbox("Output Language", ["English", "Spanish", "French", "Amharic", "Arabic"])
    st.markdown("---")
    st.caption(f"Powered by **{GEMMA_MODEL_ID}**")
    st.caption("Participating in Google Med-GemMA Impact Challenge")

# --- Main Interface ---
st.markdown('<div class="main-header">🩺 Med-GemMA Safety Assistant</div>', unsafe_allow_html=True)
st.markdown("### AI-Powered Drug Interaction Checker")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Current Medication Profile")
    current_text = st.text_area("List current drugs", height=100, placeholder="e.g. Warfarin, Lisinopril...")
with col2:
    st.subheader("New Prescription / Addition")
    new_text = st.text_area("Drug to add", height=100, placeholder="e.g. Aspirin, Ibuprofen...")

if st.button("🚀 Analyze Interaction", type="primary", use_container_width=True):
    if not current_text or not new_text:
        st.warning("Please enter medications in both fields.")
    else:
        with st.status("Processing with Gemma 2...", expanded=True) as status:
            # Step 1: NER with Gemma
            st.write("🧠 Extracting entities using Gemma 2...")
            current_drugs = extract_medications_with_gemma(current_text)
            new_drugs = extract_medications_with_gemma(new_text)
            
            if not current_drugs or not new_drugs:
                status.update(label="Extraction failed", state="error")
                st.error("Could not identify specific drug names. Please check spelling.")
            else:
                st.write(f"✅ Identified: {current_drugs} + {new_drugs}")
                
                # Step 2: Analysis Loop
                st.write("🔎 Querying Database & Generating Reports...")
                status.update(label="Analysis Complete", state="complete")
                
                st.markdown("---")
                
                # Cross-check logic
                found_interactions = False
                for d1 in current_drugs:
                    for d2 in new_drugs:
                        db_level = query_ddi_db(d1, d2)
                        
                        # Visual cues for severity
                        color_class = "risk-low"
                        if "major" in db_level.lower(): color_class = "risk-high"
                        elif "moderate" in db_level.lower(): color_class = "risk-mod"
                        
                        st.markdown(f"#### Interaction: {d1.title()} ↔ {d2.title()}")
                        st.markdown(f"**Database Severity:** <span class='{color_class}'>{db_level.upper()}</span>", unsafe_allow_html=True)
                        
                        # Step 3: Generate Explanation
                        analysis = generate_gemma_analysis(d1, d2, db_level, target_audience, language)
                        st.info(analysis)
                        found_interactions = True
                        st.markdown("---")

                if not found_interactions:
                    st.success("No interactions found between the identified drugs.")