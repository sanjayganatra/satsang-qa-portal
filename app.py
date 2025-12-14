import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import time
import os

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Gemini (same library style as your original code)
import google.generativeai as genai


import base64

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# ============================================================
# 1) PAGE CONFIG
# ============================================================
PAGE_TITLE = "Welcome to PriyaKunj"
ICON = "🙏"

# CRITICAL: Initialize default language to Hindi IMMEDIATELY
if "view_lang" not in st.session_state:
    st.session_state["view_lang"] = "Hindi"
st.set_page_config(page_title=PAGE_TITLE, page_icon=ICON)

# --- CSS ---
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Yatra+One&display=swap');
    
    /* =========================================================================
       HYBRID THEME: Apple Glass + Saffron (11 Dec Original Colors)
       ========================================================================= */
       
    /* BACKGROUND: Saffron Gradient (Original) */
    .stApp {
        background: linear-gradient(180deg, #FFF5E1 0%, #FFECB3 100%);
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
    }
    .block-container { padding-top: 1.5rem; }
    
    /* HEADER STYLES: Maroon + Yatra One (Original) */
    h1, h2, h3 {
        color: #8B0000 !important;
        font-family: 'Yatra One', cursive, sans-serif !important;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Remove Default Streamlit Header */
    [data-testid="stHeader"] {
        background: transparent !important;
        color: #8B0000 !important;
    }
    
    /* FIX: Force Sidebar Toggle Button Visibility */
    /* Give it a visible background and white icon for maximum contrast */
    [data-testid="collapsedControl"], 
    button[kind="header"],
    button[kind="headerNoPadding"] {
        background: #8B0000 !important;
        border-radius: 6px !important;
        padding: 6px !important;
        margin: 8px !important;
    }
    
    [data-testid="collapsedControl"] svg, 
    button[kind="header"] svg,
    button[kind="headerNoPadding"] svg,
    [data-testid="stSidebarNav"] svg {
        fill: white !important;
        color: white !important;
        stroke: white !important;
    }

    /* CARD STYLING: Glassmorphism + Saffron Accent */
    .answer-card, .stCard, div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.70) !important; /* Glassy White */
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-left: 5px solid #FF9933 !important; /* Saffron Accent */
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* FIX: Comment Section Styling */
    div[data-testid="stExpander"] details summary {
        background: linear-gradient(135deg, rgba(255, 153, 51, 0.2), rgba(255, 128, 0, 0.2)) !important;
        color: #8B0000 !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    
    /* Text Area in Comments */
    textarea {
        background-color: white !important;
        color: #4E342E !important;
        border: 2px solid #FF9933 !important;
        border-radius: 12px !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    textarea::placeholder {
        color: #999 !important;
    }
    
    textarea:focus {
        border-color: #8B0000 !important;
        box-shadow: 0 0 0 2px rgba(139, 0, 0, 0.1) !important;
        outline: none !important;
    }
    
    .answer-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(139, 69, 19, 0.15); /* Warm shadow */
    }

    /* BUTTONS: Glassy Saffron Capsule */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(255, 153, 51, 0.9), rgba(255, 128, 0, 0.9)) !important;
        backdrop-filter: blur(5px);
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 99px !important; /* Capsule */
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(255, 128, 0, 0.3) !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(255, 128, 0, 0.5) !important;
        background: linear-gradient(135deg, #FF9933, #FF8000) !important;
    }
    
    /* INPUTS & FILTERS */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.6) !important;
        border-radius: 12px !important;
        color: #333 !important;
    }
    
    /* Fix Input Label Contrast */
    .stTextInput label {
        color: #8B0000 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Fix input field background and text */
    div[data-baseweb="input"] {
        background-color: white !important;
        border: 2px solid #FF9933 !important;
        border-radius: 12px !important;
    }
    
    /* Force input text to be dark */
    input.st-bd {
        color: #4E342E !important;
        background-color: white !important;
    }
    
    /* Make placeholder text visible */
    input::placeholder {
        color: #999 !important;
        opacity: 1 !important;
    }
    
    /* Focus state */
    div[data-baseweb="input"]:focus-within {
        border-color: #8B0000 !important;
        box-shadow: 0 0 0 2px rgba(139, 0, 0, 0.1) !important;
    }
    
    /* Auto-height for buttons to handle wrapping text if needed */
    div.stButton > button {
        height: auto !important;
        min-height: 2.5rem !important;
        white-space: normal !important;
    }
    
    /* SELECTBOX & DROPDOWN OVERRIDES (Fix Black Background) */
    div[data-baseweb="select"] > div {
        background: linear-gradient(135deg, rgba(255, 245, 225, 0.9), rgba(255, 236, 179, 0.9)) !important;
        border: 1px solid rgba(255, 153, 51, 0.3) !important;
        border-radius: 12px !important;
        color: #4E342E !important;
    }
    
    /* POPOVER MENU (The actual dropdown list) */
    div[data-baseweb="menu"], 
    div[data-baseweb="popover"], 
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background: #FFF5E1 !important; /* Saffron Background */
        border: 1px solid #FF9933 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* OPTIONS TEXT */
    div[data-baseweb="option"], li {
        color: #4E342E !important;
        background: transparent !important;
    }
    /* HOVER STATE FOR OPTIONS */
    div[data-baseweb="option"]:hover, li[role="option"]:hover, li[aria-selected="true"] {
        background: rgba(255, 153, 51, 0.2) !important; /* Light Orange hover */
        color: #8B0000 !important;
        font-weight: 600 !important;
    }

    /* TEXT VISIBILITY FIXES */
    .stMarkdown p, .stText, div, span, li, label {
        color: #4E342E; /* Dark Brown Text */
    }
    
    /* Stronger contrast for headers and labels */
    .stSidebar label, .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar div[data-testid="stMarkdown"] p {
         color: #8B0000 !important; /* Maroon for sidebar headers/labels */
         font-weight: 600 !important;
    }
    .qa-box, .card, .content-text, .qa-q, .qa-a {
        color: #4E342E !important;
    }

    /* GOOGLE TRANSLATE WIDGET (Top Right) */
    #google_translate_element {
        position: fixed; /* Fixed to stay on top */
        top: 60px; /* Below Streamlit header */
        right: 20px;
        z-index: 99999;
    }
    .goog-te-gadget .goog-te-combo {
        background: rgba(255,255,255,0.8);
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 4px;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Mobile Optimizations - iOS, Android, Windows */
    @media (max-width: 768px) {
      /* Universal mobile fixes */
      .answer-card { 
        padding: 16px; 
        border-radius: 14px; 
        margin: 8px 0;
      }
      h1 { font-size: 1.8rem !important; }
      h2 { font-size: 1.3rem !important; }
      
      /* Input fields touch-friendly */
      input, button, .stButton button {
        min-height: 44px !important; /* Apple HIG minimum */
        font-size: 16px !important; /* Prevents iOS zoom */
      }
      
      /* Sidebar toggle more visible on mobile */
      [data-testid="collapsedControl"] {
        padding: 10px !important;
        margin: 10px !important;
      }
      
      /* Google Translate position */
      #google_translate_element { 
        top: 10px; 
        right: 10px; 
        position: absolute; 
      }
      
      /* Prevent text from being too small */
      body, p, div, span {
        -webkit-text-size-adjust: 100%;
        -moz-text-size-adjust: 100%;
        -ms-text-size-adjust: 100%;
      }
    }
    
    /* Hide Streamlit Menu, Footer, and GitHub Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;}
    .styles_viewerBadge__1yB5_ {display: none;}
    button[title="View app source"] {display: none;}
    a[href*="github"] {display: none !important;}
    [data-testid="stToolbar"] {display: none;}
    .stDeployButton {display: none;}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# BRANDED HEADER WITH PHOTOS
# ============================================================
# Using Streamlit columns for proper image display
# ============================================================
# BRANDED HEADER WITH PHOTOS (Base64 Flexbox for Perfect Alignment)
# ============================================================
img_left_b64 = get_base64_image("static/radha_krishna.jpg")
img_right_b64 = get_base64_image("static/vinod_baba.jpg")

st.markdown(f"""<style>@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');</style><div style="display: flex; align-items: center; justify-content: space-between; background: transparent; padding: 0; margin-bottom: 2rem; gap: 15px;"><div style="flex: 0 0 auto;"><img src="data:image/jpg;base64,{img_left_b64}" style="height: 110px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);"></div><div style="flex: 1; text-align: center; padding: 10px 15px; background: linear-gradient(135deg, rgba(255, 153, 51, 0.95), rgba(139, 0, 0, 0.95)); border-radius: 15px; box-shadow: 0 4px 15px rgba(139, 0, 0, 0.3); color: white; display: flex; align-items: center; justify-content: center; height: 110px;"><p style="margin: 0; font-weight: 700; font-size: 1.35rem; font-family: 'Poppins', sans-serif; letter-spacing: 0.5px; text-shadow: 1px 1px 3px rgba(0,0,0,0.3); line-height: 1.4;">🙏 श्री श्री 108 श्री विनोद बाबाजी महाराज<br><span style="font-size: 1.15rem;">Sri Sri 108 Sri Vinod Baba Ji Maharaj</span></p></div><div style="flex: 0 0 auto;"><img src="data:image/jpg;base64,{img_right_b64}" style="height: 110px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);"></div></div>""", unsafe_allow_html=True)

# ============================================================
# GLOBAL TRANSLATE WIDGET
# ============================================================
st.markdown("""
<div id="google_translate_element"></div>
<script type="text/javascript">
    function googleTranslateElementInit() {
        new google.translate.TranslateElement({
            pageLanguage: 'en', autoDisplay: false, 
            includedLanguages: 'en,gu,mr,hi,bn,es,fr,de,it,pt,ru,ar,zh-CN,ja', 
        }, 'google_translate_element');
    }
</script>
<script type="text/javascript" src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
""", unsafe_allow_html=True)



# ============================================================
# TRANSLATIONS (i18n)
# ============================================================
TRANSLATIONS = {
    "English": {
        "page_title": "Welcome to PriyaKunj",
        "home_subtitle": "Explore the divine wisdom of Shri Shri 108 Shri Vinod Bihari Das Babaji Maharaj",
        "ask_question_title": "🔍 Find an Answer",
        "ask_question_desc": "Search our extensive knowledge base of Q&A sessions.",
        "ask_question_btn": "Go to Q/A Search",
        "satsang_title": "📅 Daily Satsang",
        "satsang_desc": "Watch daily lectures, summaries, and Q&A.",
        "satsang_btn": "Go to Daily Satsang",
        "sidebar_home": "🏠 Home",
        "search_btn": "Search",
        "browse_btn": "Browse All",
        "page_size": "Page Size",
        "view_lang_label": "View Language",
        "slicer_label_en": "Quick Filters (English Keywords)",
        "slicer_label_hi": "Quick Filters (Hindi Keywords)",
        "showing_results": "Showing {start}-{end} of {total} results",
        "page_num": "Page {page} of {total}",
        "go_to": "Go to",
        "next": "Next ▶",
        "prev": "◀ Prev",
        "debug_mode": "Show Debug Info",
        "translate_toggle": "Translate English query to Hindi for search",
        "conversations_loaded": "Conversations Loaded: {count}",
        "no_satsang_files": "No {lang} content found.",
        "satsang_instruction": "How to add content:",
        "select_topic": "Select {lang} Topic",
        "viewing_file": "Viewing: {file}",
        "nav_home": "Home",
        "nav_search": "Search Q&A",
        "nav_satsang": "Satsang Notes"
    },
    "Hindi": {
        "page_title": "प्रियाकुंज में आपका स्वागत है",
        "home_subtitle": "श्री श्री 108 श्री विनोद बिहारी दास बाबाजी महाराज की दिव्य वाणी का अन्वेषण करें",
        "ask_question_title": "🔍 जवाब ढूंढें",
        "ask_question_desc": "हमारे विस्तृत प्रश्नोत्तर ज्ञान कोष में खोजें।",
        "ask_question_btn": "प्रश्नोत्तर खोज पर जाएं",
        "satsang_title": "📅 दैनिक सत्संग",
        "satsang_desc": "दैनिक प्रवचन, सारांश और प्रश्नोत्तर देखें।",
        "satsang_btn": "दैनिक सत्संग पर जाएं",
        "sidebar_home": "🏠 मुख्य पृष्ठ",
        "search_btn": "खोजें",
        "browse_btn": "सभी देखें",
        "page_size": "पृष्ठ आकार",
        "view_lang_label": "भाषा चुनें",
        "slicer_label_en": "त्वरित फिल्टर (अंग्रेजी शब्द)",
        "slicer_label_hi": "त्वरित फिल्टर (हिंदी शब्द)",
        "showing_results": "{total} परिणामों में से {start}-{end} दिखा रहा है",
        "page_num": "पृष्ठ {page} / {total}",
        "go_to": "इस पृष्ठ पर जाएं",
        "next": "अगला ▶",
        "prev": "◀ पिछला",
        "debug_mode": "डीबग जानकारी दिखाएं",
        "translate_toggle": "खोज के लिए अंग्रेजी प्रश्न का हिंदी अनुवाद करें",
        "conversations_loaded": "वार्तालाप लोड किए गए: {count}",
        "no_satsang_files": "{lang} सामग्री नहीं मिली।",
        "satsang_instruction": "सामग्री कैसे जोड़ें:",
        "select_topic": "{lang} विषय चुनें",
        "viewing_file": "देख रहे हैं: {file}",
        "nav_home": "मुख्य पृष्ठ",
        "nav_search": "प्रश्नोत्तर खोज",
        "nav_satsang": "सत्संग नोट्स"
    }
}

def get_text(key, target_lang="English", **kwargs):
    """Helper to get translated text safeley"""
    t = TRANSLATIONS.get(target_lang, TRANSLATIONS["English"]).get(key, key)
    if kwargs:
        return t.format(**kwargs)
    return t

# ============================================================
# HELPER: SATSANG METADATA EXTRACTION
# ============================================================
def extract_satsang_metadata(file_path):
    """
    Parses HTML file to find:
    1. Date (DD-MMM-YYYY or DD-MM-YYYY)
    2. Strings like 'विषय :' or the main <h1> title to use as 'Vishay'
    Returns dict: { 'date_obj': date|None, 'date_str': str, 'title': str }
    """
    import datetime
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 1. Extract Date
        # patterns: 11-DEC-2025, 11-Dec-2025, 11/12/2025, 11-12-2025
        date_obj = None
        date_str = ""
        
        # Regex for DD-MMM-YYYY (e.g., 11-DEC-2025)
        match_date_text = re.search(r"(\d{1,2})-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-(\d{4})", content, re.IGNORECASE)
        if match_date_text:
            d, m, y = match_date_text.groups()
            date_str = f"{d}-{m.upper()}-{y}"
            try:
                date_obj = datetime.datetime.strptime(date_str, "%d-%b-%Y").date()
            except:
                pass
        
        # Fallback date regex: DD-MM-YYYY
        if not date_obj:
            match_date_num = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", content)
            if match_date_num:
                d, m, y = match_date_num.groups()
                date_str = f"{d}-{m}-{y}"
                try:
                    date_obj = datetime.datetime.strptime(date_str, "%d-%m-%Y").date()
                except:
                    pass

        # 2. Extract Title (Vishay)
        # Priority 1: Line containing "विषय :" (Vishay :)
        title = "Daily Satsang"
        match_vishay = re.search(r"विषय\s*[:|-]\s*(.*?)<", content) 
        if not match_vishay:
            # try without < at end if just raw text
            match_vishay = re.search(r"विषय\s*[:|-]\s*(.*)", content)
            
        if match_vishay:
            title = match_vishay.group(1).strip()
            # remove HTML tags if any slipped in
            title = re.sub(r"<[^>]+>", "", title).strip()
        else:
            # Priority 2: content of first <h1>
            match_h1 = re.search(r"<h1[^>]*>(.*?)</h1>", content, re.DOTALL | re.IGNORECASE)
            if match_h1:
                raw_h1 = match_h1.group(1)
                # clean tags
                title = re.sub(r"<[^>]+>", "", raw_h1).strip()
        
        # Final formatting: "Title | DD, MMM, YYYY"
        # If we have a valid date object, format nicely
        display_date = date_str
        if date_obj:
            display_date = date_obj.strftime("%d, %b, %Y")
            
        full_title = title
        if display_date:
            full_title = f"{title} | {display_date}"
            
        return {
            "date_obj": date_obj,
            "date_str": display_date,
            "title": title,
            "full_title": full_title
        }
        
    except Exception as e:
        print(f"Metadata parsing error for {file_path}: {e}")
        return {"date_obj": None, "date_str": "", "title": "Satsang", "full_title": "Satsang"}

WHATSAPP_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(AM|PM)\s*-\s*",  # "1/10/25, 7:11 PM -"
    r"\b\d{1,2}:\d{2}\s*(AM|PM)\b",                                # "7:11 PM"
]

DEVOTIONAL_PHRASES_HI = [
    "दंडवत प्रणाम", "दण्डवत प्रणाम", "दंडवत", "दण्डवत",
    "प्रणाम", "प्रणाम जी", "प्रणामजी",
    "जय गुरु", "जय गुरु।", "जय गुरुजी", "जय गुरुदेव",
    "प्रभु जी", "प्रभुजी", "प्रभु जी।",
    "राधे राधे", "जय श्री राधे", "श्री राधे",
    "हरी बोल", "हरि बोल", "गौर हरि बोल", "निताई गौर हरि बोल",
]

DEVOTIONAL_PHRASES_ROMAN = [
    "dandavat pranam", "dandavat", "pranam",
    "jai guru", "jai gurudev",
    "prabhu ji", "prabhuji",
    "radhe radhe", "hari bol",
    "nitai gaur hari bol", "gaur hari bol",
]

# English stopwords (prevents false lexical matches for English queries)
EN_STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from","has","have","he","her",
    "his","i","if","in","into","is","it","its","me","my","not","of","on","or","our",
    "she","so","that","the","their","them","then","there","these","they","this","to",
    "was","we","were","what","when","where","which","who","will","with","you","your",
    "am","able","can","cant","cannot","could","couldnt","do","does","doesnt","did","didnt",
    "been","being","im","ive","id","ill","wont","dont","isnt","arent","wasnt","werent"
}

# ============================================================
# SLICER STOPWORDS (strong filtering for UI only)
# ============================================================
SLICER_STOPWORDS = {
    # generic English glue
    "should", "how", "why", "when", "where", "which", "who",
    "during", "since", "only", "also", "down", "up", "into",
    "from", "about", "after", "before", "over", "under", "within",

    # devotional / names (not search intents)
    "radheshyam", "radhe", "shyam", "shri",
    "baba", "babaji", "gurudev", "guru", "ji",
    "lord", "prabhu", "dev",
    "babashri", "maharaj",

    # politeness / structure
    "please", "kindly", "guide", "guidance",
    "salutations", "bow", "feet", "ground",
    "question",

    # weak verbs / fillers
    "do", "does", "did", "done", "make", "made",
    "use", "using", "used", "get", "got",
    "day", "month", "year", "time",
    "want", "show", "everything", "full"
}

# Common Hindi stopwords to ignore in slicers
HI_STOPWORDS = {
    "के", "का", "एक", "में", "की", "है", "यह", "और", "से", "हैं", "को", "पर", "इस", "होता", "कि", "जो",
    "कर", "मे", "गया", "करने", "किया", "लिये", "अपने", "ने", "बनी", "नहीं", "तो", "ही", "या", "एवं", "दिया",
    "हो", "इसका", "था", "द्वारा", "हुआ", "तक", "साथ", "करना", "वाले", "बाद", "लिए", "आप", "कुछ", "सकते",
    "किसी", "ये", "इसके", "सबसे", "इसमें", "थे", "दो", "मगर", "वह", "भी", "सकता", "हर", "जाने", "अपना",
    "वे", "जिसे", "गई", "ऐसे", "जिसके", "लिए", "जाता", "बहुत", "कहा", "वर्ग", "कई", "करें", "होती", "वाले",
    "कम", "से", "थी", "हुई", "जा", "न", "जिस", "किस", "तथा", "हूँ", "मै", "मैं", "मेरा", "मेरी", "मेरे",
    "मुझे", "हम", "हमारा", "हमारे", "हमें", "तुम", "तुम्हारा", "तुम्हारे", "तुम्हें", "आपका", "आपकी", "आपके",
    "क्या", "क्यों", "कैसे", "कब", "कहाँ", "कौन", "जी", "साहब", "सर", "श्री", "श्रीमती", "कुमार", "कुमारी",
    "सवाल", "प्रश्न", "उत्तर", "जवाब", "चाहिए", "चाहता", "चाहती", "चाहते", "रहा", "रही", "रहे",
    "बारे", "पास", "दूर", "सब", "सभी", "सारा", "पूरी", "पूरा",
    # Specific removals requested
    "जैसे", "बाबा", "राधे", "श्याम", "राधेश्याम", "कोई", "कृपया",
    "मिल", "करते", "कल", "बताया", "लेना", "समय", "उसमें", "जय", "भगवान", "देव",
    "गुरु",
    # Additional removals
    "कृपा", "मार्गदर्शन", "jae", "जाए", "जाएं", "वो", "करे", "करें", "कभी", "अगर",
    "उसके", "उसकी", "उसे", "उनका", "उनकी", "उनके", "उन्हें",
    "यहाँ", "वहाँ", "जहां", "अब", "जब", "तब"
}

# ============================================================
# 2A) SYNONYM EXPANSION (COMMON SCENARIOS)
# ============================================================
SYNONYMS = {
    # ===== Core: snatch/take away =====
    "छीन": [
        "छिन", "छीनना", "छीना", "छीने", "छीन लिया", "छीन लिए",
        "छीन ले", "छीन लेता", "छीन लेते", "छीन लेती", "छीन लेते हैं",
        "छीन लिया गया", "छीन लिया जाता", "छीन लिया जाता है",
        "झपट", "झपट लेना", "झपट लिया",
        "हरण", "हरण करना", "हरण कर लेना", "हरण हो गया",
        "छीन-झपट",
        "ले लेना", "ले लिया", "ले लेते", "ले लेते हैं"
    ],

    "छीन लेना": [
        "छीन लिया", "छीन लिए", "छीन लेता", "छीन लेते", "छीन लेते हैं",
        "हरण", "हरण करना", "हरण कर लेना",
        "वापस लेना", "वापस ले लेना", "वापस ले लिया",
        "ले लेना", "ले लिया", "ले लेते", "ले लेते हैं",
        "खींच लेना", "उठा लेना",
        "खो देना", "वंचित करना"
    ],

    "छीनना": [
        "छीन", "छिन", "छीन लिया", "छीन लेते", "छीन लेते हैं",
        "हरण", "हरण करना",
        "ले लेना", "ले लिया", "ले लेते"
    ],

    # ===== Take / withdraw / take back =====
    "ले लेना": [
        "ले लिया", "ले लिए", "ले लेते", "ले लेते हैं", "ले गया", "ले गए", "ले गये",
        "उठा लेना", "उठा लिया",
        "ख़ींच लेना", "खींच लेना", "खींच लिया", "खिंच लिया",
        "वापस लेना", "वापस ले लेना", "लौटा लेना", "लौटा लिया",
        "हरण", "हरण करना",
        "छीन", "छीन लेना"
    ],

    "ले लेते": [
        "ले लेते हैं", "ले लिया", "ले लेना",
        "छीन लेते", "छीन लेते हैं",
        "हरण", "हरण कर लेते"
    ],

    "ले लेते हैं": [
        "ले लेते", "ले लिया", "ले लेना",
        "छीन लेते", "छीन लेते हैं",
        "हरण"
    ],

    "वापस लेना": [
        "वापस ले लेना", "वापस ले लिया", "वापस ले लेते", "वापस ले लेते हैं",
        "लौटा लेना", "लौटा लिया",
        "छीन लेना", "हरण", "ले लेना"
    ],

    "खींच लेना": [
        "खिंच लेना", "खींच लिया", "खिंच लिया",
        "ले लेना", "वापस लेना", "छीन लेना"
    ],

    "उठा लेना": [
        "उठा लिया", "ले लेना", "ले लिया",
        "छीन लेना"
    ],

    # ===== Deprive / deny / withhold =====
    "वंचित": [
        "वंचित करना", "वंचित हो गया",
        "अधिकार छीन", "अधिकार छीनना",
        "से वंचित", "से वंचित करना",
        "न मिलने देना", "रोक देना"
    ],

    "रोक देना": [
        "रोक", "रोकना", "रुक गया", "रुक जाना",
        "बंद कर देना", "बंद कर दिया",
        "न मिलने देना", "न देना",
        "वंचित करना"
    ],

    "बंद कर देना": [
        "बंद कर दिया", "बंद कर दी", "बंद हो गया",
        "रोक देना", "रोकना"
    ],

    # ===== Remove / detach =====
    "हटा देना": [
        "हटा", "हटाना", "हटा दिया", "हटा दी",
        "दूर करना", "दूर कर देना", "दूर हो गया",
        "निकाल देना", "निकाल दिया",
        "छीन लेना", "ले लेना"
    ],

    "दूर करना": [
        "दूर कर देना", "दूर हो जाना", "हटा देना", "निकाल देना"
    ],

    # ===== Loss (common user phrasing) =====
    "खो देना": [
        "खो गया", "खो गई", "खो गए",
        "गुम हो गया", "गुम गया",
        "नष्ट हो गया", "चला गया",
        "हाथ से निकल गया",
        "ले लिया", "वापस ले लिया"
    ],

    "चला गया": [
        "चला गया था", "चले गए", "चली गई",
        "छूट गया", "छूट गई",
        "खो गया"
    ],

    # ===== Confiscate / seize =====
    "जप्त": [
        "जब्त", "जप्त करना", "जब्त करना",
        "कब्ज़ा", "कब्जा", "कब्जा कर लेना",
        "छीन लेना"
    ],

    # ===== Spiritual / satsang framing =====
    "परीक्षा": [
        "परीक्षा लेना", "परीक्षा ले रहे",
        "परखना", "परख लेते",
        "लीला", "कृपा", "अनुग्रह",
        "वैराग्य", "त्याग",
        "आसक्ति", "मोह", "बंधन"
    ],

    "आसक्ति": [
        "मोह", "बंधन", "लगाव",
        "वैराग्य", "त्याग",
        "हटा देना", "दूर करना"
    ],

    # ===== Common phrase patterns =====
    "सब कुछ ले": [
        "सब कुछ ले लिया",
        "सब कुछ छीन लिया",
        "सब कुछ हरण",
        "सब कुछ ले लेते",
        "सब कुछ ले लेते हैं"
    ],

    "सब कुछ ले लेते": [
        "सब कुछ ले लेते हैं",
        "सब कुछ ले लिया",
        "सब कुछ छीन लिया",
        "सब कुछ हरण"
    ],
}

# Guardrail: do not allow extremely common bridge tokens unless query has a “taking away” trigger
BRIDGE_TOKENS = {
    "ले", "लेना", "ले लिया", "ले लिए", "ले लेते", "ले लेते हैं",
    "ले गया", "ले गए", "ले गये",
}
BRIDGE_TRIGGERS = ["छीन", "हरण", "वापस", "खींच", "उठा", "वंचित", "जप्त", "जब्त", "कब्जा", "कब्ज़ा"]


# ============================================================
# 2B) OPTIONAL: illness bridge for English queries (helps "I am sick" -> cold/cough)
# ============================================================
ILLNESS_SYNONYMS_EN = {
    "sick": ["ill", "unwell", "fever", "cold", "cough", "flu", "temperature"],
    "ill": ["sick", "unwell", "fever", "cold", "cough", "flu"],
    "fever": ["temperature", "high", "bukhar", "bukhaar"],
    "cold": ["cough", "flu", "runny", "nose"],
    "cough": ["cold", "flu", "throat"],
}
ILLNESS_BRIDGE_HI = {
    # if query translated/typed in Hinglish/Hindi
    "बीमार": ["जुकाम", "खांसी", "बुखार", "सर्दी", "कफ"],
    "जुकाम": ["सर्दी", "खांसी", "बीमार"],
    "खांसी": ["जुकाम", "सर्दी", "बीमार"],
    "बुखार": ["ताप", "बीमार"],
    "सर्दी": ["जुकाम", "खांसी", "बीमार"],
}

EN_STOPWORDS_UI = EN_STOPWORDS | {
    "please","kindly","help","guidance","question","answer","baba","guru","ji",
    "radhe","shyam","pranam","thanks","thank"
}

def pick_english_source_column(df: pd.DataFrame) -> str | None:
    # Preferred: English Text
    for col in ["English Text", "English", "Translated Question", "Translated Answer"]:
        if col in df.columns:
            return col
    return None

def extract_top_keywords(df: pd.DataFrame, col: str, top_n: int = 30) -> list[str]:
    """
    Extract intent-worthy English keywords for slicers.
    Rules:
    - English alphabetic words only
    - length >= 4
    - remove EN_STOPWORDS + SLICER_STOPWORDS
    - frequency-based ranking
    """
    freq = {}
    series = df[col].fillna("").astype(str)

    for text in series:
        text = clean_for_search(text).lower()

        # Only alphabetic English words, min length 4
        tokens = re.findall(r"[a-z]{4,}", text)

        for t in tokens:
            if t in EN_STOPWORDS:
                continue
            if t in SLICER_STOPWORDS:
                continue
            freq[t] = freq.get(t, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]

def extract_hindi_keywords(df: pd.DataFrame, top_n: int = 30) -> list[str]:
    """
    Extract useful Hindi keywords for slicers.
    Rules:
    - Tokenize using whitespace
    - Remove HI_STOPWORDS
    - Min length 2
    - Frequency ranking
    """
    freq = {}
    # Use "Question" column for source
    series = df["Question"].fillna("").astype(str)

    for text in series:
        # Simple split by whitespace
        # (For better Devanagari tokenization, we can use clean_for_search first)
        text = clean_for_search(text)
        tokens = text.split()

        for t in tokens:
            # Check if likely Devanagari
            if not any("\u0900" <= ch <= "\u097F" for ch in t):
                continue
            
            # Stopword filter
            if t in HI_STOPWORDS:
                continue
            
            # Length filter
            if len(t) < 2:
                continue
                
            freq[t] = freq.get(t, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200c", "").replace("\u200d", "")  # ZWNJ/ZWJ
    return s

def remove_devotional_boilerplate(s: str) -> str:
    s = normalize_text(s)
    low = s.lower()

    for p in DEVOTIONAL_PHRASES_ROMAN:
        low = low.replace(p, " ")

    for p in DEVOTIONAL_PHRASES_HI:
        low = low.replace(p.lower(), " ")
        low = re.sub(rf"\b{re.escape(p.lower())}\b[।.!?,;:]*", " ", low)

    low = re.sub(r"\s+", " ", low).strip()
    return low

def clean_for_search(s: str) -> str:
    s = normalize_text(s)

    # Remove phone numbers
    s = re.sub(r"\+?\d[\d\s\-]{8,}\d", " ", s)

    # Remove WhatsApp timestamps
    for pat in WHATSAPP_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)

    # Remove WhatsApp system fragments
    s = re.sub(r"\badded\b.*", " ", s, flags=re.IGNORECASE)

    # Remove devotional boilerplate
    s = remove_devotional_boilerplate(s)

    # Keep letters/numbers/underscore/space + Devanagari
    s = re.sub(r"[^\w\s\u0900-\u097F]", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_hi_en(q: str) -> list[str]:
    """
    Tokenizer with guardrails:
      - Remove common English stopwords
      - Ignore short English tokens (<3)
      - Keep Hindi tokens (>=2)
    """
    q = clean_for_search(q).lower()
    raw = [t for t in q.split() if t]

    toks = []
    for t in raw:
        is_english = all(ord(c) < 128 for c in t)

        if is_english:
            if t in EN_STOPWORDS:
                continue
            if len(t) < 3:
                continue
            toks.append(t)
        else:
            if len(t) >= 2:
                toks.append(t)

    return toks

def has_hindi_token(tokens: list[str]) -> bool:
    return any(any("\u0900" <= ch <= "\u097F" for ch in tok) for tok in tokens)

def allow_bridge(query_text: str) -> bool:
    q = clean_for_search(query_text).lower()
    return any(t in q for t in BRIDGE_TRIGGERS)

def expand_tokens(tokens: list[str], query_text: str) -> list[str]:
    """
    Phrase-aware + token-aware expansion.
    Adds:
      - satsang synonym map (SYNONYMS)
      - limited illness bridging for English/Hindi health queries
      - guardrail for ultra-common bridge tokens
    """
    expanded = set()
    q_clean = clean_for_search(query_text).lower()

    # Original tokens
    for t in tokens:
        if t:
            expanded.add(t)

    # Phrase-aware triggers (SYNONYMS)
    for key, syns in SYNONYMS.items():
        key_clean = clean_for_search(key).lower()
        if key_clean and key_clean in q_clean:
            expanded.add(key_clean)
            for s in syns:
                expanded.add(clean_for_search(s).lower())

    # Token-level expansions (SYNONYMS)
    for t in tokens:
        for s in SYNONYMS.get(t, []):
            expanded.add(clean_for_search(s).lower())

    # Illness bridge (English tokens)
    for t in tokens:
        if all(ord(c) < 128 for c in t):
            for s in ILLNESS_SYNONYMS_EN.get(t, []):
                expanded.add(clean_for_search(s).lower())

    # Illness bridge (Hindi tokens)
    for t in tokens:
        if not all(ord(c) < 128 for c in t):
            for s in ILLNESS_BRIDGE_HI.get(t, []):
                expanded.add(clean_for_search(s).lower())

    # Guardrail for ultra-common bridge tokens
    if not allow_bridge(query_text):
        expanded = {x for x in expanded if x not in BRIDGE_TOKENS}

    return [x for x in expanded if x]


def lexical_score(query: str, text: str, phrase_boost: bool = True) -> float:
    q_clean = clean_for_search(query).lower()
    t = (text or "").lower()

    base_toks = tokenize_hi_en(query)
    toks = expand_tokens(base_toks, query)

    if not toks:
        return 0.0

    hits = sum(1 for tok in toks if tok in t)
    score = hits / len(toks)

    # Phrase boost for short queries
    if phrase_boost and len(base_toks) <= 2 and q_clean and (q_clean in t):
        score = min(1.0, score + 0.5)

    return float(score)


# ============================================================
# 3) EMBEDDERS
# ============================================================
class GoogleEmbedder:
    """
    Gemini embeddings using google.generativeai.
    Robust parsing + retries; returns stable numpy arrays.
    """
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)

    @staticmethod
    def _extract_embedding(result):
        if isinstance(result, dict):
            if "embedding" in result:
                return result["embedding"]
            if "embeddings" in result:
                return result["embeddings"]
        if hasattr(result, "embedding"):
            return getattr(result, "embedding")
        if hasattr(result, "embeddings"):
            return getattr(result, "embeddings")
        return None

    def _embed_one(self, text: str, task_type: str, max_retries: int = 3):
        last_err = None
        for attempt in range(max_retries):
            try:
                r = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=task_type
                )
                emb = self._extract_embedding(r)

                if isinstance(emb, list) and emb and isinstance(emb[0], (float, int)):
                    return [float(x) for x in emb]

                if isinstance(emb, dict) and "values" in emb:
                    return [float(x) for x in emb["values"]]

                if isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
                    return [float(x) for x in emb[0]]

                return None
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (2 ** attempt))

        st.warning(f"Gemini embedding failed for a text: {last_err}")
        return None

    def encode(self, texts: list[str], task_type: str = "retrieval_document",
               max_retries: int = 3) -> np.ndarray:
        vectors: list[list[float] | None] = []
        dim: int | None = None

        for t in texts:
            v = self._embed_one(t, task_type=task_type, max_retries=max_retries)
            if v is not None and dim is None:
                dim = len(v)
            vectors.append(v)

        if dim is None:
            return np.array([])

        fixed = []
        for v in vectors:
            if v is None or len(v) != dim:
                fixed.append([0.0] * dim)
            else:
                fixed.append(v)

        return np.array(fixed, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.encode([text], task_type="retrieval_query")
        return vec if vec.size else np.array([])


# ============================================================
# 3B) TRANSLATION BRIDGE (English -> Hindi) for better recall
# ============================================================
def translate_to_hindi_if_english(q: str, api_key: str) -> str:
    """
    If query is mostly ASCII and has no Devanagari, translate to Hindi using Gemini.
    Returns original query on failure / no API key.
    """
    # If already contains Devanagari, skip
    if any("\u0900" <= ch <= "\u097F" for ch in q):
        return q

    # Heuristic: mostly ASCII -> likely English
    ascii_ratio = sum(1 for ch in q if ord(ch) < 128) / max(1, len(q))
    if ascii_ratio < 0.85:
        return q

    if not api_key:
        return q

    try:
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Translate the user query into natural Hindi for searching a devotional Q&A dataset. "
            "Return ONLY the Hindi translation, no extra text.\n\n"
            f"Query: {q}"
        )
        r = gm.generate_content(prompt)
        out = (getattr(r, "text", None) or "").strip()
        return out if out else q
    except Exception:
        return q


# ============================================================
# 4) LOAD DATA (Google Sheet CSV)
# ============================================================
@st.cache_data(ttl=600)
def load_data():
    SHEET_ID = "1JtpDSVREK0pH2CwOMktqdQaS8zvBNox1444dkTfnjws"
    GID = "1635748443"
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

    try:
        df = pd.read_csv(url).fillna("")

        if "Question" not in df.columns:
            return pd.DataFrame(), "Error: 'Question' column missing from sheet."
        if "Answer" not in df.columns:
            df["Answer"] = ""
        if "Translated Question" not in df.columns:
            df["Translated Question"] = ""
        if "Translated Answer" not in df.columns:
            df["Translated Answer"] = ""

        df = df.reset_index(drop=True)

        # Clean fields
        df["clean_question"] = df["Question"].apply(clean_for_search)
        df["clean_translated_q"] = df["Translated Question"].apply(clean_for_search)
        df["clean_answer"] = df["Answer"].apply(clean_for_search)
        df["clean_translated_a"] = df["Translated Answer"].apply(clean_for_search)

        # Embedding corpus (question-centric)
        df["embed_text"] = (df["clean_question"] + " " + df["clean_translated_q"]).str.strip()

        # Lexical corpus (includes answer too)
        df["lex_text"] = (df["clean_question"] + " " + df["clean_translated_q"] + " " + df["clean_answer"]).str.strip()

        # Drop near-empty rows after cleaning
        df["embed_len"] = df["embed_text"].apply(lambda x: len((x or "").strip()))
        df = df[df["embed_len"] >= 10].reset_index(drop=True)

        return df, None

    except Exception as e:
        return pd.DataFrame(), f"Could not load data. Error: {e}"


# ============================================================
# 5) BUILD EMBEDDING INDEX (cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def build_index(provider: str, api_key: str, texts_tuple: tuple[str, ...]):
    try:
        texts = list(texts_tuple)

        if provider == "Google Gemini":
            if not api_key:
                return None, None, "Please enter a Google API Key."
            model = GoogleEmbedder(api_key=api_key, model_name="models/text-embedding-004")
            embeddings = model.encode(texts, task_type="retrieval_document")
            if embeddings.size == 0:
                return None, None, "Failed to build Gemini embeddings index."
            return model, embeddings, None

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
        return model, embeddings, None

    except Exception as e:
        return None, None, f"Error initializing {provider}: {e}"


# ============================================================
# STATE MANAGEMENT & LANGUAGE
# ============================================================
if "view_lang" not in st.session_state:
    st.session_state["view_lang"] = "English"

if "current_view" not in st.session_state:
    st.session_state["current_view"] = "home"

# Deep Linking Logic
if "satsang" in st.query_params:
    st.session_state["current_view"] = "satsang"
    st.session_state["deep_linked_satsang"] = st.query_params["satsang"]

# Language Toggle (Top Right or Sidebar - keeping consistent with previous UX)
# usage: we will render the radio button later, but we need the value now.
# Note: Streamlit widgets return the value *after* interaction, 
# so we might moved the widget here or just rely on session state default.

# Let's put the language toggle in the sidebar for cleaner global access, 
# OR keep it at top of main area. 
# Current design: Top of main area.
# We will read st.session_state if available, else default.

# ============================================================
# HOME PAGE LOGIC
# ============================================================
def render_home_page(lang):
    # Language Toggle at top right of page
    col_space, col_lang = st.columns([3, 1])
    with col_lang:
        selected_lang = st.radio(
            "भाषा / Language",
            ["हिंदी", "English"],
            index=0 if st.session_state["view_lang"] == "Hindi" else 1,
            horizontal=True,
            key="home_lang_toggle"
        )
        new_lang = "Hindi" if selected_lang == "हिंदी" else "English"
        if new_lang != st.session_state["view_lang"]:
            st.session_state["view_lang"] = new_lang
            st.rerun()
    
    # Use the CURRENT session state value for text (reflects any toggle change)
    current_lang = st.session_state["view_lang"]
    t_title = get_text("page_title", current_lang)
    t_sub = get_text("home_subtitle", current_lang)
    
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">{t_title}</h1>
        <p style="font-size: 1.2rem; color: #5A2D0C; margin-bottom: 40px;">
            {t_sub}
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="answer-card" style="text-align: center; min-height: 250px; display: flex; flex-direction: column; justify-content: center; align-items: center; cursor: pointer;">
            <h2 style="color: #FF8000; margin-top: 0;">{get_text('ask_question_title', current_lang)}</h2>
            <p style="color: #3E2723; margin-bottom: 20px;">{get_text('ask_question_desc', current_lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("👆 यहाँ क्लिक करें / Click Here", use_container_width=True, key="home_card_qa"):
            st.session_state["current_view"] = "qa"
            st.rerun()

    with c2:
        st.markdown(f"""
        <div class="answer-card" style="text-align: center; min-height: 250px; display: flex; flex-direction: column; justify-content: center; align-items: center; cursor: pointer;">
            <h2 style="color: #FF8000; margin-top: 0;">{get_text('satsang_title', current_lang)}</h2>
            <p style="color: #3E2723; margin-bottom: 20px;">{get_text('satsang_desc', current_lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("👆 यहाँ क्लिक करें / Click Here", use_container_width=True, key="home_card_satsang"):
            st.session_state["current_view"] = "satsang"
            st.rerun()

# ============================================================
# SATSANG (BLOG) PAGE LOGIC
# ============================================================
import os


def extract_share_content(html_content):
    """Parses HTML content to extract headers and questions for sharing."""
    import re
    # Find h2, h3 headers
    headers = re.findall(r'<h[23][^>]*>(.*?)</h[23]>', html_content)
    # Find lines ending with ? inside p tags or just lines
    questions = re.findall(r'<p[^>]*>(.*?\?)</p>', html_content)
    
    # Clean tags
    cleaned = []
    seen = set()
    
    # Prioritize headers then questions
    raw_items = headers + questions
    
    for item in raw_items:
        text = re.sub(r'<[^>]+>', '', item).strip()
        if text and len(text) > 5 and text not in seen:
            cleaned.append(text)
            seen.add(text)
            
    return cleaned[:5]  # Return top 5 interesting points

def preprocess_html_for_markdown(html_content: str) -> str:
    """
    Strips leading whitespace from each line to prevent Streamlit's Markdown parser
    from interpreting indented HTML tags as code blocks.
    """
    lines = html_content.splitlines()
    # lstrip() removes leading whitespace
    return "\n".join(line.lstrip() for line in lines)

def render_satsang_page(view_lang):
    # Check for Deep Link
    if "satsang" in st.query_params:
        target_satsang = st.query_params["satsang"]
        st.session_state["deep_linked_satsang"] = target_satsang
    
    # Google Translate Widget (per user request)
    translate_css = """
    <style>
        .goog-te-banner-frame.skiptranslate { display: none !important; }
        #google_translate_element { position: relative; z-index: 9999; }
        .goog-te-gadget { font-size: 0px !important; color: transparent !important; }
        .goog-te-gadget .goog-te-combo {
            font-size: 13px !important; font-weight: 500; color: #555 !important;
            background: rgba(255,255,255,0.8); border: 1px solid rgba(0,0,0,0.1);
            padding: 4px 8px; border-radius: 6px; outline: none; cursor: pointer;
            box-shadow: none; font-family: 'Poppins', sans-serif;
        }
    </style>
    """
    
    translate_widget = """
    <div id="google_translate_element"></div>
    <script type="text/javascript">
        function googleTranslateElementInit() {
            new google.translate.TranslateElement({
                pageLanguage: 'en', autoDisplay: false, 
                includedLanguages: 'en,gu,mr,hi,bn,es,fr,de,it,pt,ru,ar,zh-CN,ja', 
            }, 'google_translate_element');
        }
    </script>
    <script type="text/javascript" src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    """

    # Page Title - Above navigation
    t_title = get_text("satsang_title", view_lang)
    st.markdown(f'<h2 style="text-align: center; color: #8B0000; margin: 0 0 15px 0;">{t_title}</h2>', unsafe_allow_html=True)
    
    # Navigation Buttons - Equal size columns
    c_home, c_search, c_trans = st.columns([1, 1, 1])
    with c_home:
        if st.button(get_text("nav_home", view_lang), use_container_width=True, key="satsang_home_btn"):
            st.session_state["current_view"] = "home"
            st.rerun()
    
    with c_search:
        if st.button(get_text("nav_search", view_lang), use_container_width=True, key="satsang_search_btn"):
            st.session_state["current_view"] = "qa"
            st.rerun()
        
    with c_trans:
        st.markdown(translate_css + translate_widget, unsafe_allow_html=True)

    st.markdown("---")
    
    BASE_DIR = "satsang_content"
    CONTENT_DIR = BASE_DIR
    
    if not os.path.exists(CONTENT_DIR):
        os.makedirs(CONTENT_DIR, exist_ok=True)
        
    # 1. SCAN & PARSE METADATA
    all_files = []
    
    for root, cols, files in os.walk(CONTENT_DIR):
        for file in files:
            if file.endswith(".html"):
                full_path = os.path.join(root, file)
                meta = extract_satsang_metadata(full_path)
                
                # Determine sorting key (Date object > Date string > Filename)
                sort_key = meta["date_str"] if meta["date_str"] else file
                try:
                    if meta["date_obj"]: sort_key = meta["date_obj"].isoformat()
                except: pass
                
                # Determine Group (Month YYYY)
                group_label = "Uncategorized"
                if meta["date_obj"]:
                    group_label = meta["date_obj"].strftime("%B %Y") # e.g., December 2025
                elif meta["date_str"]:
                     # If string but no obj, try to just grab rightmost chunk
                     # but let's stick to Uncategorized for safety if parsing failed
                     pass

                all_files.append({
                    "path": full_path,
                    "filename": file,
                    "title": meta["title"],
                    "full_title": meta["full_title"],
                    "date_obj": meta["date_obj"],
                    "group": group_label,
                    "sort_key": sort_key
                })

    if not all_files:
        st.info(get_text("no_satsang_files", view_lang, lang=view_lang))
        return

    # 2. SORT & GROUP
    # Sort descending by date
    all_files.sort(key=lambda x: x["sort_key"], reverse=True)
    
    # Get unique groups in order of appearance (since we sorted files by date, groups will be chronological desc)
    groups = []
    files_by_group = {}
    
    for f in all_files:
        g = f["group"]
        if g not in files_by_group:
            groups.append(g)
            files_by_group[g] = []
        files_by_group[g].append(f)
        
    # 3. ARCHIVE SELECTOR UI
    
    # Check for Deep Link Match
    default_group_idx = 0
    default_file_idx = 0
    
    if "deep_linked_satsang" in st.session_state:
        target = st.session_state["deep_linked_satsang"]
        for g_idx, group_name in enumerate(groups):
            g_files = files_by_group[group_name]
            for f_idx, f in enumerate(g_files):
                if f["filename"] == target:
                    default_group_idx = g_idx
                    default_file_idx = f_idx
                    # Clear it so navigation works normally after
                    del st.session_state["deep_linked_satsang"]
                    break
            else:
                continue
            break
            
    c_month, c_topic = st.columns([1, 2])
    
    with c_month:
        st.markdown("**🗓️ Archives**")
        selected_group = st.selectbox(
            "Select Month", 
            groups, 
            index=default_group_idx,
            label_visibility="collapsed"
        )
        
    with c_topic:
        st.markdown("**📖 Select Topic**")
        filtered_files = files_by_group[selected_group]
        # Map full titles to indices
        file_options = {f["full_title"]: i for i, f in enumerate(filtered_files)}
        
        # Ensure default file index is valid for the selected group
        current_file_idx = default_file_idx if groups[default_group_idx] == selected_group else 0
        
        selected_title = st.selectbox(
            "Topic", 
            list(file_options.keys()), 
            index=min(current_file_idx, len(filtered_files)-1),
            label_visibility="collapsed"
        )
        
    selected_file_data = filtered_files[file_options[selected_title]]
    file_path = selected_file_data["path"]
    
    # 4. RENDER CONTENT
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        # 4. RENDER CONTENT (IFRAME ISOLATION - Dynamic Height)
        # Using components.html creates a sandboxed iframe. 
        # This solves ALL scope issues (const redeclaration), ID collisions, and script execution failures.
        import streamlit.components.v1 as components
        
        # ENHANCED resize script - aggressively monitors for content changes
        # This ensures minimal white space for both Short and Long options
        auto_resize_script = """
<script>
(function() {
    var lastHeight = 0;
    var resizeAttempts = 0;
    var maxAttempts = 15;
    
    function notifyResize() {
        // Ensure body/html allow expansion
        document.body.style.height = 'auto';
        document.documentElement.style.height = 'auto';
        
        // Get actual content height with generous padding
        var body = document.body;
        var html = document.documentElement;
        var height = Math.max(
            body.scrollHeight,
            body.offsetHeight,
            html.clientHeight,
            html.scrollHeight,
            html.offsetHeight
        ) + 100;  // Extra padding to prevent cutoff
        
        // Only notify if height changed significantly
        if (Math.abs(height - lastHeight) > 10) {
            lastHeight = height;
            window.parent.postMessage({
                type: "streamlit:setFrameHeight",
                height: height
            }, "*");
            console.log("Resized to:", height);
        }
        
        resizeAttempts++;
    }
    
    // MutationObserver to detect DOM changes (Short/Long toggle)
    var observer = new MutationObserver(function(mutations) {
        // Reset attempts counter on mutation
        resizeAttempts = 0;
        setTimeout(notifyResize, 100);
        setTimeout(notifyResize, 400);
        setTimeout(notifyResize, 800);
    });
    
    // Observe entire body for changes including subtrees
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true
    });
    
    // Click handler for toggle buttons
    document.addEventListener('click', function(e) {
        resizeAttempts = 0;
        setTimeout(notifyResize, 200);
        setTimeout(notifyResize, 500);
        setTimeout(notifyResize, 1000);
    }, true);
    
    // Window resize handler
    window.addEventListener('resize', function() {
        setTimeout(notifyResize, 100);
    });
    
    // Force initial resize
    notifyResize();
    setTimeout(notifyResize, 500);
    setTimeout(notifyResize, 1500);
    setTimeout(notifyResize, 3000);
    
    // Periodic check (backup)
    var periodicCheck = setInterval(function() {
        if (resizeAttempts < maxAttempts) {
            notifyResize();
        } else {
            clearInterval(periodicCheck);
        }
    }, 1000);
})();
</script>
</body>
"""
        
        # Inject script before closing body tag
        if "</body>" in raw_content:
            raw_content = raw_content.replace("</body>", auto_resize_script)
        
        # Estimate height - balanced to fit content without scrolling or excess whitespace
        import re
        text_only = re.sub(r'<[^>]+>', '', raw_content)
        text_length = len(text_only)
        
        # Lower multiplier to minimize whitespace
        # Minimum 400px, Maximum 2500px
        estimated_height = max(400, min(2500, int(text_length * 0.15)))
        
        # 0.15 multiplier with scrolling for overflow
        components.html(raw_content, height=estimated_height, scrolling=True)
        
        # 5. SOCIAL / ENGAGEMENT UI (Below the content)
        st.markdown("---")
        c_like, c_share = st.columns([1, 4])
        
        # Like Button (Session State)
        like_key = f"like_{selected_file_data['filename']}"
        if like_key not in st.session_state:
            st.session_state[like_key] = False
            
        with c_like:
            if st.button("❤️ Like" if not st.session_state[like_key] else "💖 Liked"):
                st.session_state[like_key] = not st.session_state[like_key]
                st.rerun()
                
        # Social Share Links - Enhanced Format
        with c_share:
            # Rich metadata for cross-platform compatibility
            content_highlights = extract_share_content(raw_content)
            highlights_text = "\n".join([f"✨ {h}" for h in content_highlights])
            
            # Construct Deep Link
            # TODO: Update this URL after deploying to Streamlit Cloud
            # For local testing, use localhost; for production, use your deployed URL
            base_url = "http://localhost:8502"  # Change to your Streamlit Cloud URL after deployment
            share_link = f"{base_url}/?satsang={selected_file_data['filename']}"
            
            share_text = f"""🙏 *{selected_file_data['full_title']}*

📅 {selected_file_data.get('date_str', '')}
📍 प्रियाकुंज आश्रम, बरसाना

🔍 *मुख्य अंश / Highlights:*
{highlights_text}

🔗 *पूरा सत्संग पढ़ें / Read Full:*
{share_link}

🕉️ जय श्री राधे""".strip()
            import urllib.parse
            safe_text = urllib.parse.quote(share_text)
            
            # Generate unique ID for this share link
            link_id = f"link_{selected_file_data['filename'].replace('.', '_')}"
            
            st.markdown(f"""
            <div style="display:flex; gap:10px; align-items:center; flex-wrap: wrap; margin-bottom: 10px;">
                <a href="https://wa.me/?text={safe_text}" target="_blank" style="text-decoration:none; font-size:1rem; background: #25D366; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold; border: 1px solid #128C7E;">
                    🟢 Share on WhatsApp
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            # Use Streamlit columns for Copy Link functionality
            col_copy, col_link = st.columns([1, 3])
            with col_copy:
                if st.button("📋 Copy Link", key="copy_link_btn"):
                    st.session_state["copied_link"] = share_link
                    st.toast("✅ Link copied! Use Ctrl+V to paste", icon="📋")
            
            with col_link:
                st.code(share_link, language=None)
            
        # Comments (CSV Storage) - Full Width
        st.markdown("") # Spacer
        with st.expander("💬 Comments"):
                # Load existing comments for this file
                import pandas as pd
                comment_file = "comments.csv"
                existing_comments = []
                
                if os.path.exists(comment_file):
                    try:
                        df_comments = pd.read_csv(comment_file)
                        # smooth handling if empty
                        if not df_comments.empty and "filename" in df_comments.columns:
                            file_comments = df_comments[df_comments["filename"] == selected_file_data["filename"]]
                            # Sort by time desc
                            if not file_comments.empty:
                                existing_comments = file_comments.to_dict("records")
                    except:
                        pass

                # Show existing
                if existing_comments:
                    st.markdown("**Recent Comments:**")
                    for c in existing_comments[-3:]: # Show last 3
                        st.text(f"📝 {c.get('timestamp', '')}: {c.get('text', '')}")

                # Add new
                new_comment = st.text_area("Leave a thought...", key=f"txt_{selected_file_data['filename']}", height=80)
                if st.button("Post Comment", key=f"btn_{selected_file_data['filename']}"):
                    if new_comment.strip():
                        import datetime
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        new_entry = {
                            "filename": selected_file_data["filename"],
                            "timestamp": ts,
                            "text": new_comment.strip()
                        }
                        
                        # Append to CSV
                        df_new = pd.DataFrame([new_entry])
                        if os.path.exists(comment_file):
                            df_new.to_csv(comment_file, mode='a', header=False, index=False)
                        else:
                            df_new.to_csv(comment_file, mode='w', header=True, index=False)
                            
                        st.success("Thank you for your reflection! (Saved)")
                        st.rerun()
                    else:
                        st.warning("Please write something first.")
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
            


# ============================================================
# CSS PATCH FOR RADIO BUTTONS & TEXT
# ============================================================
# Ensure radio labels and standard text are dark enough on Saffron bg
st.markdown("""
<style>
    /* Radio Button Labels */
    .stRadio label, div[data-testid="stRadio"] label {
        color: #4E342E !important; 
        font-weight: 600;
    }
    /* General text readability boost */
    .stMarkdown p, .stText {
        color: #3E2723;
    }
    
    /* FIX: Force Satsang Content Text to be Dark */
    .qa-box, .card, .content-text, .qa-q, .qa-a {
        color: #3E2723 !important;
    }
    .qa-box p, .card p {
        color: #3E2723 !important;
    }
    
    /* FIX: Satsang Button Contrast */
    .main-yt-btn, .yt-link {
        color: white !important;
        text-decoration: none !important;
    }
    /* FIX: Satsang Main Button Contrast (White on Red) */
    .main-yt-btn {
        color: white !important;
        text-decoration: none !important;
    }
    .main-yt-btn:visited {
        color: white !important;
    }
    
    /* Q&A Card Styling Override */
    .answer-number {
        background: #FF9933; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;
    }
    .answer-q {
        font-weight: 600; color: #8B0000; margin-left: 8px;
    }
    /* FIX: Revert yt-link to red text for contrast on pink background */
    .yt-link {
        color: #d32f2f !important;
        text-decoration: none !important;
    }
    .yt-link:visited {
        color: #d32f2f !important;
    }
    .answer-label {
        background: #4E342E; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-right: 8px; height: fit-content; margin-top: 4px;
    }
    .answer-body {
        display: flex; align-items: flex-start; margin-top: 8px; background: rgba(255,153,51,0.1); padding: 8px; border-radius: 8px;
    }
    .answer-a {
        color: #3E2723; font-size: 0.95rem; line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GLOBAL SIDEBAR & NAVIGATION
# ============================================================
# CRITICAL: Initialize default language to Hindi FIRST
if "view_lang" not in st.session_state:
    st.session_state["view_lang"] = "Hindi"

# Initialize default view
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "home"

with st.sidebar:
    # Home Button (Always visible)
    if st.button(get_text("sidebar_home", st.session_state["view_lang"]), use_container_width=True):
        st.session_state["current_view"] = "home"
        st.rerun()
    
    st.markdown("---")

    
    # --- HIDDEN AI SETTINGS (Using optimal defaults) ---
    # All AI config hidden from users - using Google Gemini with secrets
    provider = "Google Gemini"  # Fixed to best option
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    st.sidebar.markdown("---")
    
    # Language Toggle removed from sidebar - now on each page directly


# Capture current language for local usage
view_lang = st.session_state["view_lang"]

# ============================================================
# GLOBAL DATA & INDEX LOADING (Optimization)
# ============================================================
# Load data immediately so it's ready for any view
df, error_msg = load_data()
if error_msg:
    st.error(error_msg)
    st.stop()

# Build embeddings index globally (prevents delay on first search)
embed_texts = tuple(df["embed_text"].tolist())
with st.spinner("Building search index..."):
    model, doc_embeddings, model_error = build_index(provider, api_key, embed_texts)

if model_error:
    st.error(model_error)
    st.stop()
    
if st.session_state["current_view"] == "home":
    render_home_page(st.session_state["view_lang"])
    st.stop()
elif st.session_state["current_view"] == "satsang":
    render_satsang_page(st.session_state["view_lang"])
    st.stop()

# ============================================================
# Q/A APP LOGIC (Only runs if current_view == 'qa')


# ============================================================
# Q/A APP LOGIC (Only runs if current_view == 'qa')
# ============================================================
# ============================================================
# SEARCH SETTINGS (Simplified for End Users)
# ============================================================
# All technical settings are now hidden - using optimal defaults

# Hidden defaults - no UI exposed
search_mode = "Hybrid (Recommended)"  # Best balance
use_phrase_match = True
top_k = 40
short_query_requires_lex = True
semantic_weight = 0.75
HIGH_SEM_OVERRIDE = 0.62

lbl_translate = get_text("translate_toggle", view_lang)
enable_translation_bridge = st.sidebar.checkbox(lbl_translate, value=True)

lbl_debug = get_text("debug_mode", view_lang)
debug_mode = st.sidebar.checkbox(lbl_debug, value=False)

# Load data
# Load data and Index build (Moved to global scope)
lbl_loaded = get_text("conversations_loaded", view_lang, count=len(df))
st.sidebar.info(lbl_loaded)

# Session keys used for auto-search
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "trigger_search" not in st.session_state:
    st.session_state["trigger_search"] = False

# ============================================================
# Search Page Header: Title + Navigation Buttons + Language Toggle
# ============================================================
# Page Title - Above navigation
st.markdown(f'<h2 style="text-align: center; color: #8B0000; margin: 0 0 15px 0;">{get_text("ask_question_title", view_lang)}</h2>', unsafe_allow_html=True)

# Navigation Buttons on left, Language toggle on right
col_home, col_satsang, col_spacer, col_lang = st.columns([1, 1, 1, 1])
with col_home:
    if st.button(get_text("nav_home", view_lang), use_container_width=True, key="qa_home_btn"):
        st.session_state["current_view"] = "home"
        st.rerun()

with col_satsang:
    if st.button(get_text("nav_satsang", view_lang), use_container_width=True, key="qa_satsang_btn"):
        st.session_state["current_view"] = "satsang"
        st.rerun()

# col_spacer is empty - just for spacing
        
with col_lang:
    selected_lang = st.radio(
        "भाषा / Language",
        ["हिंदी", "English"],
        index=0 if st.session_state["view_lang"] == "Hindi" else 1,
        horizontal=True,
        key="search_lang_toggle"
    )
    new_lang = "Hindi" if selected_lang == "हिंदी" else "English"
    if new_lang != st.session_state["view_lang"]:
        st.session_state["view_lang"] = new_lang
        st.rerun()

# Update view_lang for rest of page
view_lang = st.session_state["view_lang"]

st.markdown("---")

# --- QUICK FILTERS (below navigation) ---
if view_lang == "English":
    kw_col = pick_english_source_column(df)
    keywords = extract_top_keywords(df, kw_col, top_n=30) if kw_col else []
    slicer_label = get_text("slicer_label_en", view_lang)
else:
    # Hindi keywords
    keywords = extract_hindi_keywords(df, top_n=30)
    slicer_label = get_text("slicer_label_hi", view_lang)

if keywords:
    with st.expander(slicer_label, expanded=False):
        chip_cols = st.columns(6)
        for i, kw in enumerate(keywords):
            with chip_cols[i % 6]:
                if st.button(kw, key=f"kw_{i}_{kw}", use_container_width=True):
                    st.session_state["query"] = kw
                    st.session_state["trigger_search"] = True
else:
    if view_lang == "English":
        st.caption("No English keyword column found for slicers.")

# Search Options
col_mode = st.columns([1])[0]

with col_mode:
    # Override the hidden default with user selection
    search_mode = st.radio(
        "खोज प्रकार / Search Type",
        ["🔀 स्मार्ट खोज / Smart Search", "🧠 अर्थ-आधारित / Meaning-Based", "📝 शब्द-आधारित / Word-Match"],
        index=0,
        horizontal=True,
        help="स्मार्ट खोज सबसे अच्छा परिणाम देता है / Smart Search gives best results"
    )
    # Map to internal format
    if "स्मार्ट" in search_mode or "Smart" in search_mode:
        search_mode = "Hybrid (Recommended)"
    elif "अर्थ" in search_mode or "Meaning" in search_mode:
        search_mode = "Semantic Only"
    else:
        search_mode = "Literal Only"

st.markdown("")
lbl_ask = get_text("ask_question_title", view_lang)
query = st.text_input(lbl_ask, placeholder="e.g., I am Sick / छीन लेना / नाम जप नहीं हो रहा", key="query"
)


# ============================================================
# 7) SEARCH
# ============================================================
c_search, c_browse, c_size = st.columns([3, 1, 1])
with c_search:
    lbl_search = get_text("search_btn", view_lang)
    search_clicked = st.button(lbl_search, type="primary", use_container_width=True)
with c_browse:
    lbl_browse = get_text("browse_btn", view_lang)
    browse_clicked = st.button(lbl_browse, use_container_width=True)
with c_size:
    lbl_size = get_text("page_size", view_lang)
    page_size = st.selectbox(lbl_size, [5, 10, 20, 50], index=0, label_visibility="collapsed")

auto_clicked = st.session_state.get("trigger_search", False)

# Handle Browse Mode
if browse_clicked:
    st.session_state["mode"] = "browse"
    st.session_state["page"] = 1
    # Create artificial results list (index, score=1, semantic=0, lexical=0, method="Browse")
    # We use index same as dataframe index
    all_indices = list(range(len(df)))
    browse_results = [(i, 0.0, 0.0, 0.0, "Browse") for i in all_indices]
    st.session_state["search_results"] = browse_results
    st.session_state["search_executed"] = True
    st.rerun()

# Handle Search Mode
if (search_clicked or auto_clicked) and query:
    st.session_state["mode"] = "search"
    # reset trigger
    st.session_state["trigger_search"] = False
    # reset to page 1 on new search
    st.session_state["page"] = 1

    st.markdown("---")

    # Translation bridge (English -> Hindi), used for semantic and lexical
    query_hi = translate_to_hindi_if_english(query, api_key) if enable_translation_bridge else query
    if debug_mode and query_hi != query:
        st.caption(f"Translated query (Hindi): {query_hi}")

    # Tokenization for weighting logic (use original query tokens)
    q_toks = tokenize_hi_en(query)
    phrase_boost = use_phrase_match

    # Lexical uses BOTH original + translated query; take max lexical score
    def row_lex(i: int) -> float:
        text = df.iloc[i]["lex_text"]
        s1 = lexical_score(query, text, phrase_boost=phrase_boost)
        s2 = lexical_score(query_hi, text, phrase_boost=phrase_boost) if query_hi != query else 0.0
        return max(s1, s2)

    # --- semantic candidates (Top-K), compute using BOTH queries and take max similarity ---
    semantic_candidates = []
    if doc_embeddings is not None and len(doc_embeddings) > 0:
        # Build query embeddings
        if provider == "Google Gemini":
            q_embed_1 = model.encode_query(query)
            q_embed_2 = model.encode_query(query_hi) if query_hi != query else None
        else:
            q_embed_1 = model.encode([query])
            q_embed_2 = model.encode([query_hi]) if query_hi != query else None

        sim_1 = cosine_similarity(q_embed_1, doc_embeddings)[0] if q_embed_1 is not None and q_embed_1.size > 0 else None
        sim_2 = cosine_similarity(q_embed_2, doc_embeddings)[0] if q_embed_2 is not None and q_embed_2.size > 0 else None

        if sim_1 is None and sim_2 is None:
            sim = None
        elif sim_2 is None:
            sim = sim_1
        elif sim_1 is None:
            sim = sim_2
        else:
            sim = np.maximum(sim_1, sim_2)

        if sim is not None:
            top_idx = np.argsort(sim)[::-1][:top_k]
            semantic_candidates = [(int(i), float(sim[i])) for i in top_idx]

    # --- weights ---
    # short queries: slightly more lexical influence
    if len(q_toks) <= 2:
        sem_w, lex_w = 0.55, 0.45
    else:
        sem_w, lex_w = semantic_weight, 1.0 - semantic_weight

    # English-only queries: semantic should dominate (prevents stopword-based false matches)
    if q_toks and (not has_hindi_token(q_toks)):
        sem_w, lex_w = 0.80, 0.20

    results = []  # (i, final, sem, lex, method)

    if search_mode == "Literal Only":
        for i in range(len(df)):
            ls = row_lex(i)
            if ls > 0:
                results.append((i, ls, 0.0, ls, "Literal"))

    elif search_mode == "Semantic Only":
        for i, ss in semantic_candidates:
            ls = row_lex(i)
            if short_query_requires_lex and len(q_toks) <= 2 and ls == 0 and ss < HIGH_SEM_OVERRIDE:
                continue
            results.append((i, ss, ss, ls, "Semantic"))

    else:
        # Hybrid
        for i, ss in semantic_candidates:
            ls = row_lex(i)

            # Short query: require lexical grounding unless semantic is very high
            if short_query_requires_lex and len(q_toks) <= 2 and ls == 0 and ss < HIGH_SEM_OVERRIDE:
                continue

            final = (sem_w * ss) + (lex_w * ls)
            method = "Hybrid" if ls > 0 else "Semantic"
            results.append((i, final, ss, ls, method))

    results.sort(key=lambda x: x[1], reverse=True)
    st.session_state["search_results"] = results
    st.session_state["search_executed"] = True


# --- Retrieve Results from Session State ---
if st.session_state.get("search_executed", False):
    results = st.session_state.get("search_results", [])
else:
    results = []
    
current_mode = st.session_state.get("mode", "search")

# ============================================================
# SEMANTIC VISUALIZATION HELPERS
# ============================================================
def extract_key_concepts(text, max_concepts=5):
    """Extract key concepts/phrases from text, filtering stopwords"""
    import re
    # Clean and split
    words = re.findall(r'\b[a-zA-Z\u0900-\u097F]{3,}\b', text.lower())
    # Filter common stopwords
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'being', 'there', 'this', 'that', 'with', 'they', 'from', 'what', 'which', 'when', 'how', 'why', 'who', 'will', 'would', 'could', 'should', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'where', 'than', 'too', 'very', 'just', 'only', 'now', 'की', 'के', 'का', 'में', 'है', 'हैं', 'को', 'से', 'पर', 'और', 'यह', 'इस', 'कि', 'जो', 'तो', 'भी', 'ने', 'एक', 'हो', 'या', 'था', 'थे', 'थी', 'अपने', 'किया', 'करना', 'किसी', 'कर', 'करें', 'होता', 'होती', 'होते', 'रहा', 'रही', 'रहे', 'गया', 'गई', 'गए', 'जाता', 'जाती', 'जाते'}
    filtered = [w for w in words if w not in stopwords]
    # Get unique, preserving order
    seen = set()
    unique = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:max_concepts]

def compute_sentence_relevance(query, text, embedder=None, embeddings_cache=None):
    """Split text into sentences and compute relevance score for each"""
    import re
    # Split by Hindi and English sentence delimiters
    sentences = re.split(r'[।.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return []
    
    # Simple keyword-based relevance (works without embeddings)
    query_words = set(re.findall(r'\b[a-zA-Z\u0900-\u097F]{2,}\b', query.lower()))
    
    scored = []
    for sent in sentences[:10]:  # Limit to first 10 sentences
        sent_words = set(re.findall(r'\b[a-zA-Z\u0900-\u097F]{2,}\b', sent.lower()))
        overlap = len(query_words & sent_words)
        relevance = min(1.0, overlap / max(1, len(query_words)))
        scored.append((sent, relevance))
    
    return sorted(scored, key=lambda x: x[1], reverse=True)

def generate_relevance_heatmap_html(sentences_with_scores, max_sentences=5):
    """Generate HTML with color-coded sentences based on relevance"""
    html_parts = []
    for sent, score in sentences_with_scores[:max_sentences]:
        # Color gradient: low relevance = light, high relevance = bright yellow
        if score > 0.5:
            bg_color = "#FFEB3B"  # Bright yellow - high match
            border = "2px solid #FFC107"
        elif score > 0.2:
            bg_color = "#FFF9C4"  # Light yellow - moderate match
            border = "1px solid #FFEB3B"
        else:
            bg_color = "#FFFDE7"  # Very light - low match
            border = "1px solid #FFF9C4"
        
        html_parts.append(f'''
        <div style="padding: 8px 12px; margin: 4px 0; background: {bg_color}; border: {border}; border-radius: 8px;">
            <span style="font-size: 0.8rem; color: #666;">({score:.0%})</span> {sent[:200]}{'...' if len(sent) > 200 else ''}
        </div>''')
    
    return "".join(html_parts)

def render_result_card(idx_num, row, final, sem, lex, method, show_translated_answer: bool, debug_mode: bool, view_lang: str):
    # Determine text based on language selection
    if view_lang == "English":
        q_text = str(row.get("Translated Question", "")).strip() or str(row.get("Question", "")).strip()
        a_text = str(row.get("Translated Answer", "")).strip() or str(row.get("Answer", "")).strip()
    else:
        q_text = str(row.get("Question", "")).strip()
        a_text = str(row.get("Answer", "")).strip()

    # Light formatting: keep answer clean but readable
    # (Do not remove Hindi punctuation; keep line breaks)
    safe_a = a_text

    # Build match reasoning HTML (if applicable)
    match_reasoning_html = ""
    if method != "Browse":
        query = st.session_state.get("query", "")
        
        # Context Check: Where did the match happen?
        q_raw = str(row.get("Question", "")).lower()
        a_raw = str(row.get("Answer", "")).lower()
        q_display = str(row.get("Question", ""))
        a_display = str(row.get("Answer", ""))
        query_words = [w for w in query.lower().split() if len(w) > 2]
        
        found_in_q = [w for w in query_words if w in q_raw]
        found_in_a = [w for w in query_words if w in a_raw]
        
        # Highlight matching words in Q and A text
        def highlight_keywords(text, keywords):
            """Wrap matching keywords with highlight span"""
            highlighted = text
            for kw in keywords:
                # Case-insensitive replacement with highlight
                import re
                pattern = re.compile(f'({re.escape(kw)})', re.IGNORECASE)
                highlighted = pattern.sub(r'<mark style="background: #FFEB3B; padding: 1px 3px; border-radius: 3px;">\1</mark>', highlighted)
            return highlighted
        
        # Apply highlighting
        if found_in_q:
            q_text = highlight_keywords(q_text, found_in_q)
        if found_in_a:
            safe_a = highlight_keywords(safe_a, found_in_a)
        
        context_str = ""
        context_str_hi = ""
        matched_text_snippet = ""
        
        # Extract a relevant snippet from Q or A that contains query words
        if found_in_q or found_in_a:
            # Find which text has more matches
            if len(found_in_q) >= len(found_in_a) and found_in_q:
                # Show snippet from Question
                snippet_src = q_display[:150] + "..." if len(q_display) > 150 else q_display
                matched_text_snippet = f'📄 **Matched in Question:** "{snippet_src}"'
            elif found_in_a:
                # Show snippet from Answer  
                snippet_src = a_display[:150] + "..." if len(a_display) > 150 else a_display
                matched_text_snippet = f'📄 **Matched in Answer:** "{snippet_src}"'
        
        if found_in_q and found_in_a:
            context_str = f"Keywords found in **Question & Answer**: {', '.join(set(found_in_q + found_in_a))}"
            context_str_hi = f"प्रश्न और उत्तर दोनों में शब्द मिले: {', '.join(set(found_in_q + found_in_a))}"
        elif found_in_q:
            context_str = f"Keywords found in **Question**: {', '.join(found_in_q)}"
            context_str_hi = f"प्रश्न में शब्द मिले: {', '.join(found_in_q)}"
        elif found_in_a:
            context_str = f"Keywords found in **Answer**: {', '.join(found_in_a)}"
            context_str_hi = f"उत्तर में शब्द मिले: {', '.join(found_in_a)}"
        else:
            # For pure semantic matches, show what text was compared
            snippet_src = q_display[:100] + "..." if len(q_display) > 100 else q_display
            matched_text_snippet = f'📄 **Compared with Question:** "{snippet_src}"'
        
        # Score breakup for display
        score_breakup = ""
        score_breakup_hi = ""
        if method == "Hybrid" and lex > 0:
            score_breakup = f"📊 **Score Breakup:** Total={final:.0%} (Semantic: {sem:.0%} + Lexical: {lex:.0%})"
            score_breakup_hi = f"📊 **स्कोर विवरण:** कुल={final:.0%} (अर्थ: {sem:.0%} + शब्द: {lex:.0%})"
        
        # Build query display
        query_display = f'🔍 **Your Search:** "{query}"'
        query_display_hi = f'🔍 **आपकी खोज:** "{query}"'
        
        # Highlight matched words in display
        matched_words_html = ""
        if found_in_q or found_in_a:
            all_matched = list(set(found_in_q + found_in_a))
            highlighted_words = " ".join([f'<mark style="background: #FFEB3B; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{w}</mark>' for w in all_matched])
            matched_words_html = f"<br><br>🎯 <b>Matched Words:</b> {highlighted_words}"
            matched_words_hi = f"<br><br>🎯 <b>मिलान शब्द:</b> {highlighted_words}"
        else:
            matched_words_hi = ""
            
        # Create detailed user-friendly explanation
        if method == "Semantic" or (method == "Hybrid" and lex == 0):
            reason_icon = "🧠"
            if view_lang == "Hindi":
                reason_text = f"{query_display_hi}\\n\\n**अर्थ-आधारित खोज** ({sem:.0%} समानता)\\n\\nआपके प्रश्न का अर्थ इससे मेल खाता है।"
                if matched_text_snippet: 
                    reason_text += f"\\n\\n📄 **प्रश्न से मिलान:** \"{q_display[:120]}...\""
            else:
                reason_text = f"{query_display}\\n\\n**Meaning-Based Match** ({sem:.0%} similarity)\\n\\nYour query's meaning aligns with this result."
                if context_str: reason_text += f"\\n{context_str}"
                if matched_text_snippet: reason_text += f"\\n\\n{matched_text_snippet}"
                
        elif method == "Literal Only" or (method == "Hybrid" and lex > 0 and sem < 0.3):
            reason_icon = "📝"
            if view_lang == "Hindi":
                reason_text = f"{query_display_hi}\\n\\n**शब्द-आधारित खोज**\\n\\n{context_str_hi if context_str_hi else 'सीधा शब्द मिलान'}"
            else:
                reason_text = f"{query_display}\\n\\n**Keyword Match**\\n\\n{context_str if context_str else 'Direct keyword match'}"
                if matched_text_snippet: reason_text += f"\\n\\n{matched_text_snippet}"
        else:  # Hybrid
            reason_icon = "🔀"
            if view_lang == "Hindi":
                reason_text = f"{query_display_hi}\\n\\n**स्मार्ट खोज** ({final:.0%})\\n\\n✓ अर्थ मेल खाता है ({sem:.0%})"
                if context_str_hi: reason_text += f"\\n✓ {context_str_hi}"
                if score_breakup_hi: reason_text += f"\\n\\n{score_breakup_hi}"
            else:
                reason_text = f"{query_display}\\n\\n**Smart Match** ({final:.0%})\\n\\n✓ Meaning matches ({sem:.0%})"
                if context_str: reason_text += f"\\n✓ {context_str}"
                if score_breakup: reason_text += f"\\n\\n{score_breakup}"
                if matched_text_snippet: reason_text += f"\\n\\n{matched_text_snippet}"
        
        # Convert markdown to simple HTML
        reason_html = reason_text.replace("\\n\\n", "<br><br>").replace("\\n", "<br>").replace("**", "")
        
        # Add highlighted matched words
        if view_lang == "Hindi" and matched_words_hi:
            reason_html += matched_words_hi
        elif matched_words_html:
            reason_html += matched_words_html
            
        match_reasoning_html = f"""
  <details style="margin-top: 10px; padding: 8px; background: rgba(255,153,51,0.1); border-radius: 8px; cursor: pointer;">
    <summary style="font-weight: 600; color: #8B0000; font-size: 0.9rem;">{reason_icon} Why this answer? / यह उत्तर क्यों?</summary>
    <div style="margin-top: 8px; font-size: 0.85rem; color: #4E342E; line-height: 1.5;">{reason_html}</div>
  </details>"""
    
    st.markdown(
        f'''<div class="answer-card">
  <div class="answer-header">
    <div class="answer-number">#{idx_num}</div>
    <div class="answer-q">{q_text}</div>
  </div>
  
  <div class="answer-divider"></div>
  
  <div class="answer-body">
    <div class="answer-label">Ans</div>
    <div class="answer-a">{safe_a}</div>
  </div>
  {match_reasoning_html}
</div>''',
        unsafe_allow_html=True
    )

    # If debug mode, show metadata
    if debug_mode and method != "Browse":
        st.markdown(
            f"<div class='answer-meta'>Final={final:.2f} | Semantic={sem:.2f} | Lexical={lex:.2f} | Mode={method}</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

if not results:
    if st.session_state.get("search_executed", False):
         st.info("No relevant results found.")
    elif current_mode == "browse":
         st.info("No records loaded.")
else:
    # --- Pagination ---
    total = len(results)
    total_pages = max(1, int(np.ceil(total / page_size)))
    page = st.session_state["page"]
    page = max(1, min(page, total_pages))
    st.session_state["page"] = page

    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_slice = results[start:end]

    # Render results
    st.markdown("---")
    
    # Calculate starting number based on page
    start_num = start + 1
    
    for relative_idx, (i, final, sem, lex, method) in enumerate(page_slice):
        row = df.iloc[i]
    # Pass show_translated_answer=False since we removed the checkbox
        render_result_card(start_num + relative_idx, row, final, sem, lex, method, False, debug_mode, view_lang)

    # Controls row
    st.markdown("---")
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        lbl_prev = get_text("prev", view_lang)
        if st.button(lbl_prev, disabled=(page <= 1)):
            st.session_state["page"] = page - 1
            st.rerun()
    with c2:
        lbl_next = get_text("next", view_lang)
        if st.button(lbl_next, disabled=(page >= total_pages)):
            st.session_state["page"] = page + 1
            st.rerun()
    with c3:
        lbl_showing = get_text("showing_results", view_lang, start=start+1, end=end, total=total)
        lbl_page_num = get_text("page_num", view_lang, page=page, total=total_pages)
        st.caption(f"{lbl_showing}  |  {lbl_page_num}")
    with c4:
        lbl_goto = get_text("go_to", view_lang)
        goto = st.number_input(lbl_goto, min_value=1, max_value=total_pages, value=page, label_visibility="collapsed")
        if goto != page:
            st.session_state["page"] = int(goto)
            st.rerun()
