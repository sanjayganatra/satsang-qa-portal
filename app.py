import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import time

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Gemini (same library style as your original code)
import google.generativeai as genai


# ============================================================
# 1) PAGE CONFIG
# ============================================================
PAGE_TITLE = "Priyakunj Q/A Portal"
ICON = "🙏"
st.set_page_config(page_title=PAGE_TITLE, page_icon=ICON)

# --- CSS ---
CUSTOM_CSS = """
<style>
/* Overall */
.block-container { padding-top: 1.5rem; }

/* Card styling */
.answer-card {
  background-color: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px 20px 24px 20px;
  margin-bottom: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.answer-header {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}
.answer-number {
  background-color: #e8f0fe; /* Light blue accent */
  color: #1a73e8;
  font-size: 0.8rem;
  font-weight: 700;
  padding: 4px 8px;
  border-radius: 6px;
  height: fit-content;
  min-width: 36px;
  text-align: center;
}
.answer-q {
  font-size: 1.15rem;
  font-weight: 600;
  color: #202124;
  line-height: 1.5;
  margin-top: -2px; /* slight visual adjustment */
}
.answer-divider {
  border-top: 1px solid #f1f3f4;
  margin: 0 0 12px 48px; /* Indented to match text */
}

/* Answer Section */
.answer-body {
  display: flex;
  gap: 12px;
}
.answer-label {
  font-size: 0.75rem;
  font-weight: 700;
  color: #5f6368;
  background-color: #f1f3f4;
  padding: 2px 6px;
  border-radius: 4px;
  height: fit-content;
  margin-top: 4px;
  min-width: 36px;
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.answer-a {
  font-size: 1.0rem;
  line-height: 1.6;
  color: #3c4043;
  white-space: pre-wrap;
}

.answer-meta {
  font-size: 0.75rem;
  color: #9aa0a6;
  margin-top: 12px;
  padding-left: 48px;
}

/* Slicer chips */
.slicer-wrap {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 6px;
}

/* Mobile Optimizations */
@media (max-width: 600px) {
  .answer-card {
    padding: 16px;
    margin-bottom: 12px;
  }
  .answer-q {
    font-size: 1.05rem;
  }
  .answer-a {
    font-size: 0.95rem;
  }
  .answer-header {
    gap: 8px;
    margin-bottom: 12px;
  }
  .answer-number {
    min-width: 28px;
    font-size: 0.75rem;
  }
  .answer-divider {
     margin: 0 0 12px 0; /* Full width divide on mobile */
  }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# 2) CLEANING / NORMALIZATION
# ============================================================
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
# 6) UI
# ============================================================
c_title, c_lang = st.columns([3, 1])
with c_title:
    st.title(f"{ICON} {PAGE_TITLE}")
with c_lang:
    st.caption("View Language")
    view_lang = st.radio("View Language", ["Hindi", "English"], horizontal=True, label_visibility="collapsed")

st.sidebar.header("Settings")
page_size = st.sidebar.selectbox("Results per page", [5, 10, 20, 50], index=0)
if "page" not in st.session_state:
    st.session_state["page"] = 1

provider = st.sidebar.radio("AI Provider", ["Local (Free)", "Google Gemini"], index=1)

api_key = st.secrets.get("GOOGLE_API_KEY", "")
if provider == "Google Gemini":
    if api_key:
        st.sidebar.success("API Key loaded from secrets")
    else:
        api_key = st.sidebar.text_input("Google API Key", type="password")

search_mode = st.sidebar.radio(
    "Search Mode",
    ["Hybrid (Recommended)", "Semantic Only", "Literal Only"],
    index=0
)
use_phrase_match = st.sidebar.checkbox("Prefer exact phrase for short queries", value=True)
top_k = st.sidebar.slider("Semantic candidates (Top-K)", 10, 200, 40, step=10)

short_query_requires_lex = st.sidebar.checkbox("Short query must match (expanded) tokens", value=True)
semantic_weight = st.sidebar.slider("Hybrid weight (semantic)", 0.0, 1.0, 0.75, 0.05)

HIGH_SEM_OVERRIDE = st.sidebar.slider("Short-query semantic override threshold", 0.0, 1.0, 0.62, 0.01)

enable_translation_bridge = st.sidebar.checkbox("Translate English query to Hindi for search", value=True)

show_translated_answer = st.sidebar.checkbox("Show Translated Answer", value=False)
debug_mode = st.sidebar.checkbox("Show Debug Info", value=False)

# Load data
df, error_msg = load_data()
if error_msg:
    st.error(error_msg)
    st.stop()

st.sidebar.info(f"Conversations Loaded: {len(df)}")

# Build embeddings index
embed_texts = tuple(df["embed_text"].tolist())
with st.spinner("Building search index..."):
    model, doc_embeddings, model_error = build_index(provider, api_key, embed_texts)

if model_error:
    st.error(model_error)
    st.stop()

# --- SLICERS (PowerBI-like) ---
kw_col = pick_english_source_column(df)
keywords = extract_top_keywords(df, kw_col, top_n=30) if kw_col else []

# Session keys used for auto-search
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "trigger_search" not in st.session_state:
    st.session_state["trigger_search"] = False

if keywords:
    with st.expander("Quick Filters (English Keywords)", expanded=False):
        # Render chips in columns to simulate horizontal slicers
        # (Streamlit does not have native PowerBI slicers; this is the closest UX.)
        chip_cols = st.columns(6)  # tune 5–8 depending on your layout
        for i, kw in enumerate(keywords):  # show all keywords inside expander
            with chip_cols[i % 6]:
                if st.button(kw, key=f"kw_{i}_{kw}", use_container_width=True):
                    st.session_state["query"] = kw
                    st.session_state["trigger_search"] = True
else:
    st.caption("No English keyword column found for slicers. Add a column named 'English Text' (recommended).")

query = st.text_input("Ask a question:", placeholder="e.g., I am Sick / छीन लेना / नाम जप नहीं हो रहा", key="query"
)


# ============================================================
# 7) SEARCH
# ============================================================
c_search, c_browse = st.columns([4, 1])
with c_search:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with c_browse:
    browse_clicked = st.button("Browse All", use_container_width=True)

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
        render_result_card(start_num + relative_idx, row, final, sem, lex, method, show_translated_answer, debug_mode, view_lang)

    # Controls row
    st.markdown("---")
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        if st.button("◀ Prev", disabled=(page <= 1)):
            st.session_state["page"] = page - 1
            st.rerun()
    with c2:
        if st.button("Next ▶", disabled=(page >= total_pages)):
            st.session_state["page"] = page + 1
            st.rerun()
    with c3:
        st.caption(f"Showing {start+1}-{end} of {total} results  |  Page {page} of {total_pages}")
    with c4:
        goto = st.number_input("Go to", min_value=1, max_value=total_pages, value=page, label_visibility="collapsed")
        if goto != page:
            st.session_state["page"] = int(goto)
            st.rerun()
