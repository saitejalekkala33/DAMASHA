import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import os
import nltk

from models.model import RoBERTaModernBERTCRF
from models.style_features import StyleFeatureExtractor
from utils.preprocessing import clean_text

# Download NLTK resources (only if not already present)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Model and tokenizer names
ROBERTA_MODEL_NAME = 'roberta-base'
MODERNBERT_MODEL_NAME = 'answerdotai/ModernBERT-Base'
MODEL_PATH = r"C:\teja\AAAI_2026\RoBERTa_ModernBERT_CRF.pth"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit UI setup
st.set_page_config(page_title="AI Text Detector", layout="centered")

st.markdown(
    """
    <style>
        .main { background-color: #f5f6fa; }
        .stTextArea textarea { font-size: 1.1em; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 1.1em; }
        .result-word { font-weight: bold; padding: 2px 4px; border-radius: 3px; }
        .ai { background-color: #ffcccc; }
        .human { background-color: #ccffcc; }
        .word-count { font-size: 1.1em; margin-bottom: 10px; }
        .warning { color: #b30000; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🕵️‍♂️ AI Text Detector")
st.write("Paste your text below and click **Detect** to see which parts are likely AI-generated.")

# Session state for results
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'words_plot' not in st.session_state:
    st.session_state.words_plot = []

# Text input
input_text = st.text_area("Enter your text here (max 350 words)", height=300)
words = input_text.split()
word_count = len(words)

# Show word count and warning if needed
st.markdown(f'<div class="word-count">Word count: {word_count}/350</div>', unsafe_allow_html=True)
if word_count > 350:
    st.markdown('<div class="warning">⚠️ Only the first 350 words will be analyzed.</div>', unsafe_allow_html=True)
    words = words[:350]
    input_text = ' '.join(words)
    word_count = 350

detect_disabled = word_count < 30

if st.button("Detect", disabled=detect_disabled):
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model checkpoint not found at {MODEL_PATH}")
            st.stop()
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME, add_prefix_space=True)
        language_model = AutoModel.from_pretrained(ROBERTA_MODEL_NAME, use_safetensors=True)
        if hasattr(language_model, 'is_meta') and language_model.is_meta:
            language_model = language_model.to_empty(device=device)
        else:
            language_model = language_model.to(device)
        style_extractor = StyleFeatureExtractor(tokenizer, language_model, device)
        model = RoBERTaModernBERTCRF(ROBERTA_MODEL_NAME, MODERNBERT_MODEL_NAME, num_labels=2)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # Remap CRF keys if needed (for torchcrf version compatibility)
        crf_key_map = {
            'crf.trans_matrix': 'crf.transitions',
            'crf.start_trans': 'crf.start_transitions',
            'crf.end_trans': 'crf.end_transitions',
            'crf.transitions': 'crf.trans_matrix',
            'crf.start_transitions': 'crf.start_trans',
            'crf.end_transitions': 'crf.end_trans',
        }
        state_dict = checkpoint['model_state_dict']
        # If old keys present, map to new
        if any(k in state_dict for k in ['crf.trans_matrix', 'crf.start_trans', 'crf.end_trans']):
            for old, new in crf_key_map.items():
                if old in state_dict:
                    state_dict[new] = state_dict.pop(old)
        # If new keys present, map to old (for older torchcrf)
        elif any(k in state_dict for k in ['crf.transitions', 'crf.start_transitions', 'crf.end_transitions']):
            for new, old in crf_key_map.items():
                if new in state_dict:
                    state_dict[old] = state_dict.pop(new)
        model.load_state_dict(state_dict)
        if hasattr(model, 'is_meta') and model.is_meta:
            model = model.to_empty(device=device)
        else:
            model = model.to(device)
        model.eval()

        # Preprocess text
        text_cleaned = clean_text(input_text)
        words = text_cleaned.split()
        encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt", max_length=512, truncation=True, padding=True)
        word_ids = encoding.word_ids(batch_index=0)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        style_features = style_extractor.get_style_features(words, word_ids).unsqueeze(0)

        with torch.no_grad():
            predictions, _ = model(input_ids, attention_mask, style_features)

        st.session_state.predictions = predictions[0]
        st.session_state.words_plot = words

    except Exception as e:
        st.error(f"Error processing text: {str(e)}")

# Display results
if st.session_state.predictions and st.session_state.words_plot:
    word_level_preds = []
    prev_word_id = None
    # Use the encoding from the last run
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME, add_prefix_space=True)
    encoding = tokenizer(st.session_state.words_plot, is_split_into_words=True, return_tensors="pt", max_length=512, truncation=True, padding=True)
    for idx, word_id in enumerate(encoding.word_ids(batch_index=0)):
        if word_id is not None and word_id != prev_word_id:
            word_level_preds.append((st.session_state.words_plot[word_id], st.session_state.predictions[idx]))
            prev_word_id = word_id

    # Highlight AI/Human words
    highlighted_output = ""
    for word, label in word_level_preds:
        css_class = "ai" if label == 1 else "human"
        highlighted_output += f'<span class="result-word {css_class}">{word}</span> '
    st.markdown(highlighted_output, unsafe_allow_html=True)