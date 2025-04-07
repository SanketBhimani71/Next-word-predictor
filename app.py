import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import load_model
import streamlit as st
from langdetect import detect
import fasttext
import json
import os

# Streamlit header and styling
st.header('Next Word Predictor')

st.markdown("""
    <style>
    .search-bar {
        width: 60%;
        margin: 0 auto;
        padding: 10px;
        font-size: 18px;
    }
    .suggestions {
        width: 60%;
        margin: 5px auto;
        text-align: left;
        font-size: 16px;
    }
    .suggestion-btn {
        display: inline-block;
        padding: 8px 16px;
        margin: 5px;
        background-color: #f8f9fa;
        border: 1px solid #dfe1e5;
        border-radius: 4px;
        cursor: pointer;
        color: #202124;
    }
    .suggestion-btn:hover {
        background-color: #e8eaed;
        border: 1px solid #dadce0;
    }
    </style>
""", unsafe_allow_html=True)


# Language detection function
def detect_language(text):
    """
    Args:
        text: User input text
    Returns:
        lang: Detected language code (e.g., 'en' for English, 'es' for Spanish)
    """
    try:
        model = fasttext.load_model('lid.176.bin')
        print("here")
        prediction = model.predict(text)
        print(prediction)
        label = prediction[0][0]
        language_code = label.replace('__label__', '')
        print(language_code)  
        return language_code
    except:
        return 'unknown'

# Tokenization function
def tokenization_word_index(corpus):
    """
    Args:
        corpus: All text data
    Returns:
        tokenizer: Tokenizer object
    """
    tokenizer = Tokenizer(lower = False)
    tokenizer.fit_on_texts([corpus])
    return tokenizer

# Sequence generation function
def text_sequences(tokenizer, text):
    """
    Args:
        tokenizer: Tokenizer object
        text: Text data
    Returns:
        input_sequences: List of word sequences
    """
    input_sequences = []
    for sentence in text.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i+1])
    return input_sequences

# Model creation function
def model_creation(vocab_size, max_len):
    """
    Args:
        vocab_size: Size of vocabulary
        max_len: Maximum length of a sentence in data
    Returns:
        model: Compiled Keras model
    """
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_len-1))
    model.add(LSTM(150))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Model training function
def train_model(model, input_sequences, vocab_size):
    """
    Args:
        model: Pre-trained model
        input_sequences: List of word sequences
        vocab_size: Size of vocabulary
    Returns:
        model: Trained model
        max_len: Maximum sequence length
    """
    if not input_sequences:
        return None, 0
    max_len = max([len(x) for x in input_sequences])
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
    X = padded_input_sequences[:, :-1]
    y = padded_input_sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    model.fit(X, y, epochs=50, verbose=0)  # Reduced epochs for faster training
    return model, max_len

# Check new words percentage
def check_new_words_percentage(text, tokenizer):
    """
    Args:
        text: Text data
        tokenizer: Tokenizer object
    Returns:
        float: Percentage of new words
    """
    words = text.strip().split()
    if not words:
        return 0.0
    new_words = [word for word in words if word not in tokenizer.word_index]
    return len(new_words) / len(words) if words else 0.0

# Load dataset from JSON
def load_dataset(lang):
    """
    Args:
        lang: Language code
    Returns:
        data: Text data from JSON file
    """
    dataset_file = f"dataset_{lang}.json"
    if os.path.exists(dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            return "\n".join(dataset.get("sentences", []))
    return ""

# Save dataset to JSON
def save_dataset(lang, text):
    """
    Args:
        lang: Language code
        text: New text to append
    """
    dataset_file = f"dataset_{lang}.json"
    if os.path.exists(dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = {"lang": lang, "sentences": []}
    
    dataset["sentences"].append(text.strip())
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

# Update and train model with new text
def update_and_train_model(text, model, tokenizer, input_sequences, max_len, lang):
    """
    Args:
        text: Text data
        model: Pre-trained model (or None for new language)
        tokenizer: Tokenizer object
        input_sequences: List of word sequences
        max_len: Maximum sequence length
        lang: Detected language code
    Returns:
        model: Updated/trained model
        tokenizer: Updated tokenizer
        input_sequences: Updated sequences
        max_len: Updated maximum length
    """
    if not text.strip():
        return model, tokenizer, input_sequences, max_len

    # Save new text to JSON dataset
    save_dataset(lang, text)
    data = load_dataset(lang)

    # Update tokenizer with full dataset
    tokenizer = tokenization_word_index(data)
    new_sequences = text_sequences(tokenizer, data)
    input_sequences.extend(new_sequences)
    if not input_sequences:
        return model, tokenizer, input_sequences, max_len

    vocab_size = len(tokenizer.word_index) + 1
    max_len = max([len(x) for x in input_sequences])

    # If no model exists (new language) or vocab size increased significantly, create and train
    if model is None or vocab_size > model.layers[0].input_dim:
        model = model_creation(vocab_size, max_len)
        model, max_len = train_model(model, input_sequences, vocab_size)
        model.save(f'model_{lang}.h5')
    else:
        # Retrain with updated sequences if new words exceed threshold
        new_words_ratio = check_new_words_percentage(text, tokenizer)
        if new_words_ratio >= 0.6:
            model, max_len = train_model(model, input_sequences, vocab_size)
            model.save(f'model_{lang}.h5')

    return model, tokenizer, input_sequences, max_len

# Predict next words
def predict_next_words(text, model, tokenizer, max_len, num_suggestions=4):
    """
    Args:
        text: Text data
        model: Trained model
        tokenizer: Tokenizer object
        max_len: Maximum sequence length
        num_suggestions: Number of suggestions
    Returns:
        top_words: List of suggested words
    """
    if not text.strip() or model is None:
        return []

    token_text = tokenizer.texts_to_sequences([text])[0]
    if not token_text:
        return None

    vocab_size = model.layers[0].input_dim
    token_text = [min(idx, vocab_size - 1) for idx in token_text]

    padded_token_text = pad_sequences([token_text], maxlen=max_len-1, padding='pre')
    predictions = model.predict(padded_token_text, verbose=0)[0]

    top_indices = np.argsort(predictions)[-num_suggestions:][::-1]
    top_words = []
    for index in top_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_words.append(word)
                break
    return top_words

# Main flow function
@st.cache_resource
def main_flow(lang, _cache_key=0):
    """
    Args:
        lang: Detected language code
        _cache_key: Cache status
    Returns:
        model: Trained model
        tokenizer: Tokenizer object
        input_sequences: List of word sequences
        max_len: Maximum sequence length
    """
    model_file = f"model_{lang}.h5"
    dataset_file = f"dataset_{lang}.json"

    # Load existing model if available
    try:
        model = load_model(model_file)
    except:
        model = None

    # Load dataset
    data = load_dataset(lang)
    if not data.strip():
        save_dataset(lang, "")  # Initialize empty dataset
        return None, None, [], 0

    tokenizer = tokenization_word_index(data)
    input_sequences = text_sequences(tokenizer, data)
    if not input_sequences:
        return None, None, [], 0

    vocab_size = len(tokenizer.word_index) + 1
    max_len = max([len(x) for x in input_sequences])

    # Train model if it doesn't exist
    if model is None:
        model = model_creation(vocab_size, max_len)
        model, max_len = train_model(model, input_sequences, vocab_size)
        model.save(model_file)

    return model, tokenizer, input_sequences, max_len

def get_language_name(lang_code):
    return language_dict.get(lang_code.lower(), "Unknown Language")

# Main application logic

language_dict = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}


languages = ['English', 'Hindi', 'Gujarati','Spanish','French','German']

selected_language = st.selectbox('Select a Language:', languages)



if 'search_input' not in st.session_state:
    st.session_state.search_input = ""

st.markdown('<div class="search-bar">', unsafe_allow_html=True)
st.text_input("Search", value=st.session_state.search_input, key="search_bar", placeholder="Type here...",
              on_change=lambda: st.session_state.__setitem__('search_input', st.session_state.search_bar))
user_input = st.session_state.search_input
st.markdown('</div>', unsafe_allow_html=True)


if user_input:
    
    lang = detect_language(user_input)
    
    lang_full_name = get_language_name(lang)
    
    
    
    if selected_language == lang_full_name:
       
    # Load or initialize model components
        if 'model_components' not in st.session_state or st.session_state.get('lang') != lang:
            model, tokenizer, input_sequences, max_len = main_flow(lang)
            if model is None and load_dataset(lang).strip():
                # If dataset exists but no model, train one
                model, tokenizer, input_sequences, max_len = update_and_train_model("", None, tokenizer, input_sequences, max_len, lang)
            st.session_state.model_components = [model, tokenizer, input_sequences, max_len]
            st.session_state.lang = lang
        else:
            model, tokenizer, input_sequences, max_len = st.session_state.model_components

        # Update and potentially retrain model with new input
        if model is None or check_new_words_percentage(user_input, tokenizer) >= 0.6:
            with st.spinner(f"Training model for {lang}..."):
                model, tokenizer, input_sequences, max_len = update_and_train_model(user_input, model, tokenizer, input_sequences, max_len, lang)
                if model is not None:
                    main_flow.clear()
                    st.session_state.model_components = [model, tokenizer, input_sequences, max_len]
                    st.success(f"Model for {lang} updated and trained!")
                else:
                    st.error("Failed to train model.")
        else:
            # Append new data without retraining if new words are below threshold
            save_dataset(lang, user_input)
            new_sequences = text_sequences(tokenizer, user_input)
            input_sequences.extend(new_sequences)
            max_len = max([len(x) for x in input_sequences])
            st.session_state.model_components = [model, tokenizer, input_sequences, max_len]

        # Predict next words
        suggestions = predict_next_words(user_input, model, tokenizer, max_len)
        if suggestions:
            st.markdown('<div class="suggestions">', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, suggestion in enumerate(suggestions):
                with cols[i]:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        st.session_state.search_input = user_input.rstrip() + " " + suggestion
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.write("No suggestions available.")   
    else:
      st.write(f"Detected language: {lang_full_name} is not match with selected language: {selected_language}")       
else:
      st.markdown('<div class="suggestions">Start typing to see suggestions...</div>', unsafe_allow_html=True)
