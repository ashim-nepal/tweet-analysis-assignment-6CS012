import streamlit as st
import numpy as np
import re, string, pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load models and tokenizer
model_rnn = load_model('TweetAnalysis-6CS012/model1.h5')
model_lstm = load_model('TweetAnalysis-6CS012/model2.h5')
model_w2v = load_model('TweetAnalysis-6CS012/model3.h5')

with open('TweetAnalysis-6CS012/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


max_len = 13

# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|\d+", '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r"\b(?:{})\b".format('|'.join(stop_words)), '', text)
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    return text.strip()

# Prediction function
def predict_sentiment(text, model):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)[0][0]
    return "Racist/Sexist" if pred > 0.5 else "Not Racist Sexist"

# Streamlit UI
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #333;'>💬 Real-Time Tweet Sentiment Analyzer</h1>
    <p style='text-align: center; font-size: 18px;'>Enter a tweet below and click one of the model buttons to get the sentiment analysis result in real-time.</p>
""", unsafe_allow_html=True)

text_input = st.text_area("Enter your tweet here:", height=150, placeholder="e.g. I've completed Assignment of AI'!")

col1, col2, col3 = st.columns(3)
result = ""

if col1.button("🔁 Predict with RNN"):
    result = predict_sentiment(text_input, model_rnn)

if col2.button("🔁 Predict with LSTM"):
    result = predict_sentiment(text_input, model_lstm)

if col3.button("🔁 Predict with Word2Vec-LSTM"):
    result = predict_sentiment(text_input, model_w2v)

if result:
    st.markdown(f"""
        <div style='margin-top: 20px; padding: 15px; background-color: #f0f2f6; border-radius: 10px; text-align: center; font-size: 22px; color: #333;'>
            <strong>Prediction:</strong> {result}
        </div>
    """, unsafe_allow_html=True)
