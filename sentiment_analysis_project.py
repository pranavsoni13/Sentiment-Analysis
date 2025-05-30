import re
import os
import threading
import joblib
import requests
import nltk
import pandas as pd
import tweepy
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from collections import Counter
from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok, conf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv, find_dotenv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout

# Load environment variables
load_dotenv(find_dotenv())
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
print("🔐 Bearer Token Loaded:", bool(BEARER_TOKEN))

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

# Clean and preprocess text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\@\w+|\#", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english') and word.isalpha()]
    return " ".join(tokens)

# Load Sentiment140 Dataset
print("📅 Loading dataset...")
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df['sentiment'] = df['target'].apply(lambda x: 0 if x == 0 else 1)  # 0=negative, 1=positive

# Use a smaller sample for faster training
sample_df = df[['text', 'sentiment']].sample(10000, random_state=42)
sample_df['cleaned'] = sample_df['text'].apply(clean_text)

# Tokenization and Padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sample_df['cleaned'])
sequences = tokenizer.texts_to_sequences(sample_df['cleaned'])
padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

X = padded
y = sample_df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build DL Model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate and save
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc}")
model.save("tweet_dl_model.h5")
joblib.dump(tokenizer, "tweet_tokenizer.pkl")

# Flask App
conf.get_default().auth_token = "2xjO5X93hglwuNgxUy831NmhwH0_7GcuWozTi7366RzoxNHb1"  # Replace with your token
model = load_model("tweet_dl_model.h5")
tokenizer = joblib.load("tweet_tokenizer.pkl")
app = Flask(__name__)
public_url = ngrok.connect(5000, bind_tls=True)
print(f"🚀 ngrok tunnel available at: {public_url.public_url}")

@app.route("/")
def index():
    return render_template("Html1.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    tweet = data.get("tweet", "")
    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400
    cleaned = clean_text(tweet)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(pad)[0][0]
    label = "positive" if prediction >= 0.5 else "negative"
    return jsonify({"tweet": tweet, "prediction": label, "confidence": float(prediction)})

def run_app():
    app.run(port=5000, use_reloader=False)

threading.Thread(target=run_app).start()

# Test API
try:
    response = requests.post(f"{public_url.public_url}/predict", json={"tweet": "I hate this terrible AI!"})
    if response.status_code == 200:
        print("✅ API Response:", response.json())
    else:
        print(f"❌ API request failed: {response.status_code}")
except Exception as e:
    print("❌ API call error:", e)
