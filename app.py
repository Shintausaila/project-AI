import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# 1. Judul Aplikasi
st.title("Hotel Review Sentiment Analysis")

# 2. Upload File
uploaded_file = st.file_uploader("Upload file CSV review hotel", type="csv")

if uploaded_file:
    # 3. Tampilkan data
    data = pd.read_csv(uploaded_file)
    st.write("Data Review:")
    st.write(data.head())

    # 4. Sentimen Analisis
    sia = SentimentIntensityAnalyzer()

    # Pastikan Review_Text tidak mengandung NaN dan mengonversi semua nilai ke string
    data['Review_Text'] = data['Review_Text'].fillna('').apply(str)

    # Lakukan analisis sentimen
    data['Sentiment'] = data['Review_Text'].apply(lambda x: 'Positive' if sia.polarity_scores(x)['compound'] > 0.05
                                                  else 'Negative' if sia.polarity_scores(x)['compound'] < -0.05
                                                  else 'Neutral')

    st.write("Hasil Analisis Sentimen:")
    st.write(data)

    # 5. Pie Chart Sentimen
    st.write("Distribusi Sentimen:")
    sentiment_counts = data['Sentiment'].value_counts()
    st.write(sentiment_counts)

    # Opsional: Pie Chart
    st.bar_chart(sentiment_counts)
    
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
