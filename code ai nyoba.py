import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

model_path = 'naive_bayes_model.pkl'
naive_bayes_model = joblib.load(model_path)

# Fungsi untuk klasifikasi sentimen
def classify_document(doc, model):
    prior_prob = model['prior_probabilities']
    prob_cond_positive = model['prob_cond_positive']
    prob_cond_neutral = model['prob_cond_neutral']
    prob_cond_negative = model['prob_cond_negative']
    total_positive_words = model['total_positive_words']
    total_neutral_words = model['total_neutral_words']
    total_negative_words = model['total_negative_words']
    feature_count = model['feature_count']

    words = doc.split()
    posterior_positive = prior_prob['positif']
    posterior_neutral = prior_prob['netral']
    posterior_negative = prior_prob['negatif']

    for word in words:
        posterior_positive *= prob_cond_positive.get(word, 1 / (total_positive_words + feature_count))
        posterior_neutral *= prob_cond_neutral.get(word, 1 / (total_neutral_words + feature_count))
        posterior_negative *= prob_cond_negative.get(word, 1 / (total_negative_words + feature_count))

    return max(
        {'positif': posterior_positive, 'netral': posterior_neutral, 'negatif': posterior_negative},
        key=lambda x: {'positif': posterior_positive, 'netral': posterior_neutral, 'negatif': posterior_negative}[x]
    )

# Streamlit UI
st.title("Klasifikasi dan Analisis Sentimen Ulasan Hotel")
st.write("Aplikasi ini menggunakan model Naive Bayes untuk menganalisis sentimen ulasan hotel.")

# Informasi Model
st.subheader("Informasi Model")
st.write("Model Naive Bayes telah dilatih menggunakan dataset ulasan hotel. Berikut adalah ringkasan probabilitas prior untuk setiap sentimen:")

prior_probabilities = naive_bayes_model['prior_probabilities']
st.write("**Probabilitas Prior:**")
st.write(prior_probabilities)

st.write("**Total Kata dalam Setiap Kategori:**")
st.write({
    "Positif": naive_bayes_model['total_positive_words'],
    "Netral": naive_bayes_model['total_neutral_words'],
    "Negatif": naive_bayes_model['total_negative_words']
})

st.write("**Jumlah Fitur Unik (Kata):**")
st.write(naive_bayes_model['feature_count'])

# Analisis Teks Individu
st.subheader("Analisis Teks Individu")
input_text = st.text_area("Masukkan teks ulasan:", key="individual_review")

if st.button("Analisis Sentimen"):
    if input_text.strip():
        preprocessed_text = input_text.lower()
        preprocessed_text = ''.join([char for char in preprocessed_text if char.isalnum() or char.isspace()])

        sentiment = classify_document(preprocessed_text, naive_bayes_model)

        st.write("**Hasil Analisis Sentimen:**")
        st.write(f"Sentimen: **{sentiment.capitalize()}**")
    else:
        st.warning("Harap masukkan teks sebelum menganalisis.")

# Analisis Batch Ulasan
st.subheader("Analisis Batch Ulasan")
file_upload = st.file_uploader("Unggah file CSV dengan kolom 'Review_Text':", type=["csv"], key="batch_review")

if file_upload:
    df = pd.read_csv(file_upload)

    if 'Review_Text' in df.columns:
        df['Preprocessed_Text'] = df['Review_Text'].astype(str).str.lower()
        df['Preprocessed_Text'] = df['Preprocessed_Text'].apply(lambda x: ''.join([char for char in x if char.isalnum() or char.isspace()]))
        
        # Klasifikasi sentimen untuk batch
        df['Predicted_Sentiment'] = df['Preprocessed_Text'].apply(lambda x: classify_document(x, naive_bayes_model))

        st.write("Hasil Analisis:")
        st.dataframe(df[['Review_Text', 'Predicted_Sentiment']])

        # Visualisasi distribusi sentimen
        sentiment_counts = df['Predicted_Sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
        ax.set_title("Distribusi Sentimen")
        ax.set_xlabel("Sentimen")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

        # Wordcloud untuk setiap kategori sentimen
        positive_reviews = " ".join(df[df['Predicted_Sentiment'] == 'positif']['Review_Text'])
        neutral_reviews = " ".join(df[df['Predicted_Sentiment'] == 'netral']['Review_Text'].astype(str))  # Konversi menjadi string
        negative_reviews = " ".join(df[df['Predicted_Sentiment'] == 'negatif']['Review_Text'])

        wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
        wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_reviews)
        wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)

        st.subheader("Wordcloud Sentimen Positif")
        st.image(wordcloud_positive.to_array())
        st.subheader("Wordcloud Sentimen Netral")
        st.image(wordcloud_neutral.to_array())
        st.subheader("Wordcloud Sentimen Negatif")
        st.image(wordcloud_negative.to_array())

        # Tombol untuk mengunduh hasil analisis
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil sebagai CSV",
            data=csv,
            file_name='predicted_sentiments.csv',
            mime='text/csv',
        )
    else:
        st.error("File CSV harus memiliki kolom 'Review_Text'.")

# Pencarian Nama Hotel
st.subheader("Pencarian Nama Hotel")
search_hotel = st.text_input("Masukkan nama hotel:", key="hotel_search")

if st.button("Cari Hotel", key="search_button"):
    if search_hotel.strip():
        # Load the dataset
        df = pd.read_csv('hotel_reviews.csv')

        # Filter data based on the hotel name entered
        filtered_data = df[df['Name'].str.contains(search_hotel, case=False, na=False)]
        
        if not filtered_data.empty:
            st.write("**Hasil Pencarian Ulasan untuk Hotel:**")
            # Add sentiment classification to the filtered reviews
            filtered_data['Predicted_Sentiment'] = filtered_data['Review_Text'].apply(lambda x: classify_document(x, naive_bayes_model))
            st.dataframe(filtered_data[['Name', 'Review_Text', 'Predicted_Sentiment']])
        else:
            st.write(f"Tidak ada ulasan ditemukan untuk hotel '{search_hotel}'.")
            # Generate and display predictions for new hotel name
            new_reviews = ["Ulasan pertama untuk " + search_hotel, "Ulasan kedua untuk " + search_hotel]
            new_data = pd.DataFrame({
                'Name': [search_hotel] * len(new_reviews),
                'Review_Text': new_reviews
            })

            new_data['Predicted_Sentiment'] = new_data['Review_Text'].apply(lambda x: classify_document(x, naive_bayes_model))
            st.write(f"**Hasil Prediksi untuk Hotel Baru '{search_hotel}':**")
            st.dataframe(new_data[['Name', 'Review_Text', 'Predicted_Sentiment']])
    else:
        st.warning("Harap masukkan nama hotel untuk mencari ulasan.")
