import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Data contoh
data = ['saya suka makan nasi', 'dia tidak suka makan roti', 'saya suka olahraga']
labels = [1, 0, 1]

# Proses data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)

# Buat dan latih model
model = MultinomialNB()
model.fit(X, labels)

# Simpan model dan vectorizer
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model dan vectorizer berhasil disimpan!")


import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Data contoh
data = ['saya suka makan nasi', 'dia tidak suka makan roti', 'saya suka olahraga']
labels = [1, 0, 1]

# Proses data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)

# Buat dan latih model
model = MultinomialNB()
model.fit(X, labels)

# Simpan model dan vectorizer
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model dan vectorizer berhasil disimpan!")


