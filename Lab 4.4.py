#1
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_album = pd.read_csv('ITA105_Lab_4_Album_reviews.csv')

df_album['genre'] = df_album['genre'].fillna('Unknown')
df_album['platform'] = df_album['platform'].fillna('Other')

df_encoded = pd.get_dummies(df_album, columns=['genre'], prefix='genre')

le = LabelEncoder()
df_encoded['platform_id'] = le.fit_transform(df_album['platform'])

print("--- Dữ liệu Album Reviews sau khi Encoding ---")
cols_to_show = [col for col in df_encoded.columns if 'genre_' in col] + ['platform', 'platform_id', 'rating']
print(df_encoded[cols_to_show].head(10))

#2
import nltk
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')

df_album = pd.read_csv('ITA105_Lab_4_Album_reviews.csv')

df_album['review_text'] = df_album['review_text'].fillna("").astype(str)

vietnamese_stopwords = ["và", "của", "là", "nhưng", "có", "rất", "với", "cho", "đã", "đang"]

def preprocess_album_review(text):
   
    text = text.lower()
    
  
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    

    filtered_tokens = [word for word in tokens if word not in vietnamese_stopwords]
    
    return " ".join(filtered_tokens)

df_album['clean_review'] = df_album['review_text'].apply(preprocess_album_review)


print("--- SO SÁNH TRƯỚC VÀ SAU TIỀN XỬ LÝ (ALBUM REVIEWS) ---")
print(df_album[['review_text', 'clean_review']].head(10))

#3
from sklearn.feature_extraction.text import TfidfVectorizer

df_album = pd.read_csv('ITA105_Lab_4_Album_reviews.csv')
df_album['review_text'] = df_album['review_text'].fillna("").astype(str)

tfidf_album = TfidfVectorizer(max_features=50, ngram_range=(1, 2))

tfidf_matrix = tfidf_album.fit_transform(df_album['review_text'])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=tfidf_album.get_feature_names_out()
)

print("--- Ma trận TF-IDF cho Album Reviews (5 dòng đầu) ---")
print(tfidf_df.head())

print("\n--- Top 10 đặc trưng văn bản quan trọng nhất ---")
print(tfidf_df.mean().sort_values(ascending=False).head(10))

#4
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

df_album = pd.read_csv('ITA105_Lab_4_Album_reviews.csv')

sentences = [word_tokenize(str(text).lower()) for text in df_album['review_text'] if pd.notnull(text)]

model_album = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

target_word = "sáng" 

if target_word in model_album.wv.key_to_index:
    similar_words = model_album.wv.most_similar(target_word, topn=5)
    print(f"5 từ có ngữ cảnh gần nhất với '{target_word}':")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")
else:
    print(f"Từ '{target_word}' không xuất hiện trong dữ liệu huấn luyện.")

#5
