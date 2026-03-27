#1
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_player = pd.read_csv('ITA105_Lab_4_Player_feedback.csv')

df_player['device'] = df_player['device'].fillna('unknown')
df_player['player_type'] = df_player['player_type'].fillna('unknown')

df_player_encoded = pd.get_dummies(df_player, columns=['device'], prefix='dev')

le = LabelEncoder()
df_player_encoded['player_type_encoded'] = le.fit_transform(df_player['player_type'])

print("--- Dữ liệu Player Feedback sau khi Encoding ---")
cols_to_show = [col for col in df_player_encoded.columns if 'dev_' in col] + ['player_type', 'player_type_encoded', 'score']
print(df_player_encoded[cols_to_show].head(10))

#2
import nltk
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')

df_player = pd.read_csv('ITA105_Lab_4_Player_feedback.csv')

df_player['feedback_text'] = df_player['feedback_text'].fillna("").astype(str)

vietnamese_stopwords = ["và", "của", "là", "nhưng", "có", "rất", "hơi", "cũng", "đã", "đang"]

def preprocess_feedback(text):
  
    text = text.lower()
    
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    
    tokens = word_tokenize(text)
    
   
    filtered_tokens = [word for word in tokens if word not in vietnamese_stopwords]
    
    return " ".join(filtered_tokens)

df_player['clean_feedback'] = df_player['feedback_text'].apply(preprocess_feedback)

print("--- SO SÁNH TRƯỚC VÀ SAU TIỀN XỬ LÝ (PLAYER FEEDBACK) ---")
print(df_player[['feedback_text', 'clean_feedback']].head(10))

#3

from sklearn.feature_extraction.text import TfidfVectorizer

df_player = pd.read_csv('ITA105_Lab_4_Player_feedback.csv')
df_player['feedback_text'] = df_player['feedback_text'].fillna("").astype(str)

tfidf_vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))

tfidf_matrix = tfidf_vectorizer.fit_transform(df_player['feedback_text'])


tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("--- Ma trận TF-IDF cho Player Feedback (5 dòng đầu) ---")
print(tfidf_df.head())

print("\n--- Top 10 từ khóa quan trọng nhất trong Feedback ---")
importance = tfidf_df.mean().sort_values(ascending=False)
print(importance.head(10))

#4
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

df_player = pd.read_csv('ITA105_Lab_4_Player_feedback.csv')

sentences = [word_tokenize(str(text).lower()) for text in df_player['feedback_text'] if pd.notnull(text)]

model_player = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

target_word = "đẹp"

if target_word in model_player.wv.key_to_index:
    similar_words = model_player.wv.most_similar(target_word, topn=5)
    print(f"5 từ có ngữ cảnh gần nhất với '{target_word}':")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")
else:
    print(f"Từ '{target_word}' không xuất hiện trong dữ liệu feedback.")