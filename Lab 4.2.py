#1
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_match = pd.read_csv('ITA105_Lab_4_Match_comments.csv')

df_match['team'] = df_match['team'].fillna('Unknown')
df_match['author'] = df_match['author'].fillna('Anonymous')

df_encoded = pd.get_dummies(df_match, columns=['team'], prefix='team')

le = LabelEncoder()
df_encoded['author_id'] = le.fit_transform(df_match['author'])

print("--- Dữ liệu sau khi mã hóa ---")
cols_to_show = [col for col in df_encoded.columns if 'team_' in col] + ['author', 'author_id']
print(df_encoded[cols_to_show].head(10))

#2
import nltk
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')

df_match = pd.read_csv('ITA105_Lab_2_Match_comments.csv')

df_match['comment_text'] = df_match['comment_text'].fillna("").astype(str)

vietnamese_stopwords = ["và", "của", "là", "nhưng", "có", "rất", "đã", "đang", "sẽ", "phải"]

def preprocess_match_comment(text):
    
    text = text.lower()
    
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
  
    tokens = word_tokenize(text)
    
    
    filtered_tokens = [word for word in tokens if word not in vietnamese_stopwords]
    
    return " ".join(filtered_tokens)
df_match['clean_comment'] = df_match['comment_text'].apply(preprocess_match_comment)

print("--- SO SÁNH TRƯỚC VÀ SAU TIỀN XỬ LÝ (BÌNH LUẬN) ---")
print(df_match[['comment_text', 'clean_comment']].head(10))

#3
from sklearn.feature_extraction.text import TfidfVectorizer

df_match = pd.read_csv('ITA105_Lab_4_Match_comments.csv')
df_match['comment_text'] = df_match['comment_text'].fillna("").astype(str)

tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))

tfidf_matrix = tfidf_vectorizer.fit_transform(df_match['comment_text'])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("--- Ma trận TF-IDF (5 dòng đầu) ---")
print(tfidf_df.head())

sample_idx = 0
print(f"\n--- Các từ quan trọng trong bình luận: '{df_match['comment_text'].iloc[sample_idx]}' ---")
weights = pd.Series(tfidf_matrix.toarray()[sample_idx], index=tfidf_vectorizer.get_feature_names_out())
print(weights.sort_values(ascending=False).head(5))

#4
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

df_match = pd.read_csv('ITA105_Lab_4_Match_comments.csv')

sentences = [word_tokenize(str(comment).lower()) for comment in df_match['comment_text'] if pd.notnull(comment)]

model_match = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

target_word = "xuất" 

if target_word in model_match.wv.key_to_index:
    similar_words = model_match.wv.most_similar(target_word, topn=5)
    print(f"5 từ có ngữ cảnh gần nhất với '{target_word}':")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")
else:
    print(f"Từ '{target_word}' không tồn tại trong từ điển huấn luyện.")

#5
