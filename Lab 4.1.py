#1
import pandas as pd
df_hotel = pd.read_csv('ITA105_Lab_4_Hotel_reviews.csv')
print("--- Thông tin tổng quan về bộ dữ liệu ---")
print(df_hotel.info())

print("\n--- Số lượng giá trị thiếu trong từng cột ---")
missing_data = df_hotel.isnull().sum()
print(missing_data)

#2
from sklearn.preprocessing import LabelEncoder
df_hotel['hotel_name'] = df_hotel['hotel_name'].fillna('Unknown')
df_hotel['customer_type'] = df_hotel['customer_type'].fillna('Unknown')

le = LabelEncoder()
df_label = df_hotel.copy()
df_label['customer_type_encoded'] = le.fit_transform(df_label['customer_type'])

df_onehot = pd.get_dummies(df_hotel, columns=['customer_type'], prefix='type')

print("--- Kết quả Label Encoding (Cột customer_type) ---")
print(df_label[['customer_type', 'customer_type_encoded']].head())

print("\n--- Kết quả One-Hot Encoding (Các cột mới tạo ra) ---")
print(df_onehot.filter(like='type_').head())

#3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

df_hotel = pd.read_csv('ITA105_Lab_4_Hotel_reviews.csv')
df_hotel['review_text'] = df_hotel['review_text'].astype(str) # Đảm bảo là dạng chuỗi

vietnamese_stopwords = ["và", "của", "là", "nhưng", "có", "rất", "hơi", "cũng", "đã"]

def preprocess_text(text):
    
    text = text.lower()
    
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    
    tokens = word_tokenize(text)
    
   
    filtered_tokens = [word for word in tokens if word not in vietnamese_stopwords]
    
    return " ".join(filtered_tokens)


df_hotel['clean_review'] = df_hotel['review_text'].apply(preprocess_text)


print("--- SO SÁNH TRƯỚC VÀ SAU TIỀN XỬ LÝ ---")
print(df_hotel[['review_text', 'clean_review']].head())

#4
from gensim.models import Word2Vec
sentences = [word_tokenize(str(review).lower()) for review in df_hotel['review_text']]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)


target_word = "sạch"

if target_word in model.wv.key_to_index:
    similar_words = model.wv.most_similar(target_word, topn=5)
    print(f"5 từ gần nghĩa nhất với '{target_word}':")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")
else:
    print(f"Từ '{target_word}' không xuất hiện trong từ điển của mô hình.")