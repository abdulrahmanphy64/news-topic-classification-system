import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text:str) -> list[str]:
    clean = re.sub(r'[^\w\s]','',text).lower().split()

    filtered_tokens = [word for word in clean if word not in stop_words and len(word) >= 3]

    return filtered_tokens

