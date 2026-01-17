from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFVectorizerWrapper:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, tokenized_texts):
        documents = []
        for tokenized_text in tokenized_texts:
            document_text = " ".join(tokenized_text)
            documents.append(document_text)
        
        self.vectorizer.fit(documents)
        return self
    
    def transform(self, tokenized_texts):
        documents = []
        for tokenized_text in tokenized_texts:
            documents_text = " ".join(tokenized_text)
            documents.append(documents_text)

        matrix = self.vectorizer.transform(documents)
        return matrix

    def fit_transform(self, tokenized_texts):
        self.fit(tokenized_texts)
        return self.transform(tokenized_texts)
