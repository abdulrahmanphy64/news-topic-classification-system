class NewsDatasetBuilder:
    def __init__(self,validator,preprocessor, vectorizer, mode):
        self.validator = validator
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.mode = self.validate_mode(mode)

    def validate_mode(self,mode: str):
        if not  mode.lower() in ("training", "inference"):
            raise ValueError("Mode must be 'training' or 'inference' ")
        
        return mode.lower()
        
    def build(self, records: list[dict]):
        if not isinstance(records, list):
            raise TypeError("Records must be lists of dicts")
        
        if not all(isinstance(r,dict) for r in records):
            raise TypeError("Each record must be a dict")
        
        valid_records = []
        for record in records:
            result = self.validator(record,self.mode)
            if not result["is_valid"]:
                continue

            valid_records.append(record)

        tokenized_texts = []
        filtered_texts = []

        for record in valid_records:
            tokens = self.preprocessor(record["text"])

            if not tokens:
                continue

            tokenized_texts.append(tokens)
            filtered_texts.append(record)

        if self.mode == "training":
            labeled_records = []
            labeled_tokens = []

            for record, tokens in zip(filtered_texts, tokenized_texts):
                if "label" not in record:
                    continue
                labeled_records.append(record)
                labeled_tokens.append(tokens)

            if not labeled_tokens:
                raise ValueError("No labeled records available for training")
            
            X = self.vectorizer.fit_transform(labeled_tokens)
            y = [r["label"] for r in labeled_records]
            return X,y
        

        