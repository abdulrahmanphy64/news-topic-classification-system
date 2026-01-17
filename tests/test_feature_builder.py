import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_pipeline.feature_builder import TFIDFVectorizerWrapper

def test_vectorizer_fit_transform_basic():
    token_words = [["this", "is", "my", "project"]]
    vectorizer = TFIDFVectorizerWrapper()

    X = vectorizer.fit_transform(token_words)

    assert X.shape[0] == 1
    assert X.shape[1] > 0

def test_vectorizer_empty_list():
    token_words = [[]]
    vectorizer = TFIDFVectorizerWrapper()

    with pytest.raises(ValueError):
        vectorizer.fit_transform(token_words)