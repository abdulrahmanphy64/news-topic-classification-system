import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_pipeline.cleaner import preprocess_text


def test_token_list():
    text = "This is a sample test for preprocessing"

    result = preprocess_text(text)

    assert isinstance(result,list)
    assert result == ["sample", "test", "preprocessing"]
    assert all(isinstance(token, str) for token in result)
    assert all(len(token) >= 3 for token in result)
    assert all(token.islower() for token in result)