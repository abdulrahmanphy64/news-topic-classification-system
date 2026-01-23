import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_pipeline.data_source import NewsDatasetBuilder
from src.data_pipeline.validator import validate_record
from src.data_pipeline.cleaner import preprocess_text
from src.data_pipeline.feature_builder import TFIDFVectorizerWrapper


def test_invalid_input():
    records = [["This","is","my","test case"],["This","is","my","project"]]
    mode = "Training"
    validate = validate_record(records,mode)
    preprocessor = preprocess_text
    vectorizer = TFIDFVectorizerWrapper()
    builder = NewsDatasetBuilder(validate,preprocessor,vectorizer,mode)

    with pytest.raises(TypeError):
        data = builder.build(records)


def test_training():
    long_text = "One of the best ways to help students improve their essay writing is by focusing on shorter tasks. Writing a 200-word paragraph teaches them how to organize their thoughts around a specific topic and keep their main points sharp and clear. With a limited number of words, students learn the importance of staying focused and avoiding common mistakes like including unnecessary details.For younger writers, a 200-word essay might feel like the first thing they can fully control, which is empowering. They can express their personal experiences or reflect on personal growth in their writing without feeling overwhelmed by a longer essay assignment. Plus, it’s a common prompt in many writing exercises, allowing students to practice this skill frequently."

    records = [{"text": long_text, "label" : "Sports"}, 
               {"text": long_text, "label" : "Science"}]
    
    mode = "Training"
    validate = validate_record
    preprocessor = preprocess_text
    vectorizer = TFIDFVectorizerWrapper()
    builder = NewsDatasetBuilder(validate, preprocessor, vectorizer, mode)
    X,y = builder.build(records)

    assert X.shape[0] == 2
    assert X.shape[1] > 0
    assert len(y) == 2



