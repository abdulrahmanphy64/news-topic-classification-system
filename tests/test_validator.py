import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_pipeline.validator import validate_record

def empty_text():
    record = {
        "text" : " ",
        "label" : "sports"
    }

    result = validate_record(record, mode="training")

    assert result["is_valid"] is False
    assert result["reason"] == "EMPTY_TEXT"

def test_text_too_short():
    record = {
        "text" : "short text" * 5,
        "label" : "politics"
    }

    result = validate_record(record, mode="training")

    assert result["is_valid"] is False
    assert result["reason"] == "TEXT_TOO_SHORT"

def test_high_junk_ratio():
    text = "@@@@ #### $$$$$ try new things always" * 200

    record = {
        "text" : text,
        "label" : "tech"
    }

    result = validate_record(record, mode = "training")

    assert result["is_valid"] is False
    assert result["reason"] == "HIGH_JUNK_RATIO"

def test_non_english_text():
    text = "यह एक समाचार लेख है जो अंग्रेजी में नहीं है।" * 20

    record = {
        "text" : text,
        "label" : "world"
    }

    result = validate_record(record, mode = "training")

    assert result["is_valid"] is False
    assert result["reason"] == "NON_ENGLISH_TEXT"

def test_low_unique_word_ratio():
    text = "news try " * 300

    record = {
        "text" : text,
        "label" : "general"
    }

    result = validate_record(record, mode = "training")

    assert result["is_valid"] is False
    assert result["reason"] == "LOW_UNIQUE_WORD_RATIO"

def test_valid_record():
    text = "One of the best ways to help students improve their essay writing is by focusing on shorter tasks. Writing a 200-word paragraph teaches them how to organize their thoughts around a specific topic and keep their main points sharp and clear. With a limited number of words, students learn the importance of staying focused and avoiding common mistakes like including unnecessary details.For younger writers, a 200-word essay might feel like the first thing they can fully control, which is empowering. They can express their personal experiences or reflect on personal growth in their writing without feeling overwhelmed by a longer essay assignment. Plus, it’s a common prompt in many writing exercises, allowing students to practice this skill frequently."

    record = {
        "text": text,
        "label": "news",
        "source": "reuters"
    }

    result = validate_record(record, mode="training")

    assert result["is_valid"] is True
    assert result["reason"] is None