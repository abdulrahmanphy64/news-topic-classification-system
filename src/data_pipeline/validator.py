from langdetect import detect_langs
"""
Validator for raw news records.

Responsibilities:
-Validate raw input records before processing
-Reject invalid data with explicit reason codes
-Do not modify data
"""
# Reason codes
REASON_EMPTY_TEXT = "EMPTY_TEXT"
REASON_TEXT_TOO_SHORT = "TEXT_TOO_SHORT"
REASON_MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
REASON_NON_ENGLISH_TEXT = "NON_ENGLISH_TEXT"
REASON_HIGH_JUNK_RATIO = "HIGH_JUNK_RATIO"
REASON_LOW_UNIQUE_WORD_RATIO = "LOW_UNIQUE_WORD_RATIO"
REASON_INVALID_MODE = "INVALID_MODE"


# Constant/threshold
MIN_TEXT_LENGTH = 200
MAX_JUNK_RATIO = 0.4
MIN_UNIQUE_RATIO = 0.2

def validate_record(record: dict, mode: str) -> dict:   
    ok, reason = _check_text_exists(record)
    if not ok:
        return {
            "is_valid" : False,
            "reason" : reason
        }
    
    text = record["text"]

    ok, reason = _check_text_length(text)
    if not ok:
        return {
            "is_valid" : False,
            "reason" : reason
        }
    
    ok , reason = _check_language(text)
    if not ok:
        return {
            "is_valid": False,
            "reason": reason
        }
    
    ok, reason = _check_junk_ratio(text)
    if not ok:
        return {
            "is_valid" : False,
            "reason" : reason
        }
    
    
    ok, reason = _check_unique_word_ratio(text)
    if not ok:
        return {
            "is_valid" : False,
            "reason" : reason
        }
    
    return {
        "is_valid" : True,
        "reason" : None
    }

#------Internal rule check---------

def _check_text_exists(record: dict):
    if "text" not in record:
        return False, REASON_MISSING_REQUIRED_FIELD
    
    text = record.get("text")

    if text is None:
        return False, REASON_EMPTY_TEXT
    
    if not isinstance(text, str):
        return False, REASON_EMPTY_TEXT
    
    if text.strip() == "":
        return False, REASON_EMPTY_TEXT
    
    return True, None

def _check_text_length(text : str):
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return False, REASON_TEXT_TOO_SHORT
    
    return True, None 

def _check_language(text: str):
    try:    
        languages = detect_langs(text)
        language = languages[0]
    except:
        return False, REASON_NON_ENGLISH_TEXT
    
    if language.lang != "en":
        return False, REASON_NON_ENGLISH_TEXT
    
    if language.prob < 0.8:
        return False, REASON_NON_ENGLISH_TEXT
    
    return True, None

def _check_junk_ratio(text: str):
    text = text.strip()
    count = sum(1 for ch in text if not ch.isalpha() and not ch.isspace())

    space_remove = text.replace(" ", "")
    effective_length = len(space_remove)

    if effective_length == 0:
        return False, REASON_EMPTY_TEXT

    if count/effective_length > MAX_JUNK_RATIO:
        return False, REASON_HIGH_JUNK_RATIO
    
    return True, None

def _check_unique_word_ratio(text : str):
    tokens = text.lower().split()
    clean_words = [t for t in tokens if t.isalpha()]
    total_words = len(clean_words)
    unique_words = len(set(clean_words))

    if total_words == 0:
        return False, REASON_EMPTY_TEXT
    
    if unique_words / total_words < MIN_UNIQUE_RATIO:
        return False, REASON_LOW_UNIQUE_WORD_RATIO
    
    return True, None



