import pytest
from synaesthesia.utils import check_camel_case_format


# Test for valid camel case strings
def test_valid_camel_case():
    valid_strings = [
        "camelCase",
        "machineName",
        "thisIsCamelCase",
        "lowerUpper"
        "aVeryVeryLongCamelCaseStringThatWorks",  # Very long camel case string (valid)
        "CamelCase",  # Starts with uppercase
        "camelCase15",  # Ends with a number
    ]
    for string in valid_strings:
        # No exception should be raised for valid strings
        check_camel_case_format(string)


# Test for invalid camel case strings
def test_invalid_camel_case():
    invalid_strings = [
        "camel_Case",  # Contains underscore
        "camel case",  # Contains spaces
        "CAMELCASE",  # All uppercase
        "123camelCase",  # Starts with a number
        "camel.Case"  # Contains dot
        "aB",  # Two letters
        "camel15Case",  # Middle number
    ]
    for string in invalid_strings:
        with pytest.raises(ValueError, match=f"'{string}' is not in camel case format"):
            check_camel_case_format(string)
