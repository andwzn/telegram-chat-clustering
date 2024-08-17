import string 
import emoji

def char_is_emoji(character: str) -> bool:
    """
    Check if a given character is an emoji.
    Parameters:
    character (str): The character to be checked.
    """
    
    emoji_range = ('\U0001F600' <= character <= '\U0001F64F') or \
                  ('\U0001F300' <= character <= '\U0001F5FF') or \
                  ('\U0001F680' <= character <= '\U0001F6FF') or \
                  ('\U0001F700' <= character <= '\U0001F77F') or \
                  ('\U0001F780' <= character <= '\U0001F7FF') or \
                  ('\U0001F800' <= character <= '\U0001F8FF') or \
                  ('\U0001F900' <= character <= '\U0001F9FF') or \
                  ('\U0001FA00' <= character <= '\U0001FA6F') or \
                  ('\U0001FA70' <= character <= '\U0001FAFF') or \
                  ('\U00002702' <= character <= '\U000027B0') or \
                  ('\U000024C2' <= character <= '\U0001F251')

    return emoji_range or character in ['\uFE0E', '\uFE0F'] or emoji.is_emoji(character)

def is_punctuation(character: str) -> bool:
    """
    Check if a character is a punctuation mark.
    Parameters:
    character (str): The character to be checked.
    """
    return character in string.punctuation

def is_emoji_or_punctuation_only(input_string: str) -> bool:
    """
    Checks if a given string contains only emojis or punctuation characters.
    Parameters:
    string (str): The input string to be checked.
    """
    for c in input_string:
        if not (char_is_emoji(c) or is_punctuation(c) or c==' '):
            return False
    return True