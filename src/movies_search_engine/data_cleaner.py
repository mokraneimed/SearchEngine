import contractions
import string


def remove_contractions(text):
  text_without_contractions = contractions.fix(text)
  return text_without_contractions

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator)
    return text_without_punct