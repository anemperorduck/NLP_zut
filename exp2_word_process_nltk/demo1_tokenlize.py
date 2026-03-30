import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def tokenize_sent(text):
    sentences = sent_tokenize(text)
    return sentences


def tokenize_word(text):
    words = word_tokenize(text)
    return words

def tokenize(text):
    result = []
    sents = tokenize_sent(text)

    for sent in sents:
        words = tokenize_word(sent)
        result.extend(words)

    return result

if __name__ =='__main__':
    sample_text = "NLTK is a powerful library! It provides tools for NLP tasks including tokenization and stemming."

    sent_result = tokenize_sent(sample_text)
    print(f"tokenlize of the sentence:\n{sent_result}")

    word_before_result = tokenize_word(sample_text)
    print(f"tokenlize of the word defore sent_sentence:\n{word_before_result}")

    word_after_result = tokenize(sample_text)
    print(f"tokenlize of the word after sent_sentence:\n{word_after_result}")
    