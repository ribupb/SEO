from textstat import textstat
from nltk.tokenize import sent_tokenize
import numpy as np

def compute_metrics(text):
    words = len(text.split())
    sentences = len(sent_tokenize(text)) if text else 1
    readability = textstat.flesch_reading_ease(text)
    reading_time = textstat.reading_time(text)
    avg_sentence_length = words / sentences if sentences > 0 else 0

    return np.array([words, sentences, readability, reading_time, avg_sentence_length])
