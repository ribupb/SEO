import numpy as np

def get_seo_label(pred):
    labels = ["Low", "Medium", "High"]
    return labels[int(pred)]
