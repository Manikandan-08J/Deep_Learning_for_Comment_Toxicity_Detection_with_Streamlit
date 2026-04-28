import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocess import clean_text

LABEL_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
MAX_LEN    = 200
THRESHOLD  = 0.5

def load_artifacts():
    model = load_model("models/toxicity_model.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_single(text, model, tokenizer):
    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN)
    probs   = model.predict(padded, verbose=0)[0]
    result  = {label: float(prob) for label, prob in zip(LABEL_COLS, probs)}
    result["is_toxic"]       = any(p >= THRESHOLD for p in probs)
    result["dominant_label"] = LABEL_COLS[int(np.argmax(probs))] if result["is_toxic"] else "clean"
    return result

def predict_batch(texts, model, tokenizer):
    cleaned = [clean_text(t) for t in texts]
    seqs    = tokenizer.texts_to_sequences(cleaned)
    padded  = pad_sequences(seqs, maxlen=MAX_LEN)
    probs   = model.predict(padded, verbose=0)
    df_out  = pd.DataFrame(probs, columns=LABEL_COLS)
    df_out.insert(0, "comment_text", texts)
    df_out["is_toxic"]       = (probs >= THRESHOLD).any(axis=1)
    df_out["dominant_label"] = [
        LABEL_COLS[int(np.argmax(row))] if any(p >= THRESHOLD for p in row) else "clean"
        for row in probs
    ]
    return df_out
