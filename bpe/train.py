# train 
from bpe import Tokenizer
import os
import pickle

if __name__ == "__main__":
    model = Tokenizer()
    with open("./manual.txt", "r") as f:
        text = f.read()
    print("here")
    if os.path.exists("merge.pkl") and os.path.exists("id2vocab.pkl") and os.path.exists("vocab2id.pkl"):
        print("load data......")
        with open("merge.pkl", "rb") as f:
            model.merge_list = pickle.load(f)
        with open("id2vocab.pkl", "rb") as f:
            model.id2vocab = pickle.load(f)
        with open("vocab2id.pkl", "rb") as f:
            model.vocab2id = pickle.load(f)
    else:
        model.train(text, 1024)
    print("Begin to encode...")
    encodings = model.encode(text)
    print("Begin to decode...")
    ans_text = model.decode(encodings)
    with open("./test_manual.txt", "w") as f:
        f.write(ans_text)