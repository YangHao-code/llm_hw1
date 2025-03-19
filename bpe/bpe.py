import regex as re
from collections import defaultdict
import json
import yaml
import pickle

class Tokenizer:
    def __init__(self):
        self.vocab2id = {}
        self.id2vocab = {}
        self.merge_list = []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 这个是为了避免多次重复找相同的单词
        for i in range(256):
            self.id2vocab[i] = bytes([i])

    def train(self, text, vocab_size):
        """
        Train the tokenizer using BPE algorithm.
        Params:
            text (str): string-type data used to run BPE.
            vocab_size (int): the size of final vocabulary.

        Return:
            None
        """
        vocab_num = 256
        pair_counts = defaultdict(int)
        word_num = defaultdict(int)
        for word in re.findall(self.PAT, text):
            codings = word.strip().encode("utf-8")
            word_key = tuple([codings[i:i+1] for i in range(len(codings))])
            word_num[word_key] += 1
        for word in word_num.keys():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += word_num[word]
        while vocab_num < vocab_size:
            max_pair = max(pair_counts, key = lambda k: (pair_counts[k], k))
            pair_counts[max_pair] = 0
            self.merge_list.append(max_pair)
            self.id2vocab[vocab_num] = b"".join([max_pair[0], max_pair[1]])
            vocab_num += 1
            for word in word_num.keys():
                if max_pair[0] in word and max_pair[1] in word:
                    for i in range(len(word) - 1):
                        if (word[i], word[i + 1]) == max_pair:
                            tem = b"".join(max_pair)
                            if i - 1 >= 0:
                                pair_counts[(word[i - 1], word[i])] -= word_num[word]
                                pair_counts[(word[i - 1], tem)] += word_num[word]
                            if i + 2 < len(word):
                                pair_counts[(word[i + 1], word[i + 2])] -= word_num[word]
                                pair_counts[(tem, word[i + 1])] += word_num[word]
        self.vocab2id = {v: k for k, v in self.id2vocab.items()}
        # with open("merge.yaml", "w") as f:
        #     #f.write(self.merge_list)
        #     yaml.dump(self.merge_list, f)
        # with open("id2vocab.json", "w") as f:
        #     json.dump(self.id2vocab, f, ensure_ascii=False)
        # with open("vocab2id.json", "w") as f:
        #     json.dump(self.vocab2id, f, ensure_ascii=False)
        with open("merge.pkl", "wb") as f:
            pickle.dump(self.merge_list, f)
        with open("vocab2id.pkl", "wb") as f:
            pickle.dump(self.vocab2id, f)
        with open("id2vocab.pkl", "wb") as f:
            pickle.dump(self.id2vocab, f)

    def encode(self, text):
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.

        Return:
            ids (list): list of integer-type tokens.
        """

        tokens = text.encode("utf-8")
        tokens = [tokens[i: i+1] for i in range(len(tokens))]
        for merge_pair in self.merge_list:
            tem = []
            length = len(tokens)
            i = 0
            while i < length - 1:
                if (tokens[i], tokens[i + 1]) == merge_pair:
                    tem.append(b"".join(merge_pair))
                    i += 2
                else:
                    tem.append(tokens[i])
                    i += 1
            if i == length - 1:
                tem.append(tokens[i])
            tokens = tem
        encodings = [self.vocab2id[item] for item in tokens]
        return encodings

    def decode(self, ids):
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.

        Return:
            text (str): string-type data.
        """
        tokens = [self.id2vocab[num] for num in ids]
        return (b"".join(tokens)).decode("utf-8")