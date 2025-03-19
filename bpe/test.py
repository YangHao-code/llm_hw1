from transformers import GPT2Tokenizer
import torch
from bpe import Tokenizer
import os
import pickle

def get_valid_ids(input_ids, attention_mask):
    valid_ids = [input_ids[i] for i, mask in enumerate(attention_mask) if mask == 1]
    return valid_ids

if __name__ == "__main__":
    text1 = "Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university."

    text2 = "博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。"


    # 加载 GPT-2 的 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")
    my_tokenizer = Tokenizer()

    if os.path.exists("merge.pkl") and os.path.exists("id2vocab.pkl") and os.path.exists("vocab2id.pkl"):
        print("load data......")
        with open("merge.pkl", "rb") as f:
            my_tokenizer.merge_list = pickle.load(f)
        with open("id2vocab.pkl", "rb") as f:
            my_tokenizer.id2vocab = pickle.load(f)
        with open("vocab2id.pkl", "rb") as f:
            my_tokenizer.vocab2id = pickle.load(f)
    else:
        raise ValueError("You need to train the model to get these hyper parameters first before you try this test.")
    # 打印特殊 token（如 padding, unk, bos, eos 等）
    # print("特殊 Token:", tokenizer.special_tokens_map)

    print("开始对比文本1...")
    encoded1 = gpt2_tokenizer(text1)
    encoded2 = my_tokenizer.encode(text1)
    print(f"文本1encode后的tokens长度为：GPT2模型为{sum(encoded1['attention_mask'])}，我自己实现的为{len(encoded2)}")
    print(f"encode后的具体tokens为：GPT2模型为{get_valid_ids(encoded1['input_ids'], encoded1['attention_mask'])}，\n我自己实现的为{encoded2}")

    print("开始对比文本2...")
    encoded1 = gpt2_tokenizer(text2)
    encoded2 = my_tokenizer.encode(text2)
    print(f"文本2encode后的tokens长度为：GPT2模型为{sum(encoded1['attention_mask'])}，我自己实现的为{len(encoded2)}")
    print(f"encode后的具体tokens为：GPT2模型为{get_valid_ids(encoded1['input_ids'], encoded1['attention_mask'])}，\n我自己实现的为{encoded2}")
