import json
import os
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class Dictionary:
    def __init__(self, dic_path=None):
        self.word2idx = {}
        self.idx2word = []

        if dic_path:
            with open(dic_path, "r", encoding="utf-8") as f:
                dic = json.load(f)
            self.idx2word = list(dic.values())
            self.word2idx = {v: int(k) for k, v in dic.items()}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, dic_path=None):
        self.dictionary = Dictionary(dic_path)

    def construct_dict(self, path):
        word_set = set()
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                word_set.update(["<bos>"] + list(line.strip()) + ["<eos>"])
            f.close()
        for word in word_set:
            self.dictionary.add_word(word)
        dic = dict(zip(range(len(self.dictionary.idx2word)), self.dictionary.idx2word))
        with open(".\\data\\dict.json", "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False)

    def tokenize(self, path):
        input_ids = []
        input_ids_flatten = []
        root = os.path.dirname(path)
        filename = os.path.basename(path)
        file_type = filename.split("_")[0]
        with open(path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Tokenizing " + filename):
                words = ["<bos>"] + list(line.strip()) + ["<eos>"]
                words_ids = [self.dictionary.word2idx[word] for word in words]
                input_ids.append(words_ids)
                input_ids_flatten.extend(words_ids)
        save_path = os.path.join(root, file_type + "_input_ids")
        save_path_flatten = os.path.join(root, file_type + "_input_ids_flatten")
        torch.save(input_ids, save_path)
        torch.save(input_ids_flatten, save_path_flatten)

        return input_ids_flatten

    def encode(self, input_text):
        if isinstance(input_text, str):
            input_text = list(input_text)
        input_ids = [self.dictionary.word2idx[word] for word in input_text]
        return input_ids

    def decode(self, input_ids):
        if isinstance(input_ids, int):
            input_ids = [input_ids]
        input_text = [self.dictionary.idx2word[ids] for ids in input_ids]
        input_text = "".join(input_text)
        return input_text

    def create_input_ids(self, root_path):
        train = self.tokenize(os.path.join(root_path, 'train_data.txt'))
        valid = self.tokenize(os.path.join(root_path, 'valid_data.txt'))
        test = self.tokenize(os.path.join(root_path, 'test_data.txt'))

        return train, valid, test

def data_prepare(opt):
    if not (os.path.isfile("./data/dict.json") and os.path.isfile("./data/train_input_ids") and os.path.isfile(
            "./data/valid_input_ids") and os.path.isfile("./data/test_input_ids")):
        my_corpus = Corpus(dic_path=None)
        my_corpus.construct_dict("./data/data.txt")
        with open("./data/data.txt", "r", encoding="utf-8") as f:
            data = f.readlines()
        train_data, valid_test_data = train_test_split(data, test_size=0.2, random_state=42)
        valid_data, test_data = train_test_split(valid_test_data, test_size=0.5, random_state=42)
        with open("./data/train_data.txt", "w", encoding="utf-8") as f:
            f.writelines(train_data)
        with open("./data/valid_data.txt", "w", encoding="utf-8") as f:
            f.writelines(valid_data)
        with open("./data/test_data.txt", "w", encoding="utf-8") as f:
            f.writelines(test_data)

        train_input_ids_flatten, valid_input_ids_flatten, test_input_ids_flatten = my_corpus.create_input_ids(
            "./data")
    else:
        my_corpus = Corpus(dic_path="./data/dict.json")
        train_input_ids_flatten, valid_input_ids_flatten, test_input_ids_flatten = torch.load(
            "./data/train_input_ids_flatten"), torch.load("./data/valid_input_ids_flatten"), torch.load(
            "./data/test_input_ids_flatten")

    vocab_size = len(my_corpus.dictionary.word2idx)
    return vocab_size, train_input_ids_flatten, valid_input_ids_flatten, test_input_ids_flatten

def create_batch_fnn(d, batch_size, seq_len, device):
    x = []
    y = []

    x = [d[i - seq_len:i] for i in range(seq_len, len(d))]
    y = d[seq_len:]

    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def create_batch_rnn(d, batch_size, seq_len, device):
    x = []
    y = []

    x = [d[i - seq_len:i] for i in range(seq_len, len(d) - 1)]
    y = [d[i - seq_len:i] for i in range(seq_len + 1, len(d))]

    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

