import os
import re
import collections
from tqdm import tqdm

result_path = "/data/hongbang/projects/Learning/nlp_course/homework3/BiLSTM-CRF/data/"
result_files = ["train.txt", "dev.txt", "test.txt"]
with open(os.path.join(result_path ,"test.txt")) as f:
    lines = f.readlines()
test_corpus = [" ".join([elem for elem in single_line if elem.strip()]) for single_line in lines]

vocab = collections.defaultdict(int)
for line in tqdm(test_corpus):
    vocab[line+' </w>'] += 1
    # for word in line:
    #     vocab[word+' </w>'] += 1


def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        # symbols = list(word.split()[0])+[word.split()[-1]]
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

num_merges = 200
pbar = tqdm(range(num_merges))
for i in pbar:
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    pbar.set_description("Merging {}".format("".join(list(best))))
    vocab = merge_vocab(best, vocab)
result = get_tokens(vocab)
print("End")