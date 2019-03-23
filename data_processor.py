#coding=utf-8
import fileinput
import numpy as np
import re

from multiprocessing import Pool

class DataSet(object):
    def __init__(self, s1, s2, label):
        self.index_in_epoch = 0
        self.s1 = s1
        self.s2 = s2
        self.label = label
        self.example_nums = len(label)
        self.epochs_completed = 0

    # Next batch data
    def next_batch2(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.example_nums:
            self.epochs_completed += 1 # begin next epoch
            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.s1 = self.s1[perm]
            self.s2 = self.s2[perm]
            self.label = self.label[perm]
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.example_nums
        end = self.index_in_epoch
        return np.array(self.s1[start:end]),np.array(self.s2[start:end]), np.array(self.label[start:end])

    def next_batch(self, batch_size):
        if self.index_in_epoch > self.example_nums:
            self.epochs_completed += 1
            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.s1 = self.s1[perm]
            self.s2 = self.s2[perm]
            self.label = self.label[perm]
            self.index_in_epoch = 0
            assert batch_size <= self.example_nums
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        end = min(self.index_in_epoch, self.example_nums)
        return np.array(self.s1[start:end]),np.array(self.s2[start:end]), np.array(self.label[start:end])


class WordVec(object):
    """
    Pandas replacement
    """
    def __init__(self):
        self.vocab = []
        self.vocab_vec = {}
        self.vectors = []
        return

def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def padding_sentence(s1, s2):
    s1_max = max([len(s) for s in s1])
    s2_max = max([len(s) for s in s2])
    sentence_length = max(s1_max, s2_max)
    sentence_num = s1.shape[0]
    s1_padding = np.zeros([sentence_num, sentence_length], dtype = int)
    s2_padding = np.zeros([sentence_num, sentence_length], dtype=int)
    for i, s in enumerate(s1):
        s1_padding[i][:len(s)] = s
    for i, s in enumerate(s2):
        s2_padding[i][:len(s)] = s
    print "Sentence padding completed..."
    return s1_padding, s2_padding

def get_id(word):
    if word in sr_word2id:
        return sr_word2id[word]
    else:
        return sr_word2id['<unk>']

def seq2id(seq):
    seq = clean_string(seq)
    seq_split = seq.split(' ')
    seq_id = map(get_id, seq_split)
    return seq_id

def load_vec(word2vec_path):
    wordvec = WordVec()
    i = 0
    for line in fileinput.input(word2vec_path):
        line = line.strip()
        lists = line.split(" ")
        word = lists[0]
        vec = np.asarray(map(float, lists[1:]))
        wordvec.vocab.append(word)
        wordvec.vocab_vec[word] = i + 1
        wordvec.vectors.append(vec)
        i += 1
    return wordvec

def build_glove_dic():
    glove_path = 'glove.6B.50d.txt'
    wv = load_vec(glove_path)
    vocab = wv.vocab
    sr_word2id = wv.vocab_vec
    sr_word2id['<unk>'] = 0
    word_embedding = wv.vectors
    word_mean = np.mean(word_embedding, axis = 0)
    word_embedding = np.vstack([word_mean, word_embedding])
    return sr_word2id, word_embedding

def get_value(train_dir, use_cols, delimiter = '\t', skip_header = 0):
    row = -1
    res = {}
    if not isinstance(use_cols, list) or len(use_cols) == 0:
        return res
    for i in use_cols:
        res[i] = []
    for line in fileinput.input(train_dir):
        row += 1
        if row < skip_header:
            continue
        lists = line.split(delimiter)
        for k, value in enumerate(lists):
            if k in use_cols:
                res[k].append(value)
    return res

def read_data(train_dir):
    df_sick = get_value(train_dir, use_cols = [1, 2, 4], skip_header = 1)
    s1 = df_sick[1]
    s2 = df_sick[2]
    score = np.asarray(map(float, df_sick[4]), dtype = np.float32)
    score = (score - score.min()) / (score.max() - score.min())
    sample_num = len(score)
    global sr_word2id, word_embedding
    sr_word2id, word_embedding = build_glove_dic()

    p = Pool()
    s1 = np.asarray(p.map(seq2id, s1))
    s2 = np.asarray(p.map(seq2id, s2))
    p.close()
    p.join()

    s1, s2 = padding_sentence(s1, s2)
    new_index = np.random.permutation(sample_num)
    s1 = s1[new_index]
    s2 = s2[new_index]
    score = score[new_index]

    return s1, s2, score

#if __name__ == "__main__":
#    s1, s2, score = read_data("data/1")
