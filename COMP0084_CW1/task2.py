import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import sparse


def text_preprocessing(file_name):
    vocab = dict()
    with open(file_name, 'r') as f: 
        lines = [re.sub(u"([^\u0061-\u007a\u0030-\u0039\u0020])", "", line.strip('\n').lower()) for line in f]
        for line in lines:
            line = line.split(' ')
            line.remove('') if '' in line else line
            for word in line:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    del vocab['']
    vocab = sorted(vocab.items(), key = lambda item: item[1], reverse=True)
    return vocab

if __name__ == '__main__':
    vocab = text_preprocessing('coursework-1-data/passage-collection.txt')

    # read passage
    whole_passage = []
    with open('coursework-1-data/candidate-passages-top1000.tsv', 'r') as f: 
        for line in f.readlines():
            line = line.strip('\n').lower().split('\t')
            line[0] = int(line[0])
            line[1] = int(line[1])
            line[2] = re.sub(u"([^/u0061-\u007a\u0030-\u0039\u0020])", "", line[2])
            line[3] = re.sub(u"([^\u0061-\u007a\u0030-\u0039\u0020])", "", line[3])
            whole_passage.append(line)

    #remove redundant passages, sort them with pid

    #uni_passage is in order with rank_pid
    whole_passage = np.array(whole_passage)
    pid = whole_passage[:, 1].astype(np.int64)
    rank_pid, idx_p = np.unique(pid, return_index=True)
    whole_passage_p = whole_passage[idx_p]
    uni_passage = whole_passage_p[:, -1]


    # remove stop words in vocab
    num_stop_words = 20
    new_vocab = np.array(np.array(vocab)[num_stop_words: ])[:, 0]
    new_vocab_dict = {}

    for i, word in enumerate(new_vocab):
        new_vocab_dict[word] = i


    #construct inverted_idx matrix
    print('start constructing inverted index matrix...')
    row,col,data = [],[],[]
    Ld = []
    for i, line in enumerate(tqdm(uni_passage)):
        line = line.split(' ')
        line.remove('') if '' in line else line
        Ld.append(len(line))

        line_vocab = {}
            
        for word in line:
            #column is in order with the frequency of word in vocabulary
            if word in new_vocab_dict:
                row.append(i)
                col.append(new_vocab_dict[word])
                data.append(1)

    inverted_idx = sparse.csr_matrix((data, (row, col)), shape=(len(uni_passage), len(new_vocab_dict)))
    print('Finished!\n')

    ## construct inverted_idx_dict from inverted_idx_matrix
    print('start constructing inverted index dictionary..')
    inverted_idx_dict = {}
    for word in new_vocab:
        inverted_idx_dict[word] = {}


    for i in tqdm(range(inverted_idx.shape[0])):
        passage = inverted_idx[i].toarray()
        nonzero_idx = np.nonzero(passage)
        for j, word in enumerate(new_vocab[nonzero_idx[1]]):
            inverted_idx_dict[word][rank_pid[i]] = passage[nonzero_idx][j]
    print('Finished!\n')

