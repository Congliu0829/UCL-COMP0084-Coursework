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

def gen_zipf_dist(n):
    zipf_dist = np.zeros(n)
    denum = sum([1/i for i in range(1, n+1)])
    for i in range(1, n+1):
        zipf_dist[i-1] = 1/(i*denum)
    return zipf_dist

def plot1(freq_list, zipf_dist):
    plt.plot(np.arange(len(freq_list)), freq_list, label='data')
    plt.plot(np.arange(len(freq_list)), zipf_dist, '--', label="theory (Zipf's law)")
    plt.grid()
    plt.legend()
    plt.xlabel('Term frequency ranking')
    plt.ylabel('Term prob. of occurrence')
    plt.title("Plot of Zipf's law distribution and empr distribution")
    plt.savefig('D1_0.png', dpi=500)
    plt.show()

def plot2(freq_list, zipf_dist):
    plt.loglog(np.arange(1, len(freq_list)+1), freq_list, label='data') 
    plt.loglog(np.arange(1, len(freq_list)+1), zipf_dist, '--', label="theory (Zipf's law)")
    plt.xlim(left=1)
    plt.grid()
    plt.legend()
    plt.xlabel('Term frequency ranking (log)')
    plt.ylabel('Term prob. of occurrence (log)')
    plt.title("Log-Log plot of Zipf's law distribution and empr distribution")
    plt.savefig('D1_1.png', dpi=500)
    plt.show()

if __name__ == '__main__':
    vocab = text_preprocessing('coursework-1-data/passage-collection.txt')
    freq_list = np.array(vocab)[:, 1]
    freq_list = freq_list.astype(np.int64)
    freq_list = freq_list / sum(freq_list)
    zipf_dist = gen_zipf_dist(len(freq_list))

    plot1(freq_list, zipf_dist)
    plt.clf()
    plot2(freq_list, zipf_dist)