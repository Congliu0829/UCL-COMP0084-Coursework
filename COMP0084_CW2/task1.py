import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import sparse
import os

def Average_Precision(true_label):
    '''
        input: true_label, list, this list contains the truth label of retrieved passage with order. 
        For example, the score for 5 retrieved passages are S = [0.2, 0.3, 0.4, 0.5, 0.6], the truth labels of these passages are T = [1, 1, 0, 0, 0]
        Then the input true_label is sorted T, which is sorted by S, as T' = [0, 0, 0, 1, 1]
        This means that L' contains both order information as well as truth label information.
    '''
    # AP for single query
    rela_idx = np.where(true_label == 1)[0]
    n_rela_passage = len(rela_idx)
    denom = rela_idx + 1
    numerator = np.arange(1, n_rela_passage+1)
    return (numerator/denom/n_rela_passage).sum()

def NDCG(true_label):
    '''
        Input: true_label, same as the input of Average_Precision.
    '''
    # NDCG for single query
    DCG = np.sum((2**true_label - 1) / np.log2(np.arange(1, len(true_label)+1) + 1))
    n_rela_passage = int(np.sum(true_label))
    opt_rela_score = np.zeros(len(true_label))
    opt_rela_score[:n_rela_passage] = 1
    optDCG = np.sum((2**opt_rela_score - 1) / np.log2(np.arange(1, len(true_label)+1) + 1))
    return DCG/optDCG if optDCG != 0 else 0


def text_preprocessing(f):
    vocab = dict()
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


def cal_score_bm25(query, idf_bm25, inverted_idx, new_vocab_dict, rank_pid, Ld):
    #hyperparameters for BM25
    k1 = 1.2
    k2 = 100
    b = 0.75

    #initialize store box
    res_score = []
    res_pid = []
    res_qid = []
    res_relavance = []

    m_ap = 0
    m_ndcg = 0

    # deal with one query at a time
    for i, line in enumerate(tqdm(query)):
        line = line.split(' ')
        line.remove('') if '' in line else line

        temp = np.zeros(inverted_idx.shape[1])

        line_vocab = {}

        #construct small dictionary for one single query
        for word in line:
            if word in line_vocab:
                line_vocab[word] += 1
            else:
                line_vocab[word] = 1

        #record the frequency 
        for word in line:
            if word in new_vocab_dict:
                temp[new_vocab_dict[word]] = line_vocab[word]

        #calculate the tf for one single query
        tf_q = temp
        if np.sum(temp) == 0:
            print('Too many stop words have been deleted, please change the number of it.')

        #take out words in query (those elements have non-zero value in tf_q)
        nonzero_idx = np.nonzero(tf_q)[0]
        tf_q = tf_q[nonzero_idx]

        #take out passages-sub-matrix in corresponding position
        p_idf = idf_bm25[nonzero_idx].reshape(1, -1)
        
        p_tf_doc = inverted_idx[:, nonzero_idx].toarray()

        #take out corresponding 1000 passages
        candidate_idx = np.zeros(qid_top1000[i].shape[0])
        for p, candidate in enumerate(qid_top1000[i]):
            cur_idx = np.where(rank_pid == candidate)[0]
            candidate_idx[p] = cur_idx
        candidate_idx = candidate_idx.astype(np.int64)
        p_tf_doc = p_tf_doc[candidate_idx]

        temp_rank_pid = qid_top1000[i]
        temp_rank_relevance = qid_top1000_relavance[i]
        
        L = (Ld/np.mean(Ld))[candidate_idx].reshape(-1, 1)

        
        #calculate score, take out corresponding top 100 pid
        nonzero_p_idx = np.nonzero(np.sum(p_tf_doc, axis=1))[0]

        temp_score = np.zeros(len(qid_top1000[i]))
        temp_score[nonzero_p_idx] = np.sum(p_idf * (k1 + 1) * p_tf_doc[nonzero_p_idx] * (k2 + 1) * tf_q / ((k1*((1-b) + b*(L[nonzero_p_idx])) + p_tf_doc[nonzero_p_idx]) * (k2 + tf_q)), axis=1)
        temp_res_pid = temp_rank_pid[np.argsort(temp_score)[::-1][:100]]
        temp_res_score = temp_score[np.argsort(temp_score)[::-1][:100]]
        temp_res_relavance = temp_rank_relevance[np.argsort(temp_score)[::-1][:100]]


        ap = Average_Precision(temp_res_relavance)
        m_ap += ap

        ndcg = NDCG(temp_res_relavance)
        m_ndcg += ndcg
        
        
    return m_ap / len(query), m_ndcg / len(query)


# read passage
whole_passage = []
with open('part2/validation_data.tsv', 'r') as f: 
    i = 0
    for line in f.readlines():
        if i == 0:
            pass
            i += 1
        else:
            line = line.strip('\n').lower().split('\t')
            line[0] = int(line[0]) #qid
            line[1] = int(line[1]) #pid
            line[2] = re.sub(u"([^/u0061-\u007a\u0030-\u0039\u0020])", "", line[2]) #query  
            line[3] = re.sub(u"([^\u0061-\u007a\u0030-\u0039\u0020])", "", line[3]) #passage
            line[4] = int(float(line[4])) #relavance score
            whole_passage.append(line)


    
whole_passage = np.array(whole_passage)

pid = whole_passage[:, 1].astype(np.int64)

rank_pid, idx_p = np.unique(pid, return_index=True)

whole_passage_p = whole_passage[idx_p]

uni_passage = whole_passage_p[:, -2]

vocab = text_preprocessing(uni_passage)

# remove stop words in vocab
num_stop_words = 20
new_vocab = np.array(np.array(vocab)[num_stop_words: ])[:, 0]
new_vocab_dict = {}

for i, word in enumerate(new_vocab):
    new_vocab_dict[word] = i

#construct inverted_idx matrix
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

if not os.path.exists('idf_bm25.npy'):
    # #calculate idf_bm25 of new_vocab 
    num_passage = inverted_idx.shape[0]
    num_vocab = inverted_idx.shape[1]
    idf_bm25 = np.zeros(num_vocab)
    block = 5000

    for i in tqdm(range(0, (num_vocab // block)*block, block)):
        temp = inverted_idx[:, i:i+block].toarray()  
        count_zero_temp = np.count_nonzero(temp, axis=0) 
        idf_bm25[i:i+block] = np.log(((num_passage - count_zero_temp) + 0.5)/ (count_zero_temp + 0.5))
        
    temp = inverted_idx[:, i+block:].toarray()
    idf_bm25[i+block:] =  np.log((num_passage - np.count_nonzero(temp, axis=0) + 0.5)/ (np.count_nonzero(temp, axis=0) + 0.5))

else:
    idf_bm25 = np.load('idf_bm25.npy')

#obtain corresponding passages
qid = whole_passage[:, 0].astype(np.int64)
_, query_idx = np.unique(qid, return_index=True)
query = whole_passage[:, 2][query_idx] #ranked query
ranked_qid = qid[np.argsort(qid)]
last = ranked_qid[0]

counter = []
cur_counter = []
for i, id in enumerate(tqdm(ranked_qid)):
    cur = id
    if cur != last:
        counter.append(cur_counter)
        cur_counter = [i]
    else:
        cur_counter.append(i)
    last = cur
counter.append(cur_counter)

ranked_qid_whole_passage = whole_passage[np.argsort(qid)]

qid_top1000 = []
qid_top1000_relavance = []

pid_top1000 = ranked_qid_whole_passage[:, 1].astype(np.int64) #ranked qid corresponding pid
pid_top1000_relavance = ranked_qid_whole_passage[:, -1].astype(np.int64) #ranked qid score

for count in tqdm(counter):
    qid_top1000.append(pid_top1000[count]) #takeout corresponding 1000 candidate
    qid_top1000_relavance.append(pid_top1000_relavance[count]) #take out corresponding 1000 candidate relavance score


#obtain metric
m_ap, m_ndcg = cal_score_bm25(query, idf_bm25, inverted_idx, new_vocab_dict, rank_pid, Ld)

print('Average precision for bm25 is {0:.4f}'.format(m_ap))
print('NDCG for bm25 is {0:.4f}'.format(m_ndcg))

    