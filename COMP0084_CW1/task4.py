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

def language_model(query, inverted_idx, new_vocab_dict, rank_pid, qid, Ld, qid_top1000, mode='laplace'):
    print('start calculating score for ' + mode + ' smoothing...')
    res_score = []
    res_pid = []
    res_qid = []

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
        freq_q = temp 
        if np.sum(temp) == 0:
            print('Too many stop words have been deleted, please change the number of it.')

        #take out words in query (those elements have non-zero value in tf_q)
        nonzero_idx = np.nonzero(freq_q)[0] 
        freq_q = freq_q[nonzero_idx] 

        #take out corresponding 1000 passages
        candidate_idx = np.zeros(qid_top1000[i].shape[0])
        for p, candidate in enumerate(qid_top1000[i]):
            cur_idx = np.where(rank_pid == candidate)[0]
            candidate_idx[p] = cur_idx
        candidate_idx = candidate_idx.astype(np.int64)

        temp_rank_pid = qid_top1000[i]

        #calculate score
        if mode == 'laplace':
            #take out passage-sub-matrix in corresponding position for laplace
            q_laplace_prob = np.log((inverted_idx[:, nonzero_idx].toarray() + 1) / (len(new_vocab_dict) + np.array(Ld)).reshape(-1, 1))
            temp_score = np.sum(freq_q * q_laplace_prob[candidate_idx], axis=1) 
        else:
            epsilon = 0.1
            #take out passage-sub-matrix in corresponding position for lindstone
            q_lindstone_prob =  np.log((inverted_idx[:, nonzero_idx].toarray() + epsilon) / (len(new_vocab_dict)*epsilon + np.array(Ld)).reshape(-1, 1))
            temp_score = np.sum(freq_q * q_lindstone_prob[candidate_idx], axis=1)

        temp_res_pid = temp_rank_pid[np.argsort(temp_score)[::-1][:100]]
        temp_res_score = temp_score[np.argsort(temp_score)[::-1][:100]]

        #save pid, save score, save qid
        res_score.extend(temp_res_score)
        res_qid.extend([qid[i] for _ in range(len(temp_res_pid))])
        res_pid.extend(temp_res_pid)

    print('Finished!\n')
    #write into csv file
    data = {'qid': res_qid, 'pid': res_pid, 'score': res_score}
    data_df = pd.DataFrame(data)
    data_df.to_csv(mode+'.csv',index=False,header=False)

def dirichlet(query, inverted_idx, new_vocab_dict, rank_pid, qid, Ld, vocab_for_dirichlet, qid_top1000):
    print('start calculating score for dirichlet...')
    res_score = []
    res_pid = []
    res_qid = []

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
        freq_q = temp
        if np.sum(temp) == 0:
            print('Too many stop words have been deleted, please change the number of it.')

        #take out words in query (those elements have non-zero value in tf_q)
        nonzero_idx = np.nonzero(freq_q)[0]
        freq_q = freq_q[nonzero_idx]

        #take out corresponding 1000 passages
        candidate_idx = np.zeros(qid_top1000[i].shape[0])
        for p, candidate in enumerate(qid_top1000[i]):
            cur_idx = np.where(rank_pid == candidate)[0]
            candidate_idx[p] = cur_idx
        candidate_idx = candidate_idx.astype(np.int64)

        temp_rank_pid = qid_top1000[i]

        mu = 50
        lam = (np.array(Ld)/(np.array(Ld) + mu)).reshape(-1, 1) # (num_passage, 1)
        q_dirichlet_prob = lam * (inverted_idx[:, nonzero_idx].toarray() / np.array(Ld).reshape(-1, 1)) + (1 - lam) * vocab_for_dirichlet[nonzero_idx].reshape(1, -1) 

        temp_score = np.sum(np.log(q_dirichlet_prob[candidate_idx]) * freq_q, axis=1)

        temp_res_pid = temp_rank_pid[np.argsort(temp_score)[::-1][:100]]
        temp_res_score = temp_score[np.argsort(temp_score)[::-1][:100]]

        #save pid, save score, save qid
        res_score.extend(temp_res_score)
        res_qid.extend([qid[i] for _ in range(len(temp_res_pid))])
        res_pid.extend(temp_res_pid)

    print('Finished!\n')
    #write into csv file
    data = {'qid': res_qid, 'pid': res_pid, 'score': res_score}
    data_df = pd.DataFrame(data)
    data_df.to_csv('dirichlet.csv',index=False,header=False)

    
if __name__ == '__main__':
    print('Preprocessing...It takes about two mins')
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

    #read query
    whole_query = []
    with open('coursework-1-data/test-queries.tsv', 'r') as f: 
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            line[0] = int(line[0])
            # line[1] = re.sub('[^a-z^A-Z^0-9]+', ' ', line[1])
            line[1] = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039\u0020])", "", line[1])
            whole_query.append(line)

    whole_query = np.array(whole_query)
    qid = whole_query[:, 0].astype(np.int64)
    query = whole_query[:, 1]

    #load top1000
    qid_top1000 = []

    pid_top1000 = whole_passage[:, 1].astype(np.int64)
    for single_qid in qid:
        idx = np.where(single_qid == whole_passage[:, 0].astype(np.int64))[0]
        qid_top1000.append(pid_top1000[idx])
    
    del whole_passage, whole_passage_p
    # remove stop words in vocab
    num_stop_words = 20
    new_vocab = np.array(np.array(vocab)[num_stop_words: ])[:, 0]
    new_vocab_dict = {}

    for i, word in enumerate(new_vocab):
        new_vocab_dict[word] = i
    print('Finished!\n')
    
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




    # remove stop words in vocab
    num_stop_words = 20
    vocab_for_dirichlet = np.array(np.array(vocab)[num_stop_words: ])
    vocab_for_dirichlet = vocab_for_dirichlet[:, 1].astype(np.int64)
    vocab_for_dirichlet = vocab_for_dirichlet / np.sum(vocab_for_dirichlet)

    language_model(query, inverted_idx, new_vocab_dict, rank_pid, qid, Ld, qid_top1000, mode='laplace')

    language_model(query, inverted_idx, new_vocab_dict, rank_pid, qid, Ld, qid_top1000, mode='lidstone')

    dirichlet(query, inverted_idx, new_vocab_dict, rank_pid, qid, Ld, vocab_for_dirichlet, qid_top1000)