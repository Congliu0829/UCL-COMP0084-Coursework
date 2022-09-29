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

def cal_score_tfidf(query, tf_idf_doc, idf, inverted_idx, new_vocab_dict, rank_pid, qid, qid_top1000):
    print('Start calculating score for tfidf')
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
        tf_q = temp / np.sum(temp)
        if np.sum(temp) == 0:
            print('Too many stop words have been deleted, please change the number of it.')

        #calculate tf_idf for one single query
        tf_idf_q = tf_q * idf

        #take out words in query (those elements have non-zero value in tf_q)
        nonzero_idx = np.nonzero(tf_q)[0]
        tf_idf_q_nz = tf_idf_q[nonzero_idx]

        #take out passages-sub-matrix in corresponding position
        p_tf_idf_doc_nz = tf_idf_doc[:, nonzero_idx].toarray() 

        #take out corresponding 1000 passages
        candidate_idx = np.zeros(qid_top1000[i].shape[0])
        for p, candidate in enumerate(qid_top1000[i]):
            cur_idx = np.where(rank_pid == candidate)[0]
            candidate_idx[p] = cur_idx
        candidate_idx = candidate_idx.astype(np.int64)

        p_tf_idf_doc = tf_idf_doc[candidate_idx, :].toarray() 
        p_tf_idf_doc_nz = p_tf_idf_doc_nz[candidate_idx]

        temp_rank_pid = qid_top1000[i]

        #calculate score, take out corresponding top 100 pid
        temp_norm = np.linalg.norm(tf_idf_q) * np.linalg.norm(p_tf_idf_doc, axis=1)
        nonzero_norm_idx = np.nonzero(temp_norm)[0]
        nonzero_norm = temp_norm[nonzero_norm_idx]

        temp_score = np.zeros(len(qid_top1000[i]))
        temp_score[nonzero_norm_idx] = np.sum(tf_idf_q_nz * p_tf_idf_doc_nz[nonzero_norm_idx], axis=1) / nonzero_norm

        temp_res_pid = temp_rank_pid[np.argsort(temp_score)[::-1][:100]]
        temp_res_score = temp_score[np.argsort(temp_score)[::-1][:100]]


        #save pid, save score, save qid
        res_score.extend(temp_res_score)
        temp_len = 100 if len(p_tf_idf_doc_nz) >= 100 else len(p_tf_idf_doc_nz)
        res_qid.extend([qid[i] for _ in range(temp_len)])
        res_pid.extend(temp_res_pid)

    
    print('Finished!\n')
    #write into csv file
    data = {'qid': res_qid, 'pid': res_pid, 'score': res_score}
    data_df = pd.DataFrame(data)
    data_df.to_csv('tfidf.csv',index=False,header=False)

        
def cal_score_bm25(query, tf_idf_doc, idf_bm25, inverted_idx, new_vocab_dict, rank_pid, qid, Ld, qid_top1000):
    print('Start calculating score for BM25')
    #hyperparameters for BM25
    k1 = 1.2
    k2 = 100
    b = 0.75

    #initialize store box
    res_score = []
    res_pid = []
    res_qid = []

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

        L = (Ld/np.mean(Ld))[candidate_idx].reshape(-1, 1)
        
        #calculate score, take out corresponding top 100 pid
        nonzero_p_idx = np.nonzero(np.sum(p_tf_doc, axis=1))[0]

        temp_score = np.zeros(len(qid_top1000[i]))
        temp_score[nonzero_p_idx] = np.sum(p_idf * (k1 + 1) * p_tf_doc[nonzero_p_idx] * (k2 + 1) * tf_q / ((k1*((1-b) + b*(L[nonzero_p_idx])) + p_tf_doc[nonzero_p_idx]) * (k2 + tf_q)), axis=1)
        temp_res_pid = temp_rank_pid[np.argsort(temp_score)[::-1][:100]]
        temp_res_score = temp_score[np.argsort(temp_score)[::-1][:100]]


        #save pid, save score, save qid
        res_score.extend(temp_res_score)
        temp_len = 100 if len(p_tf_doc) >= 100 else len(p_tf_doc)
        res_qid.extend([qid[i] for _ in range(temp_len)])
        res_pid.extend(temp_res_pid)


    print('Finished!\n')
    #write into csv file
    data = {'qid': res_qid, 'pid': res_pid, 'score': res_score}
    data_df = pd.DataFrame(data)
    data_df.to_csv('bm25.csv',index=False,header=False)



if __name__ == '__main__':

    print('Preprocessing......It takes about two minutes.')
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
            line = line.strip('\n').lower().split('\t')
            line[0] = int(line[0])
            line[1] = re.sub(u"([^\u0061-\u007a\u0030-\u0039\u0020])", "", line[1])
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


    # #calculate idf and idf_bm25 of new_vocab 
    print('start constructing idf for tfidf and bm25...')
    num_passage = inverted_idx.shape[0]
    num_vocab = inverted_idx.shape[1]
    idf = np.zeros(num_vocab)
    idf_bm25 = np.zeros(num_vocab)
    block = 1000

    for i in tqdm(range(0, (num_vocab // block)*block, block)):
        temp = inverted_idx[:, i:i+block].toarray()  
        count_zero_temp = np.count_nonzero(temp, axis=0) 
        idf[i:i+block] = np.log10(num_passage / count_zero_temp)
        idf_bm25[i:i+block] = np.log(((num_passage - count_zero_temp) + 0.5)/ (count_zero_temp + 0.5))
        

    temp = inverted_idx[:, i+block:].toarray()
    idf[i+block:] = np.log10(num_passage / np.count_nonzero(temp, axis=0))
    idf_bm25[i+block:] =  np.log((num_passage - np.count_nonzero(temp, axis=0) + 0.5)/ (np.count_nonzero(temp, axis=0) + 0.5))
    print('Finished!\n')

    
    #construct tfidf doc
    print('start constructing tfidf for documents')
    data = []
    row = []
    col = []

    for i in tqdm(range((inverted_idx.shape[0]))):
        passage = inverted_idx[i].toarray()[0]
        nonzero_idx = np.nonzero(passage)[0]
        #calculate frequency of word in one single passage
        passage = passage / sum(passage[nonzero_idx])
        #tf*idf
        nonzero_item = passage[nonzero_idx] * idf[nonzero_idx]
        data.extend(nonzero_item)
        row.extend([i for _ in range(len(nonzero_idx))])
        col.extend(nonzero_idx)

    tf_idf_doc = sparse.csr_matrix((data, (row, col)), shape=(inverted_idx.shape))
    print('Finished!\n')



    #tfidf
    cal_score_tfidf(query, tf_idf_doc, idf, inverted_idx, new_vocab_dict, rank_pid, qid, qid_top1000)
    
    #bm25
    cal_score_bm25(query, tf_idf_doc, idf_bm25, inverted_idx, new_vocab_dict, rank_pid, qid, Ld, qid_top1000)