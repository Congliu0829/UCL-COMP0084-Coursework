import pandas as pd
import numpy as np
from tqdm import tqdm 
import os



class LogisticRegression:
    def __init__(self, xTr, yTr, lr, epochs=200):
        np.random.seed(42)
        self.xTr = xTr
        self.yTr = yTr
        self.w = np.random.rand(xTr.shape[1])
        self.lr = lr
        self.epochs = epochs
        self.mini_batch_size = 256
        

    def forward(self, x):
        return 1 / (1 + np.exp(x@-self.w))
    
    def criterion(self, pred, target):
        # return 1/len(target) * -(target*(np.log(pred)) + (1-target)*np.log(1-pred)).sum()
        return  -(target*(np.log(pred)) + (1-target)*np.log(1-pred)).sum()



    def train(self):
        all_loss = []
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0
            #Obtain data
            pred = self.forward(self.xTr)
            loss = self.criterion(pred, self.yTr)
            grad = -self.lr / len(self.xTr) * np.sum((self.yTr - pred) * self.xTr.T, 1)
            self.w -= grad
            total_loss += loss

            print("Epoch:\t", epoch, "Averaged loss of the epoch:", total_loss.item())
            all_loss.append(loss)

        return all_loss


#data preprocessing
import transformers
from transformers import BertTokenizerFast, BertModel
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#obtain train and test data
col=['qid','pid','query','passage','relevancy']

train_data=pd.read_csv("part2/train_data.tsv", sep='\t', header=None, names=col)
train_data=pd.DataFrame(train_data)
train_data = train_data.iloc[1:]

train_passages = train_data['passage'].values
train_queries = train_data['query'].values
train_pids = train_data['pid'].values.astype(np.int64)
train_qids = train_data['qid'].values.astype(np.int64)
train_labels = train_data['relevancy'].values.astype(np.float64).astype(np.int64)

test_data=pd.read_csv("part2/validation_data.tsv", sep='\t', header=None, names=col)
test_data=pd.DataFrame(test_data)
test_data = test_data.iloc[1:]

test_passages = test_data['passage'].values
test_queries = test_data['query'].values
test_pids = test_data['pid'].values.astype(np.int64)
test_qids = test_data['qid'].values.astype(np.int64)
test_labels = test_data['relevancy'].values.astype(np.float64).astype(np.int64)

tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertModel.from_pretrained("prajjwal1/bert-tiny").to(device)
model.eval()

class DataSeq(Dataset):
    '''
        Dataset for generating tokens of sequences.
    '''
    def __init__(self, queries, passages, labels):
        self.queries = queries
        self.passages = passages
        self.labels = labels
    
    def __getitem__(self, index):
        query, passage, label = self.queries[index], self.passages[index], self.labels[index]
        
        ids_query = tokenizer.batch_encode_plus([query], add_special_tokens=False, padding='max_length', max_length=50, truncation=True)
        ids_passage = tokenizer.batch_encode_plus([passage], add_special_tokens=False, padding='max_length', max_length=300, truncation=True)

        return np.array([ids_query['input_ids'], ids_query['attention_mask']]), np.array([ids_passage['input_ids'], ids_passage['attention_mask']]), label
        # return np.array([ids_query['input_ids'], ids_query['attention_mask']]), np.array([ids_passage['input_ids'], ids_passage['attention_mask']])
    def __len__(self):
        return len(self.labels)

def generate_embedding(queries, passages, labels, section):
    '''
        This function is used for generating embeddings using pre-processed data sequences.

        ***Note that if GPU memory runs out, we should decrease batchsize

        Input:
            -sequences: list, processed sequences
            -labels: list/np.array, corresponding labels for sequences
            -section: str, "train", "test", "dev", indicating which section we are loading.

        Output:
            -embedding-repre: np.array, embedding representation of selected data set.
    '''
    bs = 128
    dataset = DataSeq(queries=queries, passages=passages, labels=labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=2)

    progress_bar = tqdm(dataloader)
    all_embedding_q = []
    all_embedding_p = []
    with torch.no_grad():
        for i, (ids_q, ids_p, label) in enumerate(progress_bar):
            input_ids_q = torch.squeeze(ids_q[:,0,:,:]).to(device)
            attention_mask_q = torch.squeeze(ids_q[:,1,:,:]).to(device)   
            query_length = torch.count_nonzero(attention_mask_q, 1).unsqueeze(1)                 
            embedding_q = model(input_ids=input_ids_q,attention_mask=attention_mask_q)[0]
            attention_mask_q = attention_mask_q.unsqueeze(2).repeat(1, 1, 128) 
            embedding_q = torch.sum(embedding_q * attention_mask_q, 1) / query_length #bs, embedding_size

            input_ids_p = torch.squeeze(ids_p[:,0,:,:]).to(device)
            attention_mask_p = torch.squeeze(ids_p[:,1,:,:]).to(device)   
            passage_length = torch.count_nonzero(attention_mask_p, 1).unsqueeze(1)                                
            embedding_p = model(input_ids=input_ids_p,attention_mask=attention_mask_p)[0]
            attention_mask_p = attention_mask_p.unsqueeze(2).repeat(1, 1, 128) 
            embedding_p = torch.sum(embedding_p * attention_mask_p, 1) / passage_length #bs, embedding_size

            all_embedding_q.append(embedding_q.cpu().numpy())
            all_embedding_p.append(embedding_p.cpu().numpy())

    all_embedding_q = np.concatenate(all_embedding_q[:])
    all_embedding_p = np.concatenate(all_embedding_p[:])
    np.save('embedding_queries_{0:}.npy'.format(section), all_embedding_q)
    np.save('embedding_passages_{0:}.npy'.format(section), all_embedding_p)

    return all_embedding_q, all_embedding_p

if (not os.path.exists('embedding_passages_train.npy')) or (not os.path.exists('embedding_passages_test.npy')) :
    train_all_embedding_q, train_all_embedding_p = generate_embedding(train_queries, train_passages, train_labels, 'train')
    test_all_embedding_q, test_all_embedding_p = generate_embedding(test_queries, test_passages, test_labels, 'test')

else:
    train_all_embedding_p = np.load('embedding_passages_train.npy')
    train_all_embedding_q = np.load('embedding_queries_train.npy')

    test_all_embedding_p = np.load('embedding_passages_test.npy')
    test_all_embedding_q = np.load('embedding_queries_test.npy')



#concatenate passage and query to input into logistic regression model
xTr = np.concatenate((train_all_embedding_q, train_all_embedding_p), 1)
#add bias term
xTr = np.concatenate((xTr, np.ones((xTr.shape[0], 1))), 1)
yTr = train_labels

del train_all_embedding_p, train_all_embedding_q

#concatenate passage and query to input into logistic regression model
xTe = np.concatenate((test_all_embedding_q, test_all_embedding_p), 1)
#add bias term
xTe = np.concatenate((xTe, np.ones((xTe.shape[0], 1))), 1)
yTe = test_labels

del test_all_embedding_p, test_all_embedding_q

#train LR
lr1 = LogisticRegression(xTr, yTr, lr=0.5, epochs=500)
loss_1 = lr1.train()
del lr1

lr2 = LogisticRegression(xTr, yTr, lr=5e-2, epochs=500)
loss_2 = lr2.train()
del lr2

lr3 = LogisticRegression(xTr, yTr, lr=10, epochs=500)
loss_3 = lr3.train()

lr4 = LogisticRegression(xTr, yTr, lr=50, epochs=500)
loss_4 = lr4.train()
del lr4


#plot
import matplotlib.pyplot as plt
plt.plot(np.arange(1, len(loss_1)+1), loss_1, label='lr = 0.5')
plt.plot(np.arange(1, len(loss_2)+1), loss_2, label='lr = 0.05')
plt.plot(np.arange(1, len(loss_3)+1), loss_3, label='lr = 10')
plt.plot(np.arange(1, len(loss_4)+1), loss_4, label='lr = 50')
plt.xlabel('Epochs')
plt.ylabel('Epoch Loss')
plt.yscale("log")
plt.title('Comparsion on training loss of using different learning rate')
plt.legend()
plt.savefig('Comparsion_lr.png', dpi=400, transparent=True)

import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import sparse

def Average_Precision(true_label):
    # AP for single query
    rela_idx = np.where(true_label == 1)[0]
    n_rela_passage = len(rela_idx)
    denom = rela_idx + 1
    numerator = np.arange(1, n_rela_passage+1)
    return (numerator/denom/n_rela_passage).sum()

def NDCG(true_label):
    # NDCG for single query
    DCG = np.sum((2**true_label - 1) / np.log2(np.arange(1, len(true_label)+1) + 1))
    n_rela_passage = int(np.sum(true_label))
    opt_rela_score = np.zeros(len(true_label))
    opt_rela_score[:n_rela_passage] = 1
    optDCG = np.sum((2**opt_rela_score - 1) / np.log2(np.arange(1, len(true_label)+1) + 1))
    return DCG/optDCG if optDCG != 0 else 0

def mean_metric_lr(test_qids, xTe, yTe, test_pids, lr, write=False):
    ranked_qid = test_qids[np.argsort(test_qids)]
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

    xTe = xTe[np.argsort(test_qids)]
    yTe = yTe[np.argsort(test_qids)]
    test_pids = test_pids[np.argsort(test_qids)]
    uni_qids = np.unique(test_qids)
    
    m_ap = 0
    m_ndcg = 0

    res_qid = []
    res_pid = []
    res_score = []
    res_rank = []
    res_A1 = []
    res_algoname = []

    for i, count in enumerate(tqdm(counter)):
        sub_xTe = xTe[count]
        pred_yTe = lr.forward(sub_xTe)
        sort_idx = np.argsort(pred_yTe)[::-1][:]
        sub_yTe = yTe[count]
        true_label = sub_yTe[sort_idx]
        ap = Average_Precision(true_label)
        ndcg = NDCG(true_label)
        m_ap += ap
        m_ndcg += ndcg

        sub_pids = test_pids[count]

        res_qid.extend([uni_qids[i] for _ in range(len(sort_idx))])
        res_pid.extend(sub_pids[sort_idx])
        res_score.extend(pred_yTe[np.argsort(pred_yTe)[::-1][:]])
        res_rank.extend(np.arange(1, len(sort_idx)+1))
        res_A1.extend(['A1' for _ in range(len(sort_idx))])
        res_algoname.extend(['LR' for _ in range(len(sort_idx))])
    
    if write:
        data = {'qid': res_qid, 'A1': res_A1, 'pid': res_pid, 'rank': res_rank, 'score': res_score, 'algoname': res_algoname}
        data_df = pd.DataFrame(data)
        data_df.to_csv('LR.txt',index=False,header=False, sep=' ')

    
    return m_ap / len(counter), m_ndcg / len(counter)

m_ap, m_ndcg = mean_metric_lr(test_qids, xTe, yTe, test_pids, lr3, write=False)
print('Average precision for bm25 is {0:.4f}'.format(m_ap))
print('NDCG for bm25 is {0:.4f}'.format(m_ndcg))