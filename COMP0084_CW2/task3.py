#data preprocessing
import transformers
from transformers import BertTokenizerFast, BertModel
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import os
from bayes_opt import BayesianOptimization
import xgboost as xgb

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

def neg_sampling(qids, xTr, yTr, ratio=5):
    ranked_qid = qids[np.argsort(qids)]
    yTr = yTr[np.argsort(qids)]
    xTr = xTr[np.argsort(qids)]

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
    

    sample_xTr = []
    sample_yTr = []
    sample_qids = []

    for i, qid in enumerate(tqdm(np.unique(qids))):
        idx1 = np.where(yTr[counter[i]] == 1)
        num_pos = idx1[0].shape[0]

        #positive sample
        sample_xTr.extend(xTr[counter[i]][idx1])
        sample_yTr.extend(yTr[counter[i]][idx1])
        sample_qids.extend([qid]*(len(xTr[counter[i]][idx1])))

        #negative sample
        idx0 = np.delete(np.arange(len(counter[i])), idx1)
        np.random.shuffle(idx0)
        sample_xTr.extend(xTr[counter[i]][idx0[:ratio*num_pos]])
        sample_yTr.extend(yTr[counter[i]][idx0[:ratio*num_pos]])
        sample_qids.extend([qid]*len(xTr[counter[i]][idx0[:ratio*num_pos]]))

    return sample_xTr, sample_yTr, sample_qids
        
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
    return DCG/optDCG.sum() if optDCG != 0 else 0

def make_group(qids, yTr, xTr):
    xTr = np.array(xTr)
    yTr = np.array(yTr)
    qids = np.array(qids)

    ranked_qid = qids

    # ranked_qid = qids[np.argsort(qids)]
    # yTr = yTr[np.argsort(qids)]
    # xTr = xTr[np.argsort(qids)]


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

    group = [len(counter[i]) for i in range(len(counter))]

    return xTr, yTr, group




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
yTr = train_labels

#concatenate passage and query to input into logistic regression model
xTe = np.concatenate((test_all_embedding_q, test_all_embedding_p), 1)
yTe = test_labels

del train_all_embedding_p, train_all_embedding_q, test_all_embedding_p, test_all_embedding_q

sample_xTr, sample_yTr, sample_qids = neg_sampling(train_qids, xTr, yTr, ratio=5)

xTr_sample, yTr_sample, groups = make_group(sample_qids, sample_yTr, sample_xTr)

def xgb_cv(lr, colsample_bytree, max_depth, n_estimators, subsample):
    qids = sample_qids
    folds = 5
    n = len(groups)
    c_len = n // folds
    group_chunks = [groups[i*c_len:(i+1)*c_len] for i in range(folds-1)]
    group_chunks.append(groups[(folds-1)*c_len:])

    xTr_chunks = []
    yTr_chunks = []
    qid_chunks = []
    for i in range(folds):
        start_idx = 0
        chunk_length = sum(group_chunks[i])
        xTr_chunks.append(xTr_sample[start_idx:start_idx+chunk_length])
        yTr_chunks.append(yTr_sample[start_idx:start_idx+chunk_length])
        qid_chunks.append(qids[start_idx:start_idx+chunk_length])
        start_idx += chunk_length
    
    ndcg_list = []
    ap_list = []
    for i in range(folds):
        idx_list = np.arange(folds)
        idx_list = np.delete(idx_list, i)

        xVal_cv = xTr_chunks[i]
        yVal_cv = yTr_chunks[i]
        gVal_cv = group_chunks[i]
        qids_cv = qid_chunks[i]

        xTr_cv = np.concatenate([xTr_chunks[j] for j in idx_list][:])
        yTr_cv = np.concatenate([yTr_chunks[j] for j in idx_list][:])
        group_cv = np.concatenate([group_chunks[j] for j in idx_list][:])

        rank_model = xgb.XGBRanker(  
            tree_method='gpu_hist',
            booster='gbtree',
            objective='rank:pairwise',
            random_state=42, 
            learning_rate=lr,
            colsample_bytree=colsample_bytree, 
            eta=0.05, 
            max_depth=int(max_depth), 
            n_estimators=int(n_estimators), 
            subsample=subsample,
            )
        rank_model.fit(xTr_cv, yTr_cv, group_cv, verbose=True)


        ranked_qid = qids_cv

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

        m_ap = 0
        m_ndcg = 0

        for count in tqdm(counter):
            sub_xTe = xVal_cv[count]
            pred_yTe = rank_model.predict(sub_xTe)
            sort_idx = np.argsort(pred_yTe)[::-1]
            sub_yTe = yVal_cv[count]
            true_label = sub_yTe[sort_idx]
            ap = Average_Precision(true_label)
            ndcg = NDCG(true_label)
            m_ap += ap
            m_ndcg += ndcg
        
        ndcg_list.append(m_ndcg/len(counter))
        ap_list.append(m_ap/len(counter))

    return np.mean(ap_list)

xgb_bo = BayesianOptimization(
    xgb_cv,
    {'lr': (0.001, 0.1),
    'colsample_bytree': (0.5, 0.9),
    'max_depth': (4, 10),
    'n_estimators': (300, 800),
     'subsample': (0.5, 0.9)})


xgb_bo.maximize(init_points=0,n_iter=10,)

print('Best option: ', xgb_bo.max)


def make_group(qids, yTr, xTr):
    xTr = np.array(xTr)
    yTr = np.array(yTr)
    qids = np.array(qids)

    ranked_qid = qids[np.argsort(qids)]
    yTr = yTr[np.argsort(qids)]
    xTr = xTr[np.argsort(qids)]


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

    group = [len(counter[i]) for i in range(len(counter))]

    return xTr, yTr, group

xTr, yTr, groups = make_group(train_qids, yTr, xTr)

rank_model = xgb.XGBRanker(  
    tree_method='gpu_hist',
    booster='gbtree',
    objective='rank:pairwise',
    random_state=42, 
    learning_rate=0.033832164874053106,
    colsample_bytree=0.6878421461159999, 
    eta=0.05, 
    max_depth=10, 
    n_estimators=378, 
    subsample=0.9,
    )

rank_model.fit(xTr, yTr, group=groups, verbose=True)

def mean_metric_xgb(rank_model, test_qids, test_pids, xTe, yTe, write=True):
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
        pred_yTe = rank_model.predict(sub_xTe)
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
        res_algoname.extend(['LM' for _ in range(len(sort_idx))])

    if write:
        data = {'qid': res_qid, 'A1': res_A1, 'pid': res_pid, 'rank': res_rank, 'score': res_score, 'algoname': res_algoname}
        data_df = pd.DataFrame(data)
        data_df.to_csv('LM.txt',index=False,header=False, sep=' ')
    
    return m_ap / len(counter), m_ndcg / len(counter)
    
m_ap, m_ndcg = mean_metric_xgb(rank_model, test_qids, test_pids, xTe, yTe, write='False')
print('Average precision for LambdaMart is {0:.4f}'.format(m_ap))
print('NDCG for LambdaMart is {0:.4f}'.format(m_ndcg))
        


        


