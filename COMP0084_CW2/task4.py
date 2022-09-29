import os
import transformers
from transformers import BertTokenizerFast, BertModel
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import argparse
from torch import nn, optim
import torch.nn.functional as F
import random
import math

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
        
        # ids_query = tokenizer.batch_encode_plus([query], add_special_tokens=False, padding='max_length', max_length=50, truncation=True, return_tensors='pt').to(device)
        # ids_passage = tokenizer.batch_encode_plus([passage], add_special_tokens=False, padding='max_length', max_length=300, truncation=True, return_tensors='pt').to(device)
        # return model(**ids_query), model(**ids_passage), ids_query.attention_mask, ids_passage.attention_mask, label

        ids_query = tokenizer.batch_encode_plus([query], add_special_tokens=False, padding='max_length', max_length=50, truncation=True)
        ids_passage = tokenizer.batch_encode_plus([passage], add_special_tokens=False, padding='max_length', max_length=300, truncation=True)
        return np.array([ids_query['input_ids']]), np.array([ids_passage['input_ids']]), np.array([ids_query['attention_mask']]), np.array([ids_passage['attention_mask']]), label


    def __len__(self):
        return len(self.labels)

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=128,
                    choices=[64, 128, 256])
parser.add_argument('--vocab_size', type=int, default=len(tokenizer))
parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=128,
                    choices=[64, 128, 256])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--feed_forward', type=float, default=64)
parser.add_argument('--dropout', type=float, default=0.1) 
parser.add_argument('--num_head', type=int, default=4)
parser.add_argument('--num_transformer_layer', default=2)
args = parser.parse_args(args=[])




class PositionalEncoding(nn.Module):
    def __init__(self, args, max_len=350):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(max_len, self.args.hidden_size)
        self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        """
        return (l b d)
        """
        position_ids = self.position_ids[:x.size(0)] #1, seq_len
        position_ids = position_ids.repeat(x.shape[1], 1) #bs, seq_len
        position_ids = self.embeddings(position_ids) #bs, seq_len, embedding_size
        position_ids = position_ids.transpose(0, 1)

        return x + position_ids


class BinaryClassifier(torch.nn.Module):
    def __init__(self, args):
        super(BinaryClassifier, self).__init__()
        self.args = args
        ##positional encoder
        self.pos_encoder = PositionalEncoding(self.args) 
        ##Multi-atten
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.args.hidden_size, nhead=self.args.num_head, dim_feedforward=self.args.feed_forward, dropout=self.args.dropout)
        self.multi_atten = nn.TransformerEncoder(encoder_layer, num_layers=self.args.num_transformer_layer)
        
        self.fc = nn.Linear(self.args.hidden_size, self.args.hidden_size) 
        self.dropout = nn.Dropout(self.args.dropout)

        self.down1 = nn.Linear(self.args.hidden_size, self.args.hidden_size//4)
        self.down2 = nn.Linear(self.args.hidden_size//4, self.args.hidden_size//4)
        self.down3 = nn.Linear(self.args.hidden_size//4, self.args.hidden_size//8)
        self.out = nn.Linear(self.args.hidden_size//8, 1)

        # self.out = nn.Linear(self.args.hidden_size, 1)
        
        self.relu = nn.ReLU()
        


    def forward(self, query, passage, mask_q, mask_p):
        #query: bs, seq_len_q, embedding_size
        #passage: bs, seq_len_p, embedding_size


        x = torch.cat((query, passage), 1) #bs, seq_len_all
        mask = torch.cat((mask_q, mask_p), 1) #bs, seq_len_all

        #embedding layer
        x = x * math.sqrt(self.args.hidden_size) #bs, seq_len_all, embedding_size
        x = self.dropout(x)

        #multi-atten
        x = torch.transpose(x, 0, 1) # seq_len, bs, embedding_size
        x = self.pos_encoder(x)
        x = self.multi_atten(x, src_key_padding_mask=(mask==0)) # seq_len, bs, embedding_size
        x = torch.transpose(x, 0, 1) #bs, seq_len, embedding_size
        
        #pooling
        x = torch.mean(x, 1)
        x = self.relu(self.fc(x))

        #down
        x = self.relu(self.down1(x))
        x = self.relu(self.down2(x))
        x = self.relu(self.down3(x))
        x = self.out(x)

        # x = self.out(x)

        x = nn.Sigmoid()(x)

        return x

'''
    Parallel encoding for query and passage, respectively, with m_ndcg = 0.14 and m_ap = 0.02.
    Model is not performing well because query and passage are not computing attention with each other, thus model is hard to obtain interactive info
    between passage and query, thus harder to predict if they are relevant of not.
'''
# import torch
# from torch import nn, optim
# import torch.nn.functional as F
# import random
# import math


# class PositionalEncoding(nn.Module):
#     def __init__(self, args, max_len=350):
#         super().__init__()
#         self.args = args
#         self.embeddings = nn.Embedding(max_len, self.args.hidden_size)
#         self.register_buffer('position_ids', torch.arange(max_len))

#     def forward(self, x):
#         """
#         return (l b d)
#         """
#         position_ids = self.position_ids[:x.size(0)] #1, seq_len
#         position_ids = position_ids.repeat(x.shape[1], 1) #bs, seq_len
#         position_ids = self.embeddings(position_ids) #bs, seq_len, embedding_size
#         position_ids = position_ids.transpose(0, 1)

#         return x + position_ids


# class BinaryClassifier(torch.nn.Module):
#     def __init__(self, args):
#         super(BinaryClassifier, self).__init__()
#         self.args = args
#         ##positional encoder
#         self.pos_encoder_q = PositionalEncoding(self.args, max_len=50)
#         self.pos_encoder_p = PositionalEncoding(self.args, max_len=300)
#         ##Multi-atten
#         encoder_layer_q = nn.TransformerEncoderLayer(d_model=self.args.hidden_size, nhead=self.args.num_head, dim_feedforward=self.args.feed_forward, dropout=self.args.dropout)
#         self.multi_atten_q = nn.TransformerEncoder(encoder_layer_q, num_layers=self.args.num_transformer_layer)

#         encoder_layer_p = nn.TransformerEncoderLayer(d_model=self.args.hidden_size, nhead=self.args.num_head, dim_feedforward=self.args.feed_forward, dropout=self.args.dropout)
#         self.multi_atten_p = nn.TransformerEncoder(encoder_layer_p, num_layers=self.args.num_transformer_layer)

        
#         self.fc_p = nn.Linear(self.args.hidden_size, self.args.hidden_size) 
#         self.fc_q = nn.Linear(self.args.hidden_size, self.args.hidden_size) 

#         self.dropout = nn.Dropout(self.args.dropout)

#         self.down1 = nn.Linear(self.args.hidden_size, self.args.hidden_size//4)
#         self.down2 = nn.Linear(self.args.hidden_size//4, self.args.hidden_size//4)
#         self.down3 = nn.Linear(self.args.hidden_size//4, self.args.hidden_size//8)
#         self.out = nn.Linear(self.args.hidden_size//8, 1)

#         # self.out = nn.Linear(self.args.hidden_size, 1)
        
#         self.relu = nn.ReLU()
        


#     def forward(self, query, passage, mask_q, mask_p):
#         #query: bs, seq_len_q, embedding_size
#         #passage: bs, seq_len_p, embedding_size


#         #embedding layer
#         x_q = query * math.sqrt(self.args.hidden_size) #bs, seq_len_all, embedding_size
#         x_q = self.dropout(x_q)

#         #multi-atten
#         x_q = torch.transpose(x_q, 0, 1) # seq_len, bs, embedding_size
#         x_q = self.pos_encoder_q(x_q)
#         x_q = self.multi_atten_q(x_q, src_key_padding_mask=(mask_q==0)) # seq_len, bs, embedding_size
#         x_q = torch.transpose(x_q, 0, 1) #bs, seq_len, embedding_size
        
#         #pooling
#         x_q = torch.mean(x_q, 1)
#         x_q = self.relu(self.fc_q(x_q))

#         #embedding layer
#         x_p = passage * math.sqrt(self.args.hidden_size) #bs, seq_len_all, embedding_size
#         x_p = self.dropout(x_p)

#         #multi-atten
#         x_p = torch.transpose(x_p, 0, 1) # seq_len, bs, embedding_size
#         x = self.pos_encoder_p(x_p)
#         x_p = self.multi_atten_p(x_p, src_key_padding_mask=(mask_p==0)) # seq_len, bs, embedding_size
#         x_p = torch.transpose(x_p, 0, 1) #bs, seq_len, embedding_size
        
#         #pooling
#         x_p = torch.mean(x_p, 1)
#         x_p = self.relu(self.fc_p(x_p))

#         x = x_q + x_p

#         #down
#         x = self.relu(self.down1(x))
#         x = self.relu(self.down2(x))
#         x = self.relu(self.down3(x))
#         x = self.out(x)

#         # x = self.out(x)

#         x = nn.Sigmoid()(x)

#         return x

trainset = DataSeq(queries=train_queries, passages=train_passages, labels=train_labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

valset = DataSeq(queries=test_queries, passages=test_passages, labels=test_labels)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0)

pos_ratio = np.where(train_labels==1)[0].shape[0] / train_labels.shape[0]
print('Postive/Negative ratio in trainset is {0:.4f}'.format(pos_ratio))

#Negative sampling
import numpy as np
def neg_sampling(qids, pids, labels, passages, queries, ratio=10):
    ranked_qid = qids[np.argsort(qids)]
    ranked_labels = labels[np.argsort(qids)]
    ranked_passages = passages[np.argsort(qids)]
    ranked_queries = queries[np.argsort(qids)]

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

    sample_passage = []
    sample_query = []
    sample_label = []
    
    for i, qid in enumerate(tqdm(np.unique(qids))):
        idx1 = np.where(ranked_labels[counter[i]] == 1)
        num_pos = idx1[0].shape[0]

        #positive sample
        sample_passage.extend(ranked_passages[counter[i]][idx1])
        sample_query.extend(ranked_queries[counter[i]][idx1])
        sample_label.extend(ranked_labels[counter[i]][idx1])

        #negative sample
        idx0 = np.delete(np.arange(len(counter[i])), idx1)
        np.random.shuffle(idx0)
        sample_passage.extend(ranked_passages[counter[i]][idx0[:ratio*num_pos]])
        sample_query.extend(ranked_queries[counter[i]][idx0[:ratio*num_pos]])
        sample_label.extend(ranked_labels[counter[i]][idx0[:ratio*num_pos]])
    
    return sample_query, sample_passage, sample_label
        
sample_query, sample_passage, sample_label = neg_sampling(train_qids, train_pids, train_labels, train_passages, train_queries, ratio=5)

pos_ratio = sum(sample_label) / len(sample_label)
print('Postive/Negative ratio in trainset is {0:.4f}'.format(pos_ratio))

trainset = DataSeq(queries=sample_query, passages=sample_passage, labels=sample_label)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

binary_model = BinaryClassifier(args).to(args.device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(binary_model.parameters(), lr=args.lr)


if not os.path.exists('model.pth'):
    for epoch in range(150):
        progress_bar = tqdm(trainloader)
        epoch_loss = 0.
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch: ' + str(epoch))
            # query, passage, query_mask, passage_mask, label = data[0].last_hidden_state.squeeze(), data[1].last_hidden_state.squeeze(), data[2].squeeze(), data[3].squeeze(), data[4].to(device)
            query, passage, query_mask, passage_mask, label= data[0].squeeze().to(device), data[1].squeeze().to(device), data[2].squeeze().to(device), data[3].squeeze().to(device), data[4].to(device)

            with torch.no_grad():
                query = model(input_ids=query, attention_mask=query_mask).last_hidden_state
                passage = model(input_ids=passage, attention_mask=passage_mask).last_hidden_state
            
            output = binary_model(query, passage, query_mask, passage_mask)
            optimizer.zero_grad()
            loss = criterion(output.squeeze(), label.to(torch.float32))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(epoch_loss='%.3f' % (epoch_loss /(i+1)))
        # scheduler.step()

else:
    binary_model = torch.load('model.pth')
    binary_model.eval()

test_queries = test_queries[np.argsort(test_qids)]
test_passages = test_passages[np.argsort(test_qids)]
test_labels = test_labels[np.argsort(test_qids)]

valset = DataSeq(queries=test_queries, passages=test_passages, labels=test_labels)
valloader = torch.utils.data.DataLoader(valset, batch_size=512, shuffle=False, num_workers=2)

yTe_pred = []
with torch.no_grad():
    binary_model.eval()
    progress_bar = tqdm(valloader)
    for i, data in enumerate(progress_bar):
        progress_bar.set_description('Epoch: ' + str(0))
        query, passage, query_mask, passage_mask, label= data[0].squeeze().to(device), data[1].squeeze().to(device), data[2].squeeze().to(device), data[3].squeeze().to(device), data[4].to(device)
        query = model(input_ids=query, attention_mask=query_mask).last_hidden_state
        passage = model(input_ids=passage, attention_mask=passage_mask).last_hidden_state
        output = binary_model(query, passage, query_mask, passage_mask)
        yTe_pred.extend((output.squeeze().cpu().numpy().tolist()))

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
    
def mean_metric_nn(test_qids, test_pids, yTe, pred_yTe, write=True):
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


    m_ap = 0
    m_ndcg = 0
    test_pids = test_pids[np.argsort(test_qids)]
    uni_qids = np.unique(test_qids)

    res_qid = []
    res_pid = []
    res_score = []
    res_rank = []
    res_A1 = []
    res_algoname = []

    for i, count in enumerate(tqdm(counter)):
        sort_idx = np.argsort(pred_yTe[count])[::-1]
        sub_yTe = yTe[count]
        true_label = sub_yTe[sort_idx]
        ap = Average_Precision(true_label)
        ndcg = NDCG(true_label)
        m_ap += ap
        m_ndcg += ndcg

        sub_pids = test_pids[count]

        res_qid.extend([uni_qids[i] for _ in range(len(sort_idx))])
        res_pid.extend(sub_pids[sort_idx])
        res_score.extend(pred_yTe[count][np.argsort(pred_yTe[count])[::-1]])
        res_rank.extend(np.arange(1, len(sort_idx)+1))
        res_A1.extend(['A1' for _ in range(len(sort_idx))])
        res_algoname.extend(['NN' for _ in range(len(sort_idx))])


    if write:
        data = {'qid': res_qid, 'A1': res_A1, 'pid': res_pid, 'rank': res_rank, 'score': res_score, 'algoname': res_algoname}
        data_df = pd.DataFrame(data)
        data_df.to_csv('NN.txt',index=False,header=False, sep=' ')
    
    return m_ap / len(counter), m_ndcg / len(counter)

m_ap, m_ndcg = mean_metric_nn(test_qids, test_pids, test_labels, np.array(yTe_pred))

print('Average precision for Neural Network is {0:.4f}'.format(m_ap))
print('NDCG for Neural Network is {0:.4f}'.format(m_ndcg))


