import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.classifier import Model
import numpy as np
import os
import gensim
from data import TextDataset
from utils.draw import drawLoss
from utils.get_new_vectors import getPositionEncoding, getWord2vec, getTdfs, get_new_wordvec
from utils.adaptive_segment import adaptive_segment, conv_batch


# 读取数据集索引文件
file_list = np.loadtxt(fname='./data/THUCNews_train.csv', delimiter=' ', dtype=str, encoding='utf-8')

# 读取预训练好的词向量
VECTOR_DIR = './sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2'
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=False)
with open('stopwords.txt','r',encoding='utf-8') as file:
    stopwords=file.readlines()

# 创建模型
model = Model(unit_channel=256, dense_dim=64, num_classes=14)
text_dataset = TextDataset(file_list=file_list, stopwords=stopwords)


def generate_batch(batch):
    ''' 
    batch is a length of 4, each element is a tuple of (text, label)
    '''
    # print(batch[0])
    sentence = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    print(len(sentence))
    print(labels)
    # get position encoding
    pos_encodes = getPositionEncoding(vectors=sentence, d=300)
    # get initial word vector
    word_vectors = getWord2vec(documents=sentence, w2v_model=w2v_model)
    # get tdf
    tdfs = getTdfs(sentence)
    # get the new word vector
    new_vector = get_new_wordvec(word_vectors, pos_encodes, tdfs) # [tensor]
    print('new vector length: ', len(new_vector))
    print('after getting new word vector, ', new_vector[0].shape)
    # print('new vector:', new_vector)
    
    # adaptive segments
    adaptive_segments = adaptive_segment(new_vector)
    print('adaptive segments: ', len(adaptive_segments))
    print('after adaptive segments, ', adaptive_segments[0].shape)
    
    # pass to Multi-Head-Attention Layer, get a list of 3-dimention tensor, and each 3-dimention tensor is a batch
    mvs = model.get_Mvs(adaptive_segments)
    # pass to Conv Layer to unify the shape of tensor
    one_batch = conv_batch(mvs)
    # one_batch = torch.tensor(one_batch)
    one_batch = torch.stack(one_batch, dim=0) # (batch_size, 256, l, d)
    # print(one_batch.shape)
    return one_batch, torch.tensor(labels)

train_dataloader = DataLoader(dataset=text_dataset, batch_size=4, shuffle=False, collate_fn=generate_batch)


epochs = 100
loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []


for epoch in range(epochs):
    for idx, inp in enumerate(train_dataloader):
        text_data = inp[0]
        label = inp[1]
        out = model(text_data)
        # print(f'epoch {epoch}, data shape {text_data.shape}, label {label}, out {out.shape}')
        model.zero_grad()

        loss = loss_f(out, label)
        print(f'epoch {epoch}, loss {loss.item()}')
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()


prepath = './output/'
if not os.path.isdir(prepath):
    os.makedirs(prepath)

with open(prepath+'/model_final.pt', 'wb') as f:
    torch.save(model, f)


drawLoss(loss=loss_list, output=prepath)