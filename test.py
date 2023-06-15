import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.classifier import Model
import numpy as np
import pandas as pd
import os
import gensim
from data import TextDataset
from utils.draw import drawLoss
from utils.get_new_vectors import getPositionEncoding, getWord2vec, getTdfs, get_new_wordvec
from utils.adaptive_segment import adaptive_segment, conv_batch
from sklearn.metrics import classification_report


# 读取数据集索引文件
file_list = np.loadtxt(fname='./data/THUCNews_test.csv', delimiter=' ', dtype=str, encoding='utf-8')
#file_list = np.loadtxt(fname='./data/test.csv', delimiter=' ', dtype=str, encoding='utf-8')

# 读取预训练好的词向量
VECTOR_DIR = './sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2'
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=False)
with open('stopwords.txt','r',encoding='utf-8') as file:
    stopwords=file.readlines()


prepath = './output/'
if not os.path.isdir(prepath):
    os.makedirs(prepath)


# 创建模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
#warnings.filterwarnings('ignore')
# model = Model(unit_channel=256, dense_dim=64, num_classes=14, device=device)
model: nn.Module = torch.load(f=prepath+'/model_final.pt', map_location=device)
model = model.to(device)
model.eval()
text_dataset = TextDataset(file_list=file_list, stopwords=stopwords)


def generate_batch(batch):
    ''' 
    batch is a length of 4, each element is a tuple of (text, label) and put it on device(gpu or cpu)
    '''
    # print(batch[0])
    sentence = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    # get position encoding
    pos_encodes = getPositionEncoding(vectors=sentence, d=300)
    # get initial word vector
    word_vectors = getWord2vec(documents=sentence, w2v_model=w2v_model)
    # get tdf
    tdfs = getTdfs(sentence)
    # get the new word vector
    new_vector = get_new_wordvec(word_vectors, pos_encodes, tdfs) # [tensor]
    #print('new vector length: ', len(new_vector))
    #print('after getting new word vector, ', new_vector[0].shape)
    # print('new vector:', new_vector)
    
    # adaptive segments
    adaptive_segments = adaptive_segment(new_vector)
    #print('adaptive segments: ', len(adaptive_segments))
    #print('after adaptive segments, ', adaptive_segments[0].shape)
    
    # pass to Multi-Head-Attention Layer, get a list of 3-dimention tensor, and each 3-dimention tensor is a batch
    mvs = model.get_Mvs(adaptive_segments)
    # pass to Conv Layer to unify the shape of tensor
    one_batch = conv_batch(mvs, device)
    one_batch = torch.cat(one_batch, dim=0) # (batch_size, 256, l, d)
    #one_batch = torch.stack(one_batch, dim=0) # (batch_size, 256, l, d)
    # print(one_batch.shape)
    return one_batch, torch.tensor(labels).to(device)


test_dataloader = DataLoader(dataset=text_dataset, batch_size=16, shuffle=False, collate_fn=generate_batch)


predicts_all = []
labels_all = []
for idx, inp in enumerate(test_dataloader):
    text_data = inp[0]
    label = inp[1]
    current_all = len(label)
    out = model(text_data) # (b, n)
    predicts = torch.argmax(input=out, dim=1) # predictions
    predicts_all += list(predicts)
    labels_all += list(label)
    correct_num = torch.eq(input=predicts, other=label).sum().item()
    print(f'current accuracy: {correct_num/current_all}')


report = classification_report(y_pred=predicts_all, y_true=labels_all, output_dict=True)
save_df = pd.DataFrame(report).transpose()
save_df.to_csv(prepath + '/result.csv', index=True)