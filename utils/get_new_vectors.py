import jieba
import math
import numpy as np
from torchtext.vocab import Vocab, vocab
from collections import Counter, OrderedDict


def preprocess_text(content_lines, sentences, stopwords):
    for line in content_lines:
        try:
            segs=jieba.lcut(line) # 中文分词
            # print(f'segs: {len(segs)}')
            segs=filter(lambda x:len(x)>1, segs) # 剔除只有单个字的语气助词
            segs=filter(lambda x:x not in stopwords,segs) # 剔除停止词
            sentences.append(list(segs)) # ['中国', '是', '一个', '伟大', '的', '国家']
            # cate.append(str(category))
        except Exception as e:
            print(e) 
            continue


def extractSubwords(word):
    return [word[i:i + x + 1] for x in range(len(word)) for i in range(len(word) - x)]


def replace_vector(word, w2v_model, d):
    ''' 
    如果word不存在于词表中，就找他的子词，选取第一个存在于词表的子词的嵌入向量作为返回结果；如果都没有，返回一个随机向量
    '''
    vector = np.random.rand(d)
    for subword in extractSubwords(word):
        if subword in w2v_model:
            vector = w2v_model[subword]
            break
    return vector


def getWord2vec(documents, w2v_model):
    ''' 
    Transfer each word in all documents to vector.
    documents: a two dimension list, each element is a whole document which contains lots of words, 
                e.g. [['西江月', '夜行', '黄沙道', '中'], ['黄鹤楼', '送', '孟浩然', '之', '广陵'], ...]
    w2v_model: pretrained Word2Vec model
    return a list of two dimension numpy array, each numpy array refer to a document
    '''
    documents_vectors = []
    for document in documents:
        vectors = np.array([w2v_model[word] if word in w2v_model else replace_vector(word, w2v_model, 300) for word in document]) # get the vectors of the whole document
        documents_vectors.append(np.array(vectors))
    return documents_vectors


def getPositionEncoding(vectors, d):
    ''' 
    Calculate the position encodes of vectors which refer to all documents.
    vectors: a list of two dimension numpy array, (length, d)
    d: word embedding dimension
    return a list of two dimension numpy arrays, each numpy array refer to a position encoding result of a document.
    '''
    position_encodes = []
    for w_vector in vectors:
        max_len = len(w_vector)
        pos_encode = np.zeros((max_len, d)) # (max_len, d)
        positions = np.arange(0, max_len).reshape(-1, 1) # (max_len, 1)
        div_term = np.exp(np.arange(0, d, 2) * -(math.log(10000.0) / d)) # (d//2,)
        pos_encode[:, 0::2] = np.sin(positions * div_term) # (max_len, d//2)
        pos_encode[:, 1::2] = np.cos(positions * div_term)
        # pos_encode = pos_encode.reshape(-1)
        # print('pos: ', pos_encode.shape)
        position_encodes.append(pos_encode)
    return position_encodes


def getTdfs(sentences):
    '''返回句子列表所包含的词表
    input: sentences, a two dimension of list of strings, each element is a document which contains a set of words
    return a two dimension list, each element refer to the tdf value of all words in one document.
    '''
    # print(f'sentences length: {len(sentences)}')
    # 统计当前所有文档的词表
    corpora = Counter([w for item in sentences for w in item])
    sorted_by_freq_tuples = sorted(corpora.items(), key=lambda x:x[1], reverse=True)

    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocabulary = vocab(ordered_dict) # 转词表
    word2idx = vocabulary.get_stoi() # 按频率排序的字典{词: 编号}
    words = vocabulary.get_itos() # 按频率排序的词
    
    T = len(sentences)
    N = len(words)

    # 每一份文档做成一个词典 [{词: 出现频数}, {词: 出现频数}, {词: 出现频数}, {词: 出现频数}]
    dicts = []
    for sentence in sentences:
        dic = Counter(sentence)
        # print(dic)
        dicts.append(dic)

    tdf = np.zeros((N, T)) # 保存tdf值，行数为词汇总数，列数为文档数
    for i in range(N):
        for j in range(T):
            # 计算词汇i在文档j中的tdf值
            word = words[i]
            document = sentences[j]
            current_impor = dicts[j][word] / len(document) # 分子
            all_impor = 1
            for k in range(T):
                if k == j:
                    continue
                all_impor += (dicts[k][word] / len(sentences[k])) # 分母
            res = current_impor / all_impor
            tdf[i][j] = res
    # 区间归一化
    min_v = np.min(tdf)
    dis = np.max(tdf) - min_v
    tdf = (tdf - min_v) / dis
    # tdf = np.e ** tdf
    # total = np.sum(tdf)
    # tdf = tdf / total
    
    # print('词频排序 words: ', words)
    # print('原始tdf值: ', tdf)
    
    tdfs = []
    for idx, document in enumerate(sentences):
        # print(idx, document, dicts[idx])
        tdfs.append(np.array([tdf[word2idx[word], idx] for word in document]))
    # print(tdfs.shape)
    return tdfs
    #fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小


def get_new_wordvec(vectors, positions, tdfs, thresh=0.):
    ''' 
    Get the new word vector, which is the result of initial word vector, position encoding and tdfs.
    '''
    new_wordvecs = []
    for vec, pos, init_tdf in zip(vectors, positions, tdfs):
        # print('init_tdf: ', init_tdf)
        tdf = init_tdf.reshape(-1, 1)
        new_wordvec = vec * tdf + pos
        new_wordvec[np.where(init_tdf<thresh)] = 0 # threshhold < 0.2, 0
        new_wordvecs.append(new_wordvec)
    return new_wordvecs