import numpy as np
import math
import torch
import torch.nn.functional as F


# 自适应分段
def padding_zero(segment, l):
    '''
    Padding one segment to l length with zero vectors.
    return a segment after padding align which is a two dimension numpy array.
    '''
    # print('adaptive_segment:', segment.shape)
    length, d = segment.shape
    if length < l:
        zeros = np.zeros((l-length, d))
        segment = np.concatenate((segment, zeros), axis=0)
    return segment


def adaptive_segment(documents, l=200):
    ''' 
    Adaptive segmentation. Split one document to several segment, which will be padding to l length.
    '''
    adaptive_segments = []
    for document in documents:
        # document is a two dimension numpy array, (length, d)
        segment_num = math.ceil(len(document)/l)
        splits = np.array_split(document, segment_num, axis=0) # a list of segments, each element is a two dimension numpy array, (length, d), which need to be padding to l
        adaptive_segment_res = np.array([padding_zero(segment, l) for segment in splits])
        adaptive_segment_res = torch.from_numpy(adaptive_segment_res).to(torch.float32)
        adaptive_segments.append(adaptive_segment_res)
    return adaptive_segments


def conv_batch(batch_list, unit_channel=256):
    ''' 
    Add a extra conv to the data and return a batch.
    Params:
        batch_list:  a list of (c, l, d), each 3-dimension tensor is a batch data and its channel varies.
    '''
    unified_batch = []
    for x in batch_list:
        inp_channel = x.shape[0]
        # weights = torch.rand(1, inp_channel, 3, 3)
        weights = torch.ones(unit_channel, inp_channel, 3, 3)
        b = torch.ones(unit_channel)
        out = F.conv2d(input=x, weight=weights, bias=b, stride=1, padding=1) # out: (1, unit_channel, l, d)
        unified_batch.append(out)
    return unified_batch