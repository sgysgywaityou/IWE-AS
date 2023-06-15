from .models import *


class Model(nn.Module):
    ''' 
    Network Layer after getting batches.
    Args:
        unit_channel:   unified MVs of adaptive segments from various document, also defined as input channel of Conv Layer
        out_channel: output channel of the Conv Layer
        dense_dim:  features number before Full Connected Layer
    '''
    def __init__(self, unit_channel, dense_dim, num_classes) -> None:
        super().__init__()
        self.multiAtten = MultiHeadAttn()
        self.multiConv = MultiConv(unit_channel)
        self.capsureNet = CapsNet()
        self.fc = FCNet(dense_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def get_Mvs(self, adaptive_segments):
        '''
        Pass to MultiHeadAttention Layer.
        params:
            adaptive_segments:  a list of segments result, (n, l, d) [...], several batch and each element is a batch
        '''
        mvs = []
        for adaptive_segment in adaptive_segments:
            adaptive_segment = adaptive_segment.to(self.device)
            # adaptive_segment: segmengts result, (n, l, d)
            # print('adaptive segment: ', adaptive_segment.shape)
            c = adaptive_segment.shape[0]
            if c == 1:
                # if current adaptive_segment only has one
                mv = torch.cat([adaptive_segment, adaptive_segment], dim=0)
            else:
                # pass to Multi-Head-Attention Layer, get 2(c-1) channels of tensor
                segs = adaptive_segment[:-1]
                next_segs = adaptive_segment[1:]
                mv_1, mv_2 = self.multiAtten(segs, next_segs)
                mv = torch.cat([mv_1[0], mv_2[0]], dim=0)
            mvs.append(mv)
        return mvs

    def forward(self, x):
        ''' 
        x: (b, unit_channel, l, d)
        '''
        b = x.shape[0]
        out = self.multiConv(x) # (b, out_channel, l, d)
        print('after Multi Conv: ', out.shape)
        out = self.capsureNet(out) # (b, num_classes, out_capsure)
        print('after Capsure: ', out.shape)
        out = out.reshape(b, -1) # (b, dense_dim)
        return self.fc(out) # (b, num_classes)