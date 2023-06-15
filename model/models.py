# 网络模型
import torch
import torch.nn as nn


# 多头注意力
class MultiHeadAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(embed_dim=300, num_heads=4, batch_first=True)

    def forward(self, segs, next_segs):
        ''' 
        seg, next_seg: (b, l, d)
        '''
        mv_1 = self.multi_attn(next_segs, segs, segs)
        mv_2 = self.multi_attn(segs, next_segs, next_segs)
        return mv_1, mv_2


# 多通道卷积
class MultiConv(nn.Module):
    def __init__(self, unit_channel=256) -> None:
        super().__init__()
        self.unit_channel = unit_channel
        self.first_seq = nn.Sequential(
            nn.Conv2d(in_channels=unit_channel, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Dropout()
        )
        self.second_seq = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        ''' 
        x: (b, unit_channel, l, d)
        '''
        return self.second_seq(self.first_seq(x)) # (b, out_channel, l, d)

# 胶囊网络
def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        print('===== PrimaryCaps =====')
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: num_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        print("after init conv", out.shape)
        # Flatten out: (batch_size, num_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)
    

class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing, device='cpu'):
        """
        Initialize the layer.

        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            num_caps: 		Number of capsules in the capsule layer
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)

    def forward(self, x):
        print('===== DigitCaps =====')
        print('init input: ', x.shape)
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        print('after unsqueeze: x', x.shape)
        print('after unsqueeze: W', self.W.shape)
        #
        # W @ x =
        # (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        u_hat = torch.matmul(self.W, x)
        print('W@x: ', u_hat.shape)
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)
        # print('out: ', v.shape)

        return v


class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv2d layer
        self.conv = nn.Conv2d(64, 64, 4)
        self.relu = nn.ReLU(inplace=True)

        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=4,
                                        in_channels=64,
                                        out_channels=4,
                                        kernel_size=9,
                                        stride=2)

        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=4,
                                    in_caps=55100,
                                    num_caps=4, # 胶囊数量为64
                                    dim_caps=16,
                                    num_routing=3)

        # Reconstruction layer

    def forward(self, x):
        out = self.relu(self.conv(x))
        print('after init conv: ', out.shape)
        out = self.primary_caps(out)
        out = self.digit_caps(out)
        print('after digit capsure: ', out.shape)

        # Shape of logits: (batch_size, num_capsules)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(10).index_select(dim=0, index=torch.argmax(logits, dim=1))

        # Reconstruction
        batch_size = out.shape[0]
        # reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1))
        return out

# 全连接层
class FCNet(nn.Module):
    def __init__(self, dense_dim, num_classes) -> None:
        super().__init__()
        self.dense_dim = dense_dim
        self.num_classes = num_classes
        self.dense = nn.Linear(dense_dim, num_classes)

    def forward(self, x):
        return self.dense(x)



