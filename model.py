# pooling3
import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()  # 把tensor变成在内存中连续分布的形式
                               # torch.view等方法操作需要连续的Tensor；连续的Tensor，语义上相邻的元素，在内存中也是连续的，语义和内存顺序的一致性是缓存友好的，以提升CPU获取操作数据的速度

# 在某个数据上应用一个线性转换，公式表达就是y=xA^T+b
# bias: 默认为True.如果设置成false，则这个线性层不会加上bias。值从均匀分布U(-\sqrt{k},\sqrt{k})中获取
class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class DMSTGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3,
                 out_dim=12, residual_channels=16, dilation_channels=16, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, days=288, dims=40, order=2, in_dim=9, normalization="batch"):
        super(DMSTGCN, self).__init__()
        skip_channels = 8
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.filter_convs_a = nn.ModuleList()
        self.gate_convs_a = nn.ModuleList()
        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.normal_a = nn.ModuleList()
        self.gconv_a = nn.ModuleList()

        self.gconv_a2p = nn.ModuleList()

        self.start_conv_a = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device) #requires_grad自动
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_ak = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.residual = residual_channels
        self.diltational = dilation_channels
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                # self.filter_convs.append(
                #     dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))
                #
                # self.gate_convs.append(
                #     dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))

                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.filter_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0,
                                                  return_indices=False, ceil_mode=False)
                # self.filter_convs_a.append(
                #     dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))
                #
                # self.gate_convs_a.append(
                #     dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_a.append(nn.Conv2d(in_channels=dilation_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))
                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_a.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a2p.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))

        self.relu = nn.ReLU(inplace=True)

        self.end_conv_1 = nn.Conv2d(in_channels=304,
                                    out_channels=304,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=304,
                                    out_channels=12,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        x = adp
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)

        temp = adp[:, :, :32]

        x1 = torch.matmul(x, temp.transpose(1, 2))
        x2 = torch.matmul(temp, x.transpose(1, 2))
        adp = F.relu(F.tanh(x1 - x2))

        return adp


    def forward(self, inputs, ind):
        """
        input: (B, F, N, T)
        """
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0)) #填充区域
        else:
            xo = inputs
        x = self.start_conv(xo[:, [0]])
        x_a = self.start_conv_a(xo[:, [1]])
        skip = 0

        # dynamic graph construction
        new_supports = []
        new_supports_a = []

        for i in range(self.blocks * self.layers):
            adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
            adp_a = self.dgconstruct(self.nodevec_a1[ind], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
            # adp_a2p = self.dgconstruct(self.nodevec_a2p1[ind], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)

            new_supports.append(adp)
            new_supports_a.append(adp_a)
            # new_supports_a2p.append(adp_a2p)

        for i in range(self.layers * self.layers):
            # weight = [0.3,0.7]
            # if i == 0:
            #     fusion_adp = new_supports[i]*weight[0]+new_supports[i]*weight[1]
            #
            # else:
            #     fusion_adp = new_supports[i-1]*weight[0]+new_supports[i]*weight[1]

            # tcn for primary part
            residual = x
            # print(x.shape)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate


            # tcn for auxiliary part
            residual_a = x_a
            # print(x.shape)
            filter_a = self.filter_convs_a[i](residual_a)
            filter_a = torch.tanh(filter_a)
            gate_a = self.gate_convs_a[i](residual_a)
            gate_a = torch.sigmoid(gate_a)
            x_a = filter_a * gate_a



            # skip connection
            s = x
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # 判断两个类型是否相同
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()  #重塑输出的向量 # 输出张量第0维第2维 # 自动计算新的维数 # 保证Tensor是连续的
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            # dynamic graph convolutions
            x = self.gconv[i](x, [new_supports[i]])
            x_a = self.gconv_a[i](x_a, [new_supports_a[i]])
            x_a = x_a[:,:,:,-x.size(3):]

            # multi-faceted fusion module
            p = torch.cat((x, x_a),3)
            x_p = self.pooling(p)
            # x = x_p + x_a +x
            weight =[0.5,0.3,0.2]
            x = weight[0] * x + weight[1] * x_a+weight[2] * x_p

            # residual and normalization ([64, 32, 307, 7])
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
            residual = x
            x = x + residual[:, :, :, -x.size(3):]
            x = self.normal[i](x)
            x_a = self.normal_a[i](x_a)

        # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

# class dilated_inception(nn.Module):
#     def __init__(self, cin, cout, dilation_factor=2):
#         super(dilated_inception, self).__init__()
#         self.tconv = nn.ModuleList()
#         self.h_conv = nn.ModuleList()
#         self.kernel_set = [1,2,3,3]
#         cout = int(cout/len(self.kernel_set))
#         for kern in self.kernel_set:
#             self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
#
#
#     def forward(self, input):
#         e = []
#         h0 = input
#         x = self.tconv[0](input)
#         h0 = h0[:, -x.size(1):, :, -x.size(3):]
#         print(x.shape,h0.shape)
#         Z1 = torch.sigmoid(x + h0)
#         R1 = torch.sigmoid(x + h0)
#         H1_tilda = torch.tanh(x + (R1 * h0))
#         h1 = Z1 * H1_tilda + (1 - Z1) * h0
#         e.append(h1)
#
#         x = self.tconv[1](input)
#         h1 = h1[:, -x.size(1):, :, -x.size(3):]
#         Z2 = torch.sigmoid(x + h1)
#         R2 = torch.sigmoid(x+ h1)
#         H2_tilda = torch.tanh(x + (R2 * h1))
#         h2 = Z2 * H2_tilda + (1 - Z2) * h1
#         e.append(h2)
#
#         x = self.tconv[2](input)
#         h2 = h2[:, -x.size(1):, :, -x.size(3):]
#         Z3 = torch.sigmoid(x + h2)
#         R3 = torch.sigmoid(x + h2)
#         H3_tilda = torch.tanh(x + (R3 * h2))
#         h3 = Z3 * H3_tilda + (1 - Z3) * h2
#         e.append(h3)
#
#         x = self.tconv[3](input)
#         h3 = h3[:, -x.size(1):, :, -x.size(3):]
#         Z4 = torch.sigmoid(x + h3)
#         R4 = torch.sigmoid(x + h3)
#         H4_tilda = torch.tanh(x + (R4 * h3))
#         h4 = Z4 * H4_tilda + (1 - Z4) * h3
#         e.append(h4)
#
#         for i in range(len(self.kernel_set)):
#             e[i] = e[i][..., -e[-1].size(3):]
#         e = torch.cat(e, dim=1)
#         return e


