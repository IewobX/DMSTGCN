# cutADG
import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


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
        self.device=device
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
        self.relu1 = nn.ReLU()


        self.start_conv_a = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
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

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
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
                self.fc1 = nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, 1),
                                     bias=False)
                self.fc2 = nn.Conv2d(in_channels=residual_channels,out_channels=dilation_channels, kernel_size=(1, 1),
                                     bias=False)

                # 1x1 convolution for residual connection
                self.residual_convs_a.append(nn.Conv1d(in_channels=dilation_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))
                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(
                        nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_a.append(
                        nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
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

        self.end_conv_1 = nn.Conv2d(in_channels=832,
                                    out_channels=64,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=64,
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
        adp = F.relu(torch.tanh(x1 - x2))
        # adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, inputs, ind):
        """
        input: (B, F, N, T)
        """
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        x = self.start_conv(xo[:, [0]])
        x_a = self.start_conv_a(xo[:, [1]])
        skip = 0

        # dynamic graph construction
        new_supports = []
        # new_supports_a = []

        for i in range(self.blocks * self.layers):
            adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
            adp_a = self.dgconstruct(self.nodevec_a1[ind], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
            # adp_a2p = self.dgconstruct(self.nodevec_a2p1[ind], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)

            new_supports.append(adp)
            # new_supports_a.append(adp_a)
            # new_supports_a2p.append(adp_a2p)

        for i in range(self.blocks * self.layers):
            # tcn for primary part
            residual = x
            # #GRU1
            Z1 = torch.sigmoid(x + residual)
            R1 = torch.sigmoid(x + residual)
            H1_tilda = torch.tanh(x + (R1 * x))
            h1 = Z1 * x + (1 - Z1) * H1_tilda
            #GRU2
            Z2 = torch.sigmoid(x + h1)
            R2 = torch.sigmoid(x + h1)
            H2_tilda = torch.tanh(x + (R2 * h1))
            h2 = Z2 * x + (1 - Z2) * H2_tilda
            #GRU3
            Z3 = torch.sigmoid(x + h2)
            R3 = torch.sigmoid(x + h2)
            H3_tilda = torch.tanh(x + (R3 * h2))
            h3 = Z3 * x + (1 - Z3) * H3_tilda

            #channel attention
            avg_pool = nn.AdaptiveAvgPool2d((x.size(-2),x.size(-1))).to(self.device)
            max_pool = nn.AdaptiveMaxPool2d((x.size(-2),x.size(-1))).to(self.device)
            avg_out = self.fc2(self.relu1(self.fc1(avg_pool(residual))))
            max_out = self.fc2(self.relu1(self.fc1(max_pool(residual))))
            x = avg_out + max_out + x
            x = torch.sigmoid(x)

            # filter = self.filter_convs[i](residual)
            # filter = torch.tanh(filter)
            # gate = self.gate_convs[i](residual)
            # gate = torch.sigmoid(gate)
            # x = filter * gate

            # tcn for auxiliary part
            residual_a = x_a
            # filter_a = self.filter_convs_a[i](residual_a)
            # filter_a = torch.tanh(filter_a)
            # gate_a = self.gate_convs_a[i](residual_a)
            # gate_a = torch.sigmoid(gate_a)
            # x_a = filter_a * gate_a

            # skip connection
            s = x
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            # dynamic graph convolutions
            x = self.gconv[i](x, [new_supports[i]])
            # x_a = self.gconv_a[i](x_a, [new_supports_a[i]])
            x = x[:,:,:,-x_a.size(3):]
            # multi-faceted fusion module
            # x_a2p = self.gconv_a2p[i](x_a, [new_supports_a2p[i]])
            x = x_a + x

            # residual and normalization
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
            x = x + residual[:, :, :, -x.size(3):]
            x = self.normal[i](x)
            x_a = self.normal_a[i](x_a)

        # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
