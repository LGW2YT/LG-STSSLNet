import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import copy
import math
import torch.nn.init as init
import pandas as pd
from adj_mx import get_adj
import pickle as pkl
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
torch.autograd.set_detect_anomaly(True)

class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, D, num_vertex, num_step]
    HT:     [batch_size, D, num_vertex, num_step]
    D:      output dims
    return: [batch_size, D, num_vertex, num_step]
    '''

    def __init__(self, D, bn_decay=0.2):
        super(gatedFusion, self).__init__()

        self.FC_xs = nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1, bias=False)
        self.FC_xt = nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1, bias=True)
        self.FC_h = nn.Sequential(nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1, bias=True),
                                  nn.ReLU(inplace=False),
                                  nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1, bias=True))

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H

class Embedding(nn.Module):
    '''
    spatio-temporal embedding
    TE:     [batch_size, T, D]
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, time_futures, D):
        super(Embedding, self).__init__()
        self.FC_te = nn.Sequential(
            nn.Conv2d(time_futures, D, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(D, D, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True))

    def forward(self, TE):
        '''
        Args:
            TE: [B, T, D*]
        Returns: [B, T, D]

        '''
        # temporal embedding
        TE = torch.unsqueeze(TE, dim=3)
        TE = self.FC_te(TE.transpose(2, 1))#B,T,D,1-->B,D,T,1
        return TE.transpose(2, 1)

def clones(module, N):
    """
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """
    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)




class MultiHeadAttention_Temporal(nn.Module):  # 1d conv on query, 1d conv on key
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):

        super(MultiHeadAttention_Temporal, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = (kernel_size - 1) // 2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):

        nbatches = query.size(0)
        N = query.size(1)
        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
        key = self.conv1Ds_aware_temporal_context[1](key.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class attention2d(nn.Module):
    def __init__(self, in_planes, K,):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, K, 1,)
        self.fc2 = nn.Conv2d(K, K, 1,)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        out = self.relu(x)
        out = self.fc2(out).view(out.size(0), -1)
        #if torch.any(torch.isnan(out)):
        #    print('nan,attention2d111')
        f_out = F.softmax(out, -1)
        #if torch.any(torch.isnan(f_out)):
        #   print('nan,attention2d222')
        return f_out

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, bias=True, ):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, K, )

        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        self.bn= nn.BatchNorm2d(num_features=out_planes)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        #if torch.any(torch.isnan(output)):
        #    print('nan,Dynamic_conv2d')
        #    output[torch.isnan(output)] = 0
        output = self.bn(output)
        output1 = self.relu(output)

        return output1


class LS_TCM(nn.Module):
    def __init__(self, grid_in_channel, num_time_features, num_conv_layer, num_of_conv_filters, num_head, gru_hidden_size, emb_dim_final,
                 num_of_gru_layers, seq_len, K):
        super(LS_TCM, self).__init__()
        # Dynamic convolution layers
        self.grid_conv_star = Dynamic_conv2d(grid_in_channel, num_of_conv_filters//2, kernel_size=3, K=K, padding=1)

        if num_conv_layer > 2:
            self.grid_conv = clones(Dynamic_conv2d(num_of_conv_filters//2, num_of_conv_filters//2, kernel_size=3, K=K, padding=1), num_conv_layer - 2)

        self.grid_conv_fina = Dynamic_conv2d(num_of_conv_filters//2, num_of_conv_filters, kernel_size=3, K=K, padding=1)
        self.res_1 = nn.Sequential(
            nn.Conv2d(grid_in_channel, num_of_conv_filters, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_of_conv_filters)
        )

        # GRU and attention layers
        self.grid_gru = nn.GRU(num_of_conv_filters, gru_hidden_size, num_of_gru_layers, batch_first=True)
        self.bn_gru = nn.BatchNorm1d(gru_hidden_size)
        self.activation = nn.ReLU()

        self.time_attention = MultiHeadAttention_Temporal(num_head, gru_hidden_size)
        self.num_conv_layer = num_conv_layer

        # Fully connected layers
        self.fc = nn.Conv2d(seq_len, 1, kernel_size=1)
        self.res_2 = nn.Sequential(nn.Linear(num_of_conv_filters, gru_hidden_size),
                                   nn.Conv2d(seq_len, 1, kernel_size=1)
                                   )
        self.bn_res = nn.BatchNorm1d(gru_hidden_size)
        self.res_conv2 = nn.Conv2d(seq_len, 1, kernel_size=1)
        self.res_liner = nn.Linear(num_of_conv_filters, gru_hidden_size)
        self.time_embed = Embedding(num_time_features, gru_hidden_size)
        self.final_linear = nn.Sequential(nn.Linear(gru_hidden_size, emb_dim_final))

        self.reset_params()

    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x, target_time):
        # x: B, T, D, W, H; target_time: b, seq_len(t), d
        B, T, D, W, H = x.size()
        x = x.view(-1, D, W, H)
        #if torch.any(torch.isnan(x)):
        #   print('nan,ST_Geo')
        res = x

        x = self.grid_conv_star(x)
        #if torch.any(torch.isnan(x)):
        #    print('nan,ST_Geo')

        if self.num_conv_layer > 2:
            for i in range(self.num_conv_layer - 2):
                x = self.grid_conv[i](x)
                #if torch.any(torch.isnan(x)):
                #    print('nan,ST_Geo')

        x = self.grid_conv_fina(x)
        #if torch.any(torch.isnan(x)):
        #   print('nan,ST_Geo')
        x = x + self.res_1(res)
        res = x  # B*T, D, W, H

        #if torch.any(torch.isnan(x)):
        #    print('nan')

        x = x.contiguous().view(B, T, -1, W, H).permute(0, 3, 4, 1, 2)
        x = x.reshape(-1, T, x.size(4))

        x, _ = self.grid_gru(x)
        x = x.reshape(B, W * H, T, -1)
        x = self.bn_gru(x.contiguous().view(B*W*H*T, -1)).view(B, W * H,  T, -1)
        x = self.activation(x)
        #if torch.any(torch.isnan(x)):
        #    print('nan,ST_Geo')
        x = self.time_attention(x, x, x)  # B, N, T, D

        out = self.fc(x.transpose(1, 2))  # B, 1, N, D
        #if torch.any(torch.isnan(out)):
        #    print('nan')
        res = res.reshape(B, T, -1, W * H).transpose(2, 3)
        res = self.res_liner(self.res_conv2(res))
        out = out + self.bn_res(res.reshape(-1, res.size(3))).reshape(B, 1, W * H, -1)
        #if torch.any(torch.isnan(out)):
        #    print('nan')

        out = out.squeeze(dim=1)  # B, N, D
        time_embed = self.time_embed(target_time)
        if time_embed.shape[0] == 1:
            time_embed = time_embed.squeeze(1).squeeze(2) # B, D
        else:
            time_embed = torch.squeeze(time_embed)
        time_embed = time_embed.unsqueeze(1)
        out = self.final_linear(out + time_embed)#B,N,T
        #if torch.any(torch.isnan(out)):
        #    print('nan,ST_Geo')

        return out.transpose(1,2).contiguous().view(B, -1, W, H)  # B,D,W,H

class FlexGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, K):
        super(FlexGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.attention_node_embding = attention2d(dim_in, K, )
    def forward(self, x, adj, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [K,N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        softmax_attention = self.attention_node_embding(x.transpose(1,2).unsqueeze(3))#b,k
        node_embeddings = torch.einsum("bk,knd->bnd", softmax_attention, node_embeddings)#根据输入，不同批次有不同的节点嵌入（动态或时变节点嵌入），b,n,d
        node_num = node_embeddings.shape[1]
        supports = adj
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('bnd,dkio->bnkio', node_embeddings, self.weights_pool)  #B,N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #B,N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv


class GS_TCM(nn.Module):
    def __init__(self, num_of_vertices, num_time_features, num_gcn_layer, num_of_graph_feature, num_of_graph_filters,
                 gru_hidden_size, num_of_gru_layers, num_head, emb_dim_final,
                 seq_len, cheb_k, K, embed_dim,
                 north_south_map, west_east_map):
        super(GS_TCM, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.seq_len = seq_len
        self.num_gcn_layer = num_gcn_layer
        self.node_embeding = nn.Parameter(torch.randn(K, num_of_vertices, embed_dim), requires_grad=True)#创新点

        self.gcn_star = FlexGCN(num_of_graph_feature, num_of_graph_filters, cheb_k, embed_dim, K)
        self.bn_gcn_star = nn.BatchNorm1d(num_of_graph_filters)

        if num_gcn_layer > 1:
            self.gcn_list = nn.ModuleList(
                [FlexGCN(num_of_graph_filters, num_of_graph_filters, cheb_k, embed_dim, K) for _ in
                 range(num_gcn_layer - 1)])
            self.bn_gcn_list = nn.ModuleList([nn.BatchNorm1d(num_of_graph_filters) for _ in range(num_gcn_layer - 1)])

        self.graph_gru = nn.GRU(num_of_graph_filters, gru_hidden_size, num_of_gru_layers, batch_first=True)
        self.bn_gru = nn.BatchNorm1d(gru_hidden_size)
        self.activate = nn.ReLU()

        self.time_attention = MultiHeadAttention_Temporal(num_head, gru_hidden_size)
        self.res_linea1 = nn.Linear(num_of_graph_feature, num_of_graph_filters)

        self.fc = nn.Conv2d(seq_len, 1, kernel_size=1)
        self.res_conv = nn.Conv2d(seq_len, 1, kernel_size=1)
        self.res_liner2 = nn.Linear(num_of_graph_filters, gru_hidden_size)
        self.bn_res = nn.BatchNorm1d(gru_hidden_size)
        self.time_embed = Embedding(num_time_features, gru_hidden_size)
        self.final_linear = nn.Sequential(nn.Linear(gru_hidden_size, emb_dim_final))
        self.relu = nn.ReLU(inplace=False)
        self.reset_params()

    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self,adj, x, grid_node_map,target_time):
        node_embeddings = self.node_embeding
        B, T, D, N = x.size()
        x = x.reshape(B * T, D, N).transpose(1, 2)  # B*T, N, D
        res = x  # B*T, N, D

        out = self.gcn_star(x, adj, node_embeddings)
        out = self.bn_gcn_star(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)

        if self.num_gcn_layer > 1:
            for i in range(self.num_gcn_layer - 1):
                out = self.gcn_list[i](out, adj, node_embeddings)
                out = self.bn_gcn_list[i](out.transpose(1, 2)).transpose(1, 2)
                out = self.relu(out)

        res = self.res_linea1(res)  # B*T, N, D
        out2 = res + out
        res = out2
        #if torch.any(torch.isnan(out2)):
        #    print('nan')

        out2 = out2.reshape(B, T, N, -1).transpose(1, 2).reshape(B * N, T, -1)
        out2, _ = self.graph_gru(out2)
        out2 = self.bn_gru(out2.contiguous().view(B * N * T, -1)).view(B, N, T, -1)
        out2 = self.activate(out2)
        #if torch.any(torch.isnan(out2)):
        #    print('nan')

        out2 = self.time_attention(out2, out2, out2)  # B, N, T, gru_hidden_size
        out2 = self.fc(out2.transpose(1, 2))  # B, 1, N, gru_hidden_size
        #if torch.any(torch.isnan(out2)):
        #  print('nan')

        res = self.res_liner2(res)
        res = self.res_conv(res.reshape(B, T, N, -1))  # B, 1, N, D
        out2 = out2 + self.bn_res(res.reshape(-1,res.size(3))).reshape(B, 1, N, -1)
        #if torch.any(torch.isnan(out2)):
        #   print('nan')

        out2 = out2.squeeze(dim=1)  # B, N, D
        time_embed = self.time_embed(target_time)
        if time_embed.shape[0] == 1:
            time_embed = time_embed.squeeze(1).squeeze(2)  # B, D
        else:
            time_embed = torch.squeeze(time_embed) # B, D
        time_embed = time_embed.unsqueeze(1)
        out2 = self.final_linear(time_embed + out2)
        #if torch.any(torch.isnan(out2)):
        #   print('nan,ST_HiddenCorr')
        grid_node_map_tmp = torch.from_numpy(grid_node_map) \
            .to(x.device) \
            .repeat(B, 1, 1)
        graph_output = torch.bmm(grid_node_map_tmp, out2) \
            .permute(0, 2, 1) \
            .view(B, -1, self.north_south_map, self.west_east_map)
        return graph_output  # B, D, W, H

class GSEM(nn.Module):
    def __init__(self, num_nodes, grap_embe_dim):
        super(GSEM, self).__init__()
        self.node_embeding = nn.Parameter(torch.randn(num_nodes, grap_embe_dim, dtype=torch.float32), requires_grad=True)
        self.beta = torch.nn.Parameter(
            torch.rand(num_nodes, dtype=torch.float32),
            requires_grad=True
        )
        self.attn = torch.nn.Conv2d(2, 1, kernel_size=1)
        self.epsilon = 1 / num_nodes * 0.5
        torch.nn.init.kaiming_uniform_(self.node_embeding, a=math.sqrt(5))

    def forward(self, adj_mx):
        new_adj_mx = torch.matmul(self.node_embeding, self.node_embeding.t())#n,n
        new_adj_mx = torch.relu(new_adj_mx+torch.diag(self.beta))
        attn = torch.sigmoid(self.attn(torch.stack((new_adj_mx, adj_mx), dim=0).unsqueeze(
            dim=0)).squeeze())  # torch.stack((new_adj_mx, adj_mx), dim=0).unsqueeze(dim=0)后变为(1, 2, 307, 307),卷积后变为（1，1，307，307）--（307，307）
        new_adj_mx = attn * new_adj_mx + (1. - attn) * adj_mx
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = torch.relu(d.view(-1, 1) * new_adj_mx * d - self.epsilon)
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = d.view(-1, 1) * new_adj_mx * d

        return new_adj_mx.float()



class LG_STSSLNet(nn.Module):
    def __init__(self,grid_in_channel, num_of_vertices, num_time_features, num_conv_layer, num_of_conv_filters, num_gcn_layer, num_head, gru_hidden_size, num_of_gru_layers,
                 num_of_graph_feature, num_of_graph_filters,
                 seq_len, cheb_k, K, Conv_K, emb_size, emb_dim_final,
                 north_south_map, west_east_map, batch_size, pre_len):
        super(LG_STSSLNet, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.batch_size = batch_size
        self.emb_dim_final = emb_dim_final
        self.ST_Geo = LS_TCM(grid_in_channel, num_time_features, num_conv_layer, num_of_conv_filters, num_head, gru_hidden_size, emb_dim_final, num_of_gru_layers, seq_len, Conv_K)
        self.ST_Hid_Corr = GS_TCM(num_of_vertices, num_time_features, num_gcn_layer, num_of_graph_feature, num_of_graph_filters, gru_hidden_size, num_of_gru_layers, num_head, emb_dim_final,seq_len, cheb_k, K, emb_size,north_south_map, west_east_map)
        self.gatedfusion = gatedFusion(emb_dim_final)
        #self.fc = nn.Sequential(nn.Conv2d(emb_dim_final,1,kernel_size=1),
        #                        nn.ReLU(),
        #                        nn.Conv2d(1,1,kernel_size=1)
        #                       )
        self.fc = nn.Linear(emb_dim_final * north_south_map * west_east_map,
                                      pre_len * north_south_map * west_east_map)
        self.reset_params()

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'fc' in name or 'gatedfusion' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

    def forward(self, grid_feature, target_time, gragh_feature, adj, grid_node_map):

        st_geo_out = self.ST_Geo(grid_feature, target_time)
        st_hidd_corr_out = self.ST_Hid_Corr(adj, gragh_feature, grid_node_map,target_time)

        fusion_out = self.gatedfusion(st_geo_out, st_hidd_corr_out)#B,D,W,H
        #final_out = self.fc(fusion_out)
        final_output = fusion_out.view(-1, self.emb_dim_final*self.north_south_map*self.west_east_map)
        #print(final_output.shape)
        final_output = self.fc(final_output) \
            .view(-1, self.pre_len, self.north_south_map, self.west_east_map)

        return final_output, fusion_out





