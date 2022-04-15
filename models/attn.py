import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, attn,
                 d_keys=None, d_values=None, mix=False,seq_len=96):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.lam = False
        self.qs = False

        if(attn=="lam"):
            self.lam = True
        elif(attn=="qs"):
            self.qs = True

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        if (self.lam):
            self.key_projection = nn.Linear(d_model, d_keys)
            self.value_projection = nn.Linear(d_model, d_values)
            self.embedding = nn.Parameter(torch.randn([seq_len,seq_len,d_keys]), requires_grad=True)

        else:
            self.key_projection = nn.Linear(d_model, d_keys * n_heads)
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
           
            
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        #queries = queries.view(B, L, H, -1)
        keys = self.key_projection(keys) #.view(B, S, H, -1)
        #print("keys outer attention:",keys.shape)
        values = self.value_projection(values) #.view(B, S, H, -1)
        #print("Valuess outer attention:",values.shape)

        if (self.qs == True):
            out,attn = self.inner_attention(queries,keys,values,attn_mask)
        
        elif(self.lam == True):
            queries = queries.view(B,H,L,-1)
            out, attn = self.inner_attention(queries,keys,values,self.embedding)

        else:
            queries = queries.view(B, L, H, -1)
            keys = keys.view(B, S, H, -1)
            values = values.view(B, S, H, -1)
            out, attn = self.inner_attention(queries,keys,values,attn_mask)

        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class QuerySelector(nn.Module):
    def __init__(self, fraction=0.33,mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(QuerySelector, self).__init__()
        self.fraction = fraction

    def forward(self, queries, keys, values,attn_mask = None):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape
        l_Q = int((1.0 - self.fraction) * L_Q)
        K_reduce = torch.mean(keys.topk(l_Q, dim=1).values, dim=1).unsqueeze(1)
        sqk = torch.matmul(K_reduce, queries.transpose(1,2))
        indices = sqk.topk(l_Q, dim=-1).indices.squeeze(1)
        Q_sample = queries[torch.arange(B)[:, None], indices, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_sample, keys.transpose(-2, -1))
        attn = torch.softmax(Q_K / math.sqrt(D), dim=-1)
        mean_values = values.mean(dim=-2)
        result = mean_values.unsqueeze(-2).expand(B, L_Q, mean_values.shape[-1]).clone()
        result[torch.arange(B)[:, None], indices, :] = torch.matmul(attn, values).type_as(result)
        return result, None

    def inference(self):
        pass # no parameters


class LambdaModule(nn.Module):
    def __init__(self, mask_flag=True, factor=5, attention_dropout=0.1, output_attention=False):
        #False,False, factor, attention_dropout=dropout, output_attention=output_attention
        super(LambdaModule, self).__init__()
        #self.scale = scale
        
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        

    def forward(self, queries, keys, values,embedding):
        b, n, h, k = queries.shape
        _, m, v= values.shape

        content_lambda = torch.einsum('b m k, b m v -> b k v',keys.softmax(dim=-1), values)
        position_lambdas = torch.einsum('n m k, b m v -> b n k v',embedding, values)
        content_output = torch.einsum('b h n k, b k v -> b n h v',queries, content_lambda)

        position_output = torch.einsum('b h n k, b n k v -> b n h v',queries, position_lambdas)
        output = (content_output + position_output).view(b,n,-1)

        return output, None

