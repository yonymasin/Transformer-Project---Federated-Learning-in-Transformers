from collections import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class ShakesHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, hidden_dim, dim, client_sample, heads=8, dim_head=64, n_hidden=1, depth=6,
                 spec_norm=False):
        super(ShakesHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.wqs_value_list=nn.ModuleList([])
        self.wks_value_list=nn.ModuleList([])
        self.wvs_value_list=nn.ModuleList([])

        for d in range(self.depth):
            wq_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wk_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wv_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            self.wqs_value_list.append(wq_value)
            self.wks_value_list.append(wk_value)
            self.wvs_value_list.append(wv_value)


    def finetune(self, emd):
        features = self.mlp(emd)
        weights=OrderedDict()
        for d in range(self.depth):
            layer_d_q_value_hyper = self.wqs_value_list[d]
            layer_d_q_value = layer_d_q_value_hyper(features).view(self.inner_dim ,self.dim)
            layer_d_k_value_hyper = self.wks_value_list[d]
            layer_d_k_value = layer_d_k_value_hyper(features).view(self.inner_dim ,self.dim)
            layer_d_v_value_hyper = self.wvs_value_list[d]
            layer_d_v_value = layer_d_v_value_hyper(features).view(self.inner_dim ,self.dim)
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value
        return weights


    def forward(self, idx, test):
        weights = 0
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            for d in range(self.depth):
                layer_d_q_value_hyper = self.wqs_value_list[d]
                layer_d_q_value = layer_d_q_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                layer_d_k_value_hyper = self.wks_value_list[d]
                layer_d_k_value = layer_d_k_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                layer_d_v_value_hyper = self.wvs_value_list[d]
                layer_d_v_value = layer_d_v_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                for nn in range(self.client_sample):
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value[nn]
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value[nn]
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value[nn]
        else:
            weights=OrderedDict()
            for d in range(self.depth):
                layer_d_q_value_hyper = self.wqs_value_list[d]
                layer_d_q_value = layer_d_q_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_k_value_hyper = self.wks_value_list[d]
                layer_d_k_value = layer_d_k_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_v_value_hyper = self.wvs_value_list[d]
                layer_d_v_value = layer_d_v_value_hyper(features).view(self.inner_dim ,self.dim)
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value
        return weights


class LoraHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, hidden_dim, dim, client_sample, heads=8, dim_head=64, n_hidden=1, depth=6,
                 spec_norm=False):
        super(LoraHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        # Value list for Query, Key and Value A,B LoRA matrices
        self.waqs_value_list = nn.ModuleList([])
        self.wbqs_value_list = nn.ModuleList([])
        self.waks_value_list = nn.ModuleList([])
        self.wbks_value_list = nn.ModuleList([])
        self.wavs_value_list = nn.ModuleList([])
        self.wbvs_value_list = nn.ModuleList([])

        for d in range(self.depth):
            waq_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wbq_value = nn.Linear(hidden_dim, dim * heads * dim_head)

            wak_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wbk_value = nn.Linear(hidden_dim, dim * heads * dim_head)

            wav_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wbv_value = nn.Linear(hidden_dim, dim * heads * dim_head)

            self.waqs_value_list.append(waq_value)
            self.wbqs_value_list.append(wbq_value)
            self.waks_value_list.append(wak_value)
            self.wbks_value_list.append(wbk_value)
            self.wavs_value_list.append(wav_value)
            self.wbvs_value_list.append(wbv_value)


    def finetune(self, emd):
        features = self.mlp(emd)
        weights=OrderedDict()
        for d in range(self.depth):
            # Hyper-network Query LoRA weights
            layer_d_q_a_value_hyper = self.waqs_value_list[d]
            layer_d_q_a_value = layer_d_q_a_value_hyper(features).view(self.inner_dim, self.dim)
            layer_d_q_b_value_hyper = self.wbqs_value_list[d]
            layer_d_q_b_value = layer_d_q_b_value_hyper(features).view(self.inner_dim, self.dim)

            # Hyper-network Key LoRA weights
            layer_d_k_a_value_hyper = self.waks_value_list[d]
            layer_d_k_a_value = layer_d_k_a_value_hyper(features).view(self.inner_dim, self.dim)
            layer_d_k_b_value_hyper = self.wbks_value_list[d]
            layer_d_k_b_value = layer_d_k_b_value_hyper(features).view(self.inner_dim, self.dim)

            # Hyper-network Value LoRA weights
            layer_d_v_a_value_hyper = self.wavs_value_list[d]
            layer_d_v_a_value = layer_d_v_a_value_hyper(features).view(self.inner_dim, self.dim)
            layer_d_v_b_value_hyper = self.wbvs_value_list[d]
            layer_d_v_b_value = layer_d_v_b_value_hyper(features).view(self.inner_dim, self.dim)

            weights["bert.encoder.layer." + str(d) + ".attention.self.query.lora_A.default.weight"] = layer_d_q_a_value
            weights["bert.encoder.layer." + str(d) + ".attention.self.query.lora_B.default.weight"] = layer_d_q_b_value
            weights["bert.encoder.layer." + str(d) + ".attention.self.key.lora_A.default.weight"] = layer_d_k_a_value
            weights["bert.encoder.layer." + str(d) + ".attention.self.key.lora_B.default.weight"] = layer_d_k_b_value
            weights["bert.encoder.layer." + str(d) + ".attention.self.value.lora_A.default.weight"] = layer_d_v_a_value
            weights["bert.encoder.layer." + str(d) + ".attention.self.value.lora_B.default.weight"] = layer_d_v_b_value
        return weights


    def forward(self, idx, test):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict() for x in range(self.client_sample)]
            for d in range(self.depth):
                # Hyper-network Query LoRA weights
                layer_d_q_a_value_hyper = self.waqs_value_list[d]
                layer_d_q_a_value = layer_d_q_a_value_hyper(features).view(-1, self.inner_dim, self.dim)
                layer_d_q_b_value_hyper = self.wbqs_value_list[d]
                layer_d_q_b_value = layer_d_q_b_value_hyper(features).view(-1, self.inner_dim, self.dim)

                # Hyper-network Key LoRA weights
                layer_d_k_a_value_hyper = self.waks_value_list[d]
                layer_d_k_a_value = layer_d_k_a_value_hyper(features).view(-1, self.inner_dim, self.dim)
                layer_d_k_b_value_hyper = self.wbks_value_list[d]
                layer_d_k_b_value = layer_d_k_b_value_hyper(features).view(-1, self.inner_dim, self.dim)

                # Hyper-network Value LoRA weights
                layer_d_v_a_value_hyper = self.wavs_value_list[d]
                layer_d_v_a_value = layer_d_v_a_value_hyper(features).view(-1, self.inner_dim, self.dim)
                layer_d_v_b_value_hyper = self.wbvs_value_list[d]
                layer_d_v_b_value = layer_d_v_b_value_hyper(features).view(-1, self.inner_dim, self.dim)

                for nn in range(self.client_sample):
                    weights[nn]["bert.encoder.layer." + str(d) + ".attention.self.query.lora_A.default.weight"] = layer_d_q_a_value[nn]
                    weights[nn]["bert.encoder.layer." + str(d) + ".attention.self.query.lora_B.default.weight"] = layer_d_q_b_value[nn]
                    weights[nn]["bert.encoder.layer." + str(d) + ".attention.self.key.lora_A.default.weight"] = layer_d_k_a_value[nn]
                    weights[nn]["bert.encoder.layer." + str(d) + ".attention.self.key.lora_B.default.weight"] = layer_d_k_b_value[nn]
                    weights[nn]["bert.encoder.layer." + str(d) + ".attention.self.value.lora_A.default.weight"] = layer_d_v_a_value[nn]
                    weights[nn]["bert.encoder.layer." + str(d) + ".attention.self.value.lora_B.default.weight"] = layer_d_v_b_value[nn]
        else:
            weights = OrderedDict()
            for d in range(self.depth):
                # Hyper-network Query LoRA weights
                layer_d_q_a_value_hyper = self.waqs_value_list[d]
                layer_d_q_a_value = layer_d_q_a_value_hyper(features).view(self.inner_dim, self.dim)
                layer_d_q_b_value_hyper = self.wbqs_value_list[d]
                layer_d_q_b_value = layer_d_q_b_value_hyper(features).view(self.inner_dim, self.dim)

                # Hyper-network Key LoRA weights
                layer_d_k_a_value_hyper = self.waks_value_list[d]
                layer_d_k_a_value = layer_d_k_a_value_hyper(features).view(self.inner_dim, self.dim)
                layer_d_k_b_value_hyper = self.wbks_value_list[d]
                layer_d_k_b_value = layer_d_k_b_value_hyper(features).view(self.inner_dim, self.dim)

                # Hyper-network Value LoRA weights
                layer_d_v_a_value_hyper = self.wavs_value_list[d]
                layer_d_v_a_value = layer_d_v_a_value_hyper(features).view(self.inner_dim, self.dim)
                layer_d_v_b_value_hyper = self.wbvs_value_list[d]
                layer_d_v_b_value = layer_d_v_b_value_hyper(features).view(self.inner_dim, self.dim)

                weights["bert.encoder.layer." + str(d) + ".attention.self.query.lora_A.default.weight"] = layer_d_q_a_value
                weights["bert.encoder.layer." + str(d) + ".attention.self.query.lora_B.default.weight"] = layer_d_q_b_value
                weights["bert.encoder.layer." + str(d) + ".attention.self.key.lora_A.default.weight"] = layer_d_k_a_value
                weights["bert.encoder.layer." + str(d) + ".attention.self.key.lora_B.default.weight"] = layer_d_k_b_value
                weights["bert.encoder.layer." + str(d) + ".attention.self.value.lora_A.default.weight"] = layer_d_v_a_value
                weights["bert.encoder.layer." + str(d) + ".attention.self.value.lora_B.default.weight"] = layer_d_v_b_value
        return weights

