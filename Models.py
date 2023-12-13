import math
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU

from utils.utils_misc import get_root_dir


def kernal_mus(n_kernels):
    """
    get the mean mu for each gaussian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels, sigma_val):
    assert n_kernels >= 1
    l_sigma = [0.001] + [sigma_val] * (n_kernels - 1)
    return l_sigma


class inference_model(nn.Module):
    def __init__(self, bert_model, args, config, tokenizer=None):
        super(inference_model, self).__init__()
        self.args = args
        self.cuda = args.cuda
        self.device = 'cuda' if self.cuda and torch.cuda.is_available() else 'cpu'
        self.bert_hidden_dim = args.bert_hidden_dim
        self.batch_size = args.train_batch_size
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = bert_model
        self.evi_num = args.evi_num
        self.nlayer = args.layer
        self.kernel = args.kernel
        self.sigma_val = args.sigma
        self.proj_inference_de = nn.Linear(self.bert_hidden_dim * 2, self.num_labels)
        self.proj_att = nn.Linear(self.kernel, 1)
        self.proj_input_de = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)

        self.proj_select = nn.Linear(self.kernel, 1)
        self.mu = Variable(torch.FloatTensor(kernal_mus(self.kernel)), requires_grad=False).view(1, 1, 1, self.kernel).to(self.device)
        self.sigma = Variable(torch.FloatTensor(kernel_sigmas(self.kernel, self.sigma_val)), requires_grad=False).view(1, 1, 1,
                                                                                                       self.kernel).to(self.device)
        self.tokenizer = tokenizer

        config_kgat = config["KernelGAT"]
        type = torch.float32

        self.pr_weight = torch.empty((3, args.evi_num), dtype=type)
        nn.init.uniform_(self.pr_weight)
        self.pr_param = nn.Parameter(self.pr_weight)
        self.mode = args.mode

        self.trans_mat_weight = torch.empty((self.evi_num, self.evi_num), device=self.device, dtype=type)

        # Translation matrix
        mean = config_kgat.getfloat("translation_mat_weight_mean")
        std = config_kgat.getfloat("translation_mat_weight_std")
        nn.init.normal_(self.trans_mat_weight, mean, std)
        self.param_trans_mat = nn.Parameter(self.trans_mat_weight, requires_grad=True)

        config_s = config["gat.social"]

        mean = config_kgat.getfloat("linear_weight_mean")
        std = config_kgat.getfloat("linear_weight_std")

        self.proj_pred_interact = nn.Linear(2, 1)
        self.param_pred_K = torch.empty(1)
        nn.init.normal_(self.param_pred_K, mean, std)

        self.user_embed_dim = args.user_embed_dim
        self.num_users = args.num_users


        self.proj_gat = nn.Sequential(
            Linear(self.bert_hidden_dim * 2, 128),
            ReLU(True),
            Linear(128, 1)
        )

        self.proj_gat_usr = nn.Sequential(
            Linear(self.user_embed_dim, 128, bias=False),
            ReLU(True),
            Linear(128, 1, bias=False)
        )
        self.proj_user = Linear(self.user_embed_dim * 2, self.bert_hidden_dim * 2, bias=False)
        nn.init.normal_(self.proj_user.weight, mean, std)
        nn.init.normal_(self.proj_gat_usr[0].weight, mean, std)
        nn.init.normal_(self.proj_gat_usr[2].weight, mean, std)

    def self_attention(self, inputs, inputs_hiddens, mask, mask_evidence, index, trans_mat_prior=None):
        idx = torch.LongTensor([index]).to(self.device)

        mask = mask.view([-1, self.evi_num, self.max_len])
        mask_evidence = mask_evidence.view([-1, self.evi_num, self.max_len])
        own_hidden = torch.index_select(inputs_hiddens, 1, idx)
        own_mask = torch.index_select(mask, 1, idx)
        own_input = torch.index_select(inputs, 1, idx)
        own_hidden = own_hidden.repeat(1, self.evi_num, 1, 1)
        own_mask = own_mask.repeat(1, self.evi_num, 1)
        own_input = own_input.repeat(1, self.evi_num, 1)
        hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=-1)
        own_norm = F.normalize(own_hidden, p=2, dim=-1)
        att_score = self.get_intersect_matrix_att(hiddens_norm.view(-1, self.max_len, self.bert_hidden_dim),
                                                  own_norm.view(-1, self.max_len, self.bert_hidden_dim),
                                                  mask_evidence.view(-1, self.max_len), own_mask.view(-1, self.max_len))

        att_score = att_score.view(-1, self.evi_num, self.max_len, 1)
        denoise_inputs = torch.sum(att_score * inputs_hiddens, 2)
        weight_inp = torch.cat([own_input, inputs], -1)
        z_q_z_v = weight_inp
        # MLP()
        weight_inp = self.proj_gat(weight_inp)
        weight_inp = F.softmax(weight_inp, dim=1)
        outputs = (inputs * weight_inp).sum(dim=1)
        weight_de = torch.cat([own_input, denoise_inputs], -1)

        z_qv_z_v = weight_de
        weight_de = self.proj_gat(weight_de)
        if trans_mat_prior is not None:
            weight_de = torch.cat([weight_de, trans_mat_prior[:,index].reshape(-1, self.evi_num, 1)], dim=2)
            weight_de = self.proj_pred_interact(weight_de)
        weight_de = F.softmax(weight_de, dim=1)

        outputs_de = (denoise_inputs * weight_de).sum(dim=1)
        return outputs, outputs_de, z_qv_z_v

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1], 1)
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1],
                                                                      d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu.to(self.device)) ** 2) / (self.sigma.to(self.device) ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2) # If merge content and social representation here
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q

        log_pooling_sum_all = torch.sum(log_pooling_sum, 1) / (torch.sum(attn_q, 1) + 1e-10)
        log_pooling_sum = self.proj_select(log_pooling_sum_all).view([-1, 1])
        return log_pooling_sum, log_pooling_sum_all

    def get_intersect_matrix_att(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1])
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1],
                                                                      d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu.to(self.device)) ** 2) / (self.sigma.to(self.device) ** 2) / 2)) * attn_d

        log_pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(log_pooling_sum, min=1e-10))
        log_pooling_sum = self.proj_att(log_pooling_sum).squeeze(-1)

        log_pooling_sum = log_pooling_sum.masked_fill_((1 - attn_q).bool(), -1e4)
        log_pooling_sum = F.softmax(log_pooling_sum, dim=1)
        return log_pooling_sum

    def unpack_inputs(self, inputs):
        inp_tensor, msk_tensor, seg_tensor, step = inputs
        if self.cuda:
            msk_tensor = msk_tensor.view(-1, self.max_len).cuda()
            inp_tensor = inp_tensor.view(-1, self.max_len).cuda()
            seg_tensor = seg_tensor.view(-1, self.max_len).cuda()
        else:
            msk_tensor = msk_tensor.view(-1, self.max_len)
            inp_tensor = inp_tensor.view(-1, self.max_len)
            seg_tensor = seg_tensor.view(-1, self.max_len)
        return inp_tensor, msk_tensor, seg_tensor, step

    def predict_prior(self, score):
        prior = self.proj_score(score)
        return prior

    def reshape_input_and_masks(self, inputs_hiddens, msk_tensor, seg_tensor):

        mask_text = msk_tensor.view(-1, self.max_len).float()
        mask_text[:, 0] = 0.0
        mask_claim = (1 - seg_tensor.float()) * mask_text
        mask_evidence = seg_tensor.float() * mask_text
        inputs_hiddens = inputs_hiddens.view(-1, self.max_len, self.bert_hidden_dim)


        inputs_hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=2)
        return mask_text, mask_claim, mask_evidence, inputs_hiddens, inputs_hiddens_norm


    def sentence_level_embedding(self, inputs_hiddens, inputs, msk_tensor, seg_tensor, delta):
        mask_text, mask_claim, mask_evidence, inputs_hiddens, inputs_hiddens_norm = self.reshape_input_and_masks(inputs_hiddens, msk_tensor, seg_tensor)

        log_pooling_sum, log_pooling_sum_all = self.get_intersect_matrix(inputs_hiddens_norm, inputs_hiddens_norm, mask_claim,
                                                    mask_evidence)

        log_pooling_sum = log_pooling_sum.view([-1, self.evi_num, 1])
        select_prob = F.softmax(log_pooling_sum, dim=1)

        inputs = inputs.view([-1, self.evi_num, self.bert_hidden_dim])
        inputs_hiddens = inputs_hiddens.view([-1, self.evi_num, self.max_len, self.bert_hidden_dim])

        inputs_att_de = []
        z_qv_z_v_all = []

        for i in range(self.evi_num):

            outputs, outputs_de, z_qv_z_v = self.self_attention(inputs, inputs_hiddens, mask_text, mask_text, i)
            inputs_att_de.append(outputs_de)
            z_qv_z_v_all.append(z_qv_z_v)
        inputs_att = inputs.view([-1, self.evi_num, self.bert_hidden_dim])

        inputs_att_de = torch.cat(inputs_att_de, dim=1)
        z_qv_z_v_all = torch.cat(z_qv_z_v_all, dim=1)
        inputs_att_de = inputs_att_de.view([-1, self.evi_num, self.bert_hidden_dim])
        z_qv_z_v_all = z_qv_z_v_all.view([-1, self.evi_num, self.evi_num, self.bert_hidden_dim])

        return select_prob, inputs_att, inputs_att_de, z_qv_z_v_all
    
    def coarse_level_result(self, inputs_hiddens, inputs, msk_tensor, seg_tensor, delta):
        mask_text, mask_claim, mask_evidence, inputs_hiddens, inputs_hiddens_norm = self.reshape_input_and_masks(inputs_hiddens, msk_tensor, seg_tensor)

        # ------------------------------------------
        # tree-level embedding
        # ------------------------------------------
        inputs = inputs.view([-1, self.evi_num, self.bert_hidden_dim])

        inputs_hiddens = inputs_hiddens.view([-1, self.evi_num, self.max_len, self.bert_hidden_dim])

        inputs_att_de = []
        z_qv_z_v_all = []

        for i in range(self.evi_num):

            outputs, outputs_de, z_qv_z_v = self.self_attention(inputs, inputs_hiddens, mask_text, mask_text, i)
            inputs_att_de.append(outputs_de)
            z_qv_z_v_all.append(z_qv_z_v)

        inputs_att = inputs.view([-1, self.evi_num, self.bert_hidden_dim])
        inputs_att_de = torch.cat(inputs_att_de, dim=1)
        z_qv_z_v_all = torch.cat(z_qv_z_v_all, dim=1)
        inputs_att_de = inputs_att_de.view([-1, self.evi_num, self.bert_hidden_dim])
        z_qv_z_v_all = z_qv_z_v_all.view([-1, self.evi_num, self.evi_num, self.bert_hidden_dim])


        return inputs_att, inputs_att_de, z_qv_z_v_all



    def forward(self, inputs):
        # get input tensor, mask tensor, seg tensor firstly
        inp_tensor, msk_tensor, seg_tensor, step = self.unpack_inputs(inputs)

        # get input hiddens and inputs embedding
        inputs_hiddens, inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)
        # initialize parameter 
        lambda = 0.1

        # get attention score and embedding
        select_prob, inputs_att, inputs_att_de, z_qv_z_v_all = self.sentence_level_embedding(inputs_hiddens, inputs, msk_tensor, seg_tensor, lambda)
        # inputs_att = torch.cat([inputs_att, inputs_att_de], -1)

        # merge 2 embeddings
        inputs_att = torch.cat([inputs_att, inputs_att_de], -1)

        # get inference feature
        inference_feature = self.proj_inference_de(inputs_att)

        # get probability for sentence-level misinformation
        class_prob = F.softmax(inference_feature, dim=2)
        prob_sen = torch.sum(select_prob * class_prob, 1)
        # prob = torch.log(prob)

        # get attention score for each updated sentence representation
        inputs_att, att_j, z_qv_z_v_all = self.coarse_level_result(inputs_hiddens, inputs, msk_tensor, seg_tensor, lambda)
        # get probability for coarse-level fake news
        prob_news = torch.sum(att_j, prob_sen)

        return prob_sen, prob_news
