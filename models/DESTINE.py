import torch
import torch.nn.functional as F

from layers.activation import DisentangledSelfAttention
from torchfm.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DESTINE(torch.nn.Module):
    def __init__(
            self, field_dims, embed_dim, atten_embed_dim, num_heads,
            num_layers, mlp_dims, dropout_mlp, dropout_att,
            base_model=DisentangledSelfAttention,
            scale_att: bool = False,
            relu_before_att: bool = False,
            res_mode: str = 'last_layer',
            deep: bool = False,
            magic: bool = False
    ):
        """
        :param field_dims:
        :param embed_dim:
        :param atten_embed_dim:
        :param num_heads:
        :param num_layers:
        :param mlp_dims:
        :param dropout_mlp:
        :param dropout_att:
        :param base_model:
        :param scale_att:
        :param relu_before_att:
        :param res_mode: Possible values are `'last_layer', 'each_layer', 'shared', 'none'`.
        :param deep:
        """
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.res_mode = res_mode
        self.deep = deep
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout_mlp)
        if self.res_mode in ['last_layer', 'shared']:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.self_attns = [base_model(embed_dim, atten_embed_dim, num_heads, dropout=dropout_att, scale_att=scale_att,
                                      relu_before_att=relu_before_att, residual=self.res_mode == 'each_layer')]
        self.self_attns += [
            base_model(
                atten_embed_dim, atten_embed_dim, num_heads, scale_att=scale_att,
                dropout=dropout_att, residual=self.res_mode == 'each_layer')
            for _ in range(num_layers - 1)
        ]
        self.self_attns = torch.nn.ModuleList(self.self_attns)
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)

    def get_attention_map(self, x):
        embed_x = self.embedding(x)
        cross_term = self.atten_embedding(embed_x)
        return self.self_attns[0].get_attention_map(cross_term, cross_term)

    def forward(self, x):
        embed_x = self.embedding(x)
        # cross_term = self.atten_embedding(embed_x)
        cross_term = embed_x
        for self_attn in self.self_attns:
            cross_term = self_attn(cross_term, cross_term, cross_term)
        if self.res_mode == 'last_layer':
            V_res = self.V_res_embedding(embed_x)
            cross_term += V_res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        if self.deep:
            x = self.linear(x) + self.attn_fc(cross_term) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        else:
            x = self.linear(x) + self.attn_fc(cross_term)
        return torch.sigmoid(x.squeeze(1))
