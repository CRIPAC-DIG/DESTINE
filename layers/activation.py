import torch
from torch.nn import functional as F


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=False,
                 relu_before_att=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att
        self.relu_before_att = relu_before_att

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

    def get_attention_map(self, query, key):
        queries = self.W_query(query)
        keys = self.W_key(key)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(self.head_dim, dim=2)
        K = keys.split(self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)

        # [num_heads * batch, len(field_dims), len(field_dims)]
        output = torch.bmm(Q_, K_.transpose(1, 2))
        output /= K_.shape[-1] ** 0.5
        output = F.softmax(output, dim=2)

        return output

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
        else:
            queries = self.W_query(query)
            keys = self.W_key(key)
            values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(self.head_dim, dim=2)
        K = keys.split(self.head_dim, dim=2)
        V = values.split(self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # [num_heads * batch, len(field_dims), len(field_dims)]
        output = torch.bmm(Q_, K_.transpose(1, 2))
        if self.scale_att:
            output /= K_.shape[-1] ** 0.5
        output = F.softmax(output, dim=2)
        # print(output.cpu().detach().numpy())
        output = F.dropout(output, self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class DisentangledSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=False,
                 relu_before_att: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att
        self.relu_before_att = relu_before_att

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_unary = torch.nn.Linear(self.embed_dim, num_heads)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

    def get_attention_map(self, query, key):
        queries = self.W_query(query)
        keys = self.W_key(key)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        # pairwise /= K_.shape[-1] ** 0.5
        if self.scale_att:
            pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)

        unary = self.W_unary(key)  # [batch, len(field_dims), num_heads]
        unary = F.softmax(unary, dim=1)
        unary = unary.split(split_size=1, dim=2)  # num_heads * [batch, len(field_dims), 1]
        unary = torch.cat(unary, dim=0)  # [num_heads * batch, len(fields_dims), 1]
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]

        return pairwise, unary

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
        else:
            queries = self.W_query(query)
            keys = self.W_key(key)
            values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        # pairwise /= K_.shape[-1] ** 0.5
        if self.scale_att:
            pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)

        unary = self.W_unary(key)  # [batch, len(field_dims), num_heads]
        unary = F.softmax(unary, dim=1)
        unary = unary.split(split_size=1, dim=2)  # num_heads * [batch, len(field_dims), 1]
        unary = torch.cat(unary, dim=0)  # [num_heads * batch, len(fields_dims), 1]
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]

        # print(pairwise.cpu().detach().numpy())
        # print(unary.cpu().detach().numpy())

        output = F.dropout(pairwise + unary, self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class DisentangledSelfAttentionAverage(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, opt_payload=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_unary = torch.nn.Linear(self.embed_dim, num_heads)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

        self.weights = opt_payload
        weight_sum = float(self.weights[0] + self.weights[1])
        self.weights[0] = self.weights[0] / weight_sum
        self.weights[1] = self.weights[1] / weight_sum

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        queries = self.W_query(query)
        keys = self.W_key(key)
        values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        # pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)

        unary = self.W_unary(key)  # [batch, len(field_dims), num_heads]
        unary = F.softmax(unary, dim=1)
        unary = unary.split(split_size=1, dim=2)  # num_heads * [batch, len(field_dims), 1]
        unary = torch.cat(unary, dim=0)  # [num_heads * batch, len(fields_dims), 1]
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]

        # print(pairwise.cpu().detach().numpy())
        # print(unary.cpu().detach().numpy())

        output = pairwise * self.weights[0] + unary * self.weights[1]

        output = F.dropout(output, self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class DisentangledSelfAttentionAverageLearnable(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=False,
                 relu_before_att=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att
        self.relu_before_att = relu_before_att

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_unary = torch.nn.Linear(self.embed_dim, num_heads)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

        self.weights = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
        else:
            queries = self.W_query(query)
            keys = self.W_key(key)
            values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        if self.scale_att:
            pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)

        unary = self.W_unary(key)  # [batch, len(field_dims), num_heads]
        unary = F.softmax(unary, dim=1)
        unary = unary.split(split_size=1, dim=2)  # num_heads * [batch, len(field_dims), 1]
        unary = torch.cat(unary, dim=0)  # [num_heads * batch, len(fields_dims), 1]
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]

        # print(pairwise.cpu().detach().numpy())
        # print(unary.cpu().detach().numpy())

        output = pairwise * self.weights[0] + unary * (1 - self.weights[0])

        output = F.dropout(output, self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class UnarySelfAttention(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=False,
                 relu_before_att=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att
        self.relu_before_att = relu_before_att

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_unary = torch.nn.Linear(self.embed_dim, num_heads)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
        else:
            queries = self.W_query(query)
            keys = self.W_key(key)
            values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]

        unary = torch.bmm(K_, mu_Q.transpose(1, 2))  # [num_heads * batch, len(fields_dims), 1]
        unary = F.softmax(unary, dim=1)

        output = F.dropout(unary, self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = output.transpose(1, 2) * V_.transpose(1, 2)
        output = output.transpose(1, 2)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class PairwiseSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=False,
                 relu_before_att=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att
        self.relu_before_att = relu_before_att

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
        else:
            queries = self.W_query(query)
            keys = self.W_key(key)
            values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        if self.scale_att:
            pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)

        output = F.dropout(pairwise, self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class DisentangledSelfAttentionVariant1(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=False,
                 relu_before_att=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att
        self.relu_before_att = relu_before_att

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_unary = torch.nn.Linear(self.embed_dim, num_heads)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
        else:
            queries = self.W_query(query)
            keys = self.W_key(key)
            values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        if self.scale_att:
            pairwise /= K_.shape[-1] ** 0.5
        # pairwise = F.softmax(pairwise, dim=2)

        unary = self.W_unary(key)  # [batch, len(field_dims), num_heads]
        # unary = F.softmax(unary, dim=1)
        unary = unary.split(split_size=1, dim=2)  # num_heads * [batch, len(field_dims), 1]
        unary = torch.cat(unary, dim=0)  # [num_heads * batch, len(fields_dims), 1]
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]

        output = F.dropout(F.softmax(pairwise + unary, dim=2), self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class DisentangledSelfAttentionVariant2(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=False,
                 relu_before_att=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att
        self.relu_before_att = relu_before_att

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
        else:
            queries = self.W_query(query)
            keys = self.W_key(key)
            values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        if self.scale_att:
            pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)

        unary = torch.bmm((K_ + mu_K), mu_Q.transpose(1, 2))  # [num_heads * batch, len(fields_dims), 1]
        unary = F.softmax(unary, dim=1)
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]

        output = F.dropout(pairwise + unary, self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class DisentangledSelfAttentionWeighted(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=False,
                 relu_before_att=False, magic=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att
        self.relu_before_att = relu_before_att
        self.magic = magic

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_weight = torch.nn.Linear(self.embed_dim, self.atten_dim)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            queries_prime = F.relu(self.W_weight(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
        else:
            queries = self.W_query(query)
            queries_prime = self.W_weight(query)
            keys = self.W_key(key)
            values = self.W_value(value)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        Q_prime = queries_prime.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        Q_prime_ = torch.cat(Q_prime, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_Q_prime = Q_prime_.mean(dim=1, keepdim=True)
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        if self.scale_att:
            pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)
        if self.magic:
            pairwise = pairwise / 2.

        unary = torch.bmm((K_ + mu_K), mu_Q_prime.transpose(1, 2))  # [num_heads * batch, len(fields_dims), 1]
        unary = F.softmax(unary, dim=1)
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]
        if self.magic:
            unary = unary / 2.

        output = F.dropout(pairwise + unary, self.dropout)

        # print(pairwise.cpu().detach().numpy())
        # print(unary.cpu().detach().numpy())

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output


class ScaledDisentangledSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim=16, atten_dim=64, num_heads=2, dropout=0, residual=True, scale_att=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.atten_dim = atten_dim
        self.head_dim = atten_dim // num_heads
        self.dropout = dropout
        self.use_residual = not (residual is False)
        self.scale_att = scale_att

        self.W_query = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_key = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_value = torch.nn.Linear(self.embed_dim, self.atten_dim)
        self.W_unary = torch.nn.Linear(self.embed_dim, num_heads)

        if residual is True:
            self.W_residual = torch.nn.Linear(self.embed_dim, self.atten_dim)
        elif residual is not None:
            self.W_residual = residual

    def get_attention_map(self, query, key):
        queries = self.W_query(query)
        keys = self.W_key(key)

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        # pairwise /= K_.shape[-1] ** 0.5
        if self.scale_att:
            pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)

        unary = self.W_unary(key)  # [batch, len(field_dims), num_heads]
        unary = F.softmax(unary, dim=1)
        unary = unary.split(split_size=1, dim=2)  # num_heads * [batch, len(field_dims), 1]
        unary = torch.cat(unary, dim=0)  # [num_heads * batch, len(fields_dims), 1]
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]

        return pairwise, unary

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        queries = self.W_query(query)
        keys = self.W_key(key)
        values = self.W_value(value)
        # queries = F.relu(self.W_query(query))
        # keys = F.relu(self.W_key(key))
        # values = F.relu(self.W_value(value))

        # split to num_heads * [batch, len(field_dims), head_dim]
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)

        # whiten Q and K
        mu_Q = Q_.mean(dim=1, keepdim=True)  # [num_heads * batch, 1, head_dim]
        mu_K = K_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        # pairwise /= K_.shape[-1] ** 0.5
        pairwise = F.softmax(pairwise, dim=2)

        unary = self.W_unary(key)  # [batch, len(field_dims), num_heads]
        unary = F.softmax(unary, dim=1)
        unary = unary.split(split_size=1, dim=2)  # num_heads * [batch, len(field_dims), 1]
        unary = torch.cat(unary, dim=0)  # [num_heads * batch, len(fields_dims), 1]
        unary = unary.transpose(1, 2)  # [num_heads * batch, 1, len(fields_dims)]

        # print(pairwise.cpu().detach().numpy())
        # print(unary.cpu().detach().numpy())

        output = F.dropout(pairwise + unary, self.dropout)

        # weighted sum for values
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape
        # num_heads * [batch, len(field_dims), head_dim]
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)

        if self.use_residual:
            output += self.W_residual(query)

        return output
