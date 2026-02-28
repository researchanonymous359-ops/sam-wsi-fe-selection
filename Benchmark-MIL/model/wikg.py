# model/wikg.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention

# It is recommended to set Anomaly detection to True only during debugging
torch.autograd.set_detect_anomaly(False)

class WiKG(nn.Module):
    def __init__(self, dim_in=384, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn'):
        super().__init__()

        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError
        
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)

        if pool == "mean":
            self.readout = global_mean_pool 
        elif pool == "max":
            self.readout = global_max_pool 
        elif pool == "attn":
            att_net = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden // 2), 
                nn.LeakyReLU(), 
                nn.Linear(dim_hidden // 2, 1)
            )      
            self.readout = GlobalAttention(att_net)

    def forward(self, x):
        # x: [N, D] or [1, N, D] -> [1, N, D]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = self._fc1(x)    # [B, N, C]

        # Instance Normalization-like shift
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)

        Nb_h = e_t[batch_indices, topk_index_expanded, :] 

        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding
        
        h = self.message_dropout(embedding)

        # Pooling (Global Attention)
        # h.squeeze(0) -> [N, C] because PyG pooling expects [num_nodes, features]
        # batch=None assumes single graph per forward pass (batch size 1)
        h_pooled = self.readout(h.squeeze(0), batch=None)
        h_pooled = self.norm(h_pooled)
        logits = self.fc(h_pooled)

        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(logits, dim=1)

        return logits, y_prob, y_hat
    
    def get_attention_maps(self, x):
        """
        Returns attention scores from the readout layer if available.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = self._fc1(x)
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)

        Nb_h = e_t[batch_indices, topk_index_expanded, :] 

        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))

        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding
        
        h = self.message_dropout(embedding)

        # Get Attention Scores
        h_squeezed = h.squeeze(0) # [N, C]
        
        if isinstance(self.readout, GlobalAttention):
            # gate_nn returns raw logits
            attn_logits = self.readout.gate_nn(h_squeezed)
            attn_scores = torch.sigmoid(attn_logits).squeeze(-1) # [N]
        else:
            # Mean/Max pooling does not have attention scores, so return uniform values
            attn_scores = torch.ones(h_squeezed.size(0), device=h.device) / h_squeezed.size(0)

        h_pooled = self.readout(h_squeezed, batch=None)
        h_pooled = self.norm(h_pooled)
        logits = self.fc(h_pooled)

        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(logits, dim=1)

        return logits, y_prob, y_hat, attn_scores


if __name__ == "__main__":
    # Test
    data = torch.randn((1, 100, 384)).cuda()
    model = WiKG(dim_in=384, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn').cuda()
    
    logits, probs, preds = model(data)
    print(f"Output shapes: {logits.shape}, {probs.shape}, {preds.shape}")
    
    _, _, _, attn = model.get_attention_maps(data)
    print(f"Attention map shape: {attn.shape}")