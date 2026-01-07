import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import h5py
import os
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def MSE(y_true, y_pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def RMSE(y_true, y_pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class METRLADataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: shape (num_timesteps, num_nodes, num_features)
        seq_len: input sequence length
        pred_len: prediction sequence length
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = data.shape[0] - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]  # [seq_len, num_nodes, features]
        y = self.data[index + self.seq_len:index + self.seq_len + self.pred_len, :, 0:1]  # predict first feature
        return torch.FloatTensor(x), torch.FloatTensor(y)


def normalize_adjacency_matrix(adj):
    """Symmetric normalization of adjacency matrix"""
    adj = adj + np.eye(adj.shape[0])
    degree = np.diag(np.sum(adj, axis=1))
    degree_inv_sqrt = np.linalg.inv(np.sqrt(degree))
    normalized_adj = degree_inv_sqrt @ adj @ degree_inv_sqrt
    return normalized_adj


def load_metr_la_data(data_path, adj_path, seq_len=12, pred_len=12, subset_ratio=0.1):
    """Load METR-LA data"""
    with h5py.File(data_path, 'r') as f:
        if 'df/block0_values' in f:
            data = f['df/block0_values'][:]
        else:
            data = f['df']['block0_values'][:]
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    with open(adj_path, 'rb') as f:
        adj = pickle.load(f, encoding='latin1')
    if isinstance(adj, list):
        adj_matrix = adj[2] if len(adj) >= 3 else adj[0]
    else:
        adj_matrix = adj
    if hasattr(adj_matrix, 'toarray'):
        adj_matrix = adj_matrix.toarray()
    adj_matrix = adj_matrix.astype(np.float32)

    adj_matrix_normalized = normalize_adjacency_matrix(adj_matrix)

    num_timesteps = data.shape[0]
    train_size = int(0.7 * num_timesteps)
    val_size = int(0.1 * num_timesteps)

    train_data_raw = data[:train_size]
    mean = train_data_raw.mean()
    std = train_data_raw.std()

    if std < 1e-8:
        std = 1.0

    data = (data - mean) / std

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]




    train_dataset = METRLADataset(train_data, seq_len, pred_len)
    val_dataset = METRLADataset(val_data, seq_len, pred_len)
    test_dataset = METRLADataset(test_data, seq_len, pred_len)

    return train_dataset, val_dataset, test_dataset, adj_matrix_normalized, mean, std


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [batch_size, seq_len, d_model]"""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """Transformer for capturing global temporal dependencies"""
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """x: [batch_size, seq_len, hidden_dim]"""
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        return output


class DynamicGraphUpdater(nn.Module):
    """Dynamic Graph Updater"""
    def __init__(self, node_num, decoder_dim, mem_dim):
        super(DynamicGraphUpdater, self).__init__()
        self.node_num = node_num
        self.decoder_dim = decoder_dim
        self.mem_dim = mem_dim

        # Intermediate feature dimension
        hidden_dim = 128

        # Graph update network
        self.update_net = nn.Sequential(
            nn.Linear(decoder_dim + mem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, node_num * node_num)
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(decoder_dim + mem_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_state, prev_graph, meta_memory):
        batch_size = hidden_state.shape[0]
        combined_input = torch.cat([hidden_state, meta_memory], dim=-1)  # [B, N, decoder_dim+mem_dim]
        combined_input = combined_input.mean(dim=1)  # [B, decoder_dim+mem_dim]

        graph_update_vec = self.update_net(combined_input)  # [B, N*N]
        graph_update = graph_update_vec.view(batch_size, self.node_num, self.node_num)

        update_gate = self.gate(combined_input)  # [B, 1]
        update_gate = update_gate.view(batch_size, 1, 1)

        if len(prev_graph.shape) == 2:
            prev_graph = prev_graph.unsqueeze(0).expand(batch_size, -1, -1)

        updated_graph = prev_graph + update_gate * graph_update

        updated_graph = F.relu(updated_graph)
        row_sum = updated_graph.sum(dim=-1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        updated_graph = updated_graph / row_sum
        return updated_graph


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        """
        x: [batch_size, node_num, dim_in]
        supports: list of graph matrices, each can be [node_num, node_num] or [batch_size, node_num, node_num]
        """
        batch_size, node_num, dim_in = x.shape
        x_g = []
        support_set = []

        for support in supports:
            if len(support.shape) == 2:
                support = support.unsqueeze(0).expand(batch_size, -1, -1)

            identity = torch.eye(node_num).to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
            support_ks = [identity, support]
            for k in range(2, self.cheb_k):
                support_k = torch.bmm(2 * support, support_ks[-1]) - support_ks[-2]
                support_ks.append(support_k)

            support_set.extend(support_ks)

        for support in support_set:
            x_g.append(torch.bmm(support, x))

        x_g = torch.cat(x_g, dim=-1)

        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class GEnSHIN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units,
                 num_layers=1, cheb_k=3, ycov_dim=1, mem_num=20, mem_dim=64,
                 cl_decay_steps=2000, use_curriculum_learning=True,
                 adj_matrix=None, use_real_graph=True,
                 transformer_heads=4, transformer_layers=2,
                 use_dynamic_graph=True, device=torch.device('cuda')):
        super(GEnSHIN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim

        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.decoder_dim = self.rnn_units + self.mem_dim

        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.use_real_graph = use_real_graph
        self.use_dynamic_graph = use_dynamic_graph
        self.device = device

        if adj_matrix is not None and use_real_graph:
            if isinstance(adj_matrix, np.ndarray):
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            self.register_buffer('adj_matrix', adj_matrix.to(device))
            self.adj_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.adj_matrix = None
            self.adj_weight = None

        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # Encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim,
                                      self.rnn_units, self.cheb_k, self.num_layers)

        # Temporal Transformer
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=self.rnn_units,
            num_heads=transformer_heads,
            num_layers=transformer_layers
        )

        # Dynamic Graph Updater
        if use_dynamic_graph:
            self.graph_updater = DynamicGraphUpdater(
                node_num=self.num_nodes,
                decoder_dim=self.decoder_dim,
                mem_dim=self.mem_dim
            )

        # Decoder - adapted for dynamic graph
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim,
                                      self.decoder_dim, self.cheb_k, self.num_layers)

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(self.decoder_dim, self.output_dim, bias=True)
        )

        # Graph fusion gating
        if use_real_graph and adj_matrix is not None:
            self.fusion_gate = nn.Parameter(torch.tensor(0.5))

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)

        for param in memory_dict.values():
            nn.init.xavier_normal_(param)

        return memory_dict

    def fuse_graphs(self, learned_graph1, learned_graph2):
        """Fuse learned graphs with real graph"""
        if self.use_real_graph and hasattr(self, 'adj_matrix') and self.adj_matrix is not None:
            gate = torch.sigmoid(self.fusion_gate)
            fused_graph1 = gate * self.adj_matrix + (1 - gate) * learned_graph1
            fused_graph2 = gate * self.adj_matrix + (1 - gate) * learned_graph2
            return [fused_graph1, fused_graph2]
        else:
            return [learned_graph1, learned_graph2]

    def query_memory(self, h_t):
        """Query memory module"""
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)
        att_score = F.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)

        # Get top-2 for contrastive learning
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]  # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]]  # B, N, d

        return value, query, pos, neg, att_score

    def encode_with_transformer(self, h_en):
        """Enhance encoding with Transformer"""
        batch_size, seq_len, num_nodes, hidden_dim = h_en.shape
        h_reshaped = h_en.permute(0, 2, 1, 3).contiguous()
        h_reshaped = h_reshaped.view(batch_size * num_nodes, seq_len, hidden_dim)

        h_transformed = self.temporal_transformer(h_reshaped)

        h_last = h_transformed[:, -1, :]

        h_last = h_last.view(batch_size, num_nodes, hidden_dim)

        return h_last

    def forward(self, x, y_cov=None, labels=None, batches_seen=None, return_dynamic_graphs=False):
        """
        x: [B, T, N, D]
        y_cov: [B, T, N, ycov_dim] (optional)
        labels: [B, T, N, output_dim]
        return_dynamic_graphs
        """
        batch_size = x.shape[0]

        dynamic_graphs = [] if return_dynamic_graphs else None

        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])

        learned_g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        learned_g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)

        supports = self.fuse_graphs(learned_g1, learned_g2)

        static_supports = []
        for support in supports:
            if len(support.shape) == 3:
                support = support[0] if support.shape[0] > 0 else support.mean(dim=0)
            static_supports.append(support)

        supports = static_supports

        init_state = self.encoder.init_hidden(batch_size)
        h_en, state_en = self.encoder(x, init_state, supports)

        h_transformed = self.encode_with_transformer(h_en)

        h_att, query, pos, neg, att_score = self.query_memory(h_transformed)
        h_t = torch.cat([h_transformed, h_att], dim=-1)  # [B, N, hidden+mem_dim]

        ht_list = [h_t] * self.num_layers

        go = torch.zeros((batch_size, self.num_nodes, self.output_dim), device=self.device)
        out = []

        current_graph = supports[0]
        if return_dynamic_graphs:
            dynamic_graphs.append(current_graph.detach().cpu())

        for t in range(self.horizon):
            if self.use_dynamic_graph and hasattr(self, 'graph_updater'):
                # Update graph
                current_graph = self.graph_updater(ht_list[0], current_graph, h_att)
                if return_dynamic_graphs:
                    if len(current_graph.shape) == 3:
                        dynamic_graphs.append(current_graph[0].detach().cpu())
                    else:
                        dynamic_graphs.append(current_graph.detach().cpu())
                dynamic_supports = [current_graph, current_graph]
            else:
                dynamic_supports = supports

            # Prepare decoder input
            if y_cov is not None and y_cov.shape[1] > t:
                decoder_input = torch.cat([go, y_cov[:, t, ...]], dim=-1)
            else:
                decoder_input = go

            h_de, ht_list = self.decoder(decoder_input, ht_list, dynamic_supports)

            # Generate output
            go = self.proj(h_de)
            out.append(go)

            # Curriculum learning
            if self.training and self.use_curriculum_learning and labels is not None:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)  # [B, T, N, output_dim]

        if return_dynamic_graphs:
            return output, h_att, query, pos, neg, att_score, dynamic_graphs
        else:
            return output, h_att, query, pos, neg, att_score


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, query, pos, neg):
        pos_dist = torch.norm(query - pos, dim=-1).pow(2)
        neg_dist = torch.norm(query - neg, dim=-1).pow(2)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()


class GEnSHIN_Loss(nn.Module):
    def __init__(self, lambda1=0.01, lambda2=0.01, margin=1.0):
        super(GEnSHIN_Loss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.margin = margin
        self.mae_loss = nn.L1Loss()
        self.contrastive_loss = ContrastiveLoss(margin)

    def forward(self, predictions, targets, query, pos, neg):
        task_loss = self.mae_loss(predictions, targets)
        consistency_loss = torch.norm(query - pos, dim=-1).pow(2).mean()
        contrast_loss = self.contrastive_loss(query, pos, neg)
        total_loss = task_loss + self.lambda1 * consistency_loss + self.lambda2 * contrast_loss

        return total_loss, {
            'task_loss': task_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'total_loss': total_loss.item()
        }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train one epoch"""
    model.train()
    total_loss = 0
    task_loss_sum = 0
    consistency_loss_sum = 0
    contrast_loss_sum = 0

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        predictions, h_att, query, pos, neg, _ = model(batch_x, labels=batch_y, batches_seen=batch_idx)

        loss, loss_dict = criterion(predictions, batch_y, query, pos, neg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        total_loss += loss_dict['total_loss']
        task_loss_sum += loss_dict['task_loss']
        consistency_loss_sum += loss_dict['consistency_loss']
        contrast_loss_sum += loss_dict['contrast_loss']

        if batch_idx % 10 == 0:
            print(f"    Batch {batch_idx}/{len(train_loader)}, "
                  f"Task Loss: {loss_dict['task_loss']:.4f}, "
                  f"Total Loss: {loss_dict['total_loss']:.4f}")

    avg_total_loss = total_loss / len(train_loader)
    avg_task_loss = task_loss_sum / len(train_loader)
    avg_consistency_loss = consistency_loss_sum / len(train_loader)
    avg_contrast_loss = contrast_loss_sum / len(train_loader)
    print(f"  Epoch {epoch}/{total_epochs} - "
          f"Train Task Loss: {avg_task_loss:.4f}, "
          f"Consistency Loss: {avg_consistency_loss:.4f}, "
          f"Contrast Loss: {avg_contrast_loss:.4f}")

    return avg_total_loss, avg_task_loss, avg_consistency_loss, avg_contrast_loss


def evaluate(model, data_loader, device, mean, std, desc="Validation"):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Forward pass
            predictions, _, _, _, _, _ = model(batch_x)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # Combine all batches
    if len(all_predictions) > 0:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        mae_norm = MAE(all_targets, all_predictions)
        rmse_norm = RMSE(all_targets, all_predictions)
        mape_norm = MAPE(all_targets, all_predictions)

        all_predictions_orig = all_predictions * std + mean
        all_targets_orig = all_targets * std + mean

        mae_orig = MAE(all_targets_orig, all_predictions_orig)
        rmse_orig = RMSE(all_targets_orig, all_predictions_orig)
        mape_orig = MAPE(all_targets_orig, all_predictions_orig)

        print(f"  {desc} Results (Normalized Scale):")
        print(f"    MAE: {mae_norm:.4f}, RMSE: {rmse_norm:.4f}, MAPE: {mape_norm:.2f}%")
        print(f"  {desc} Results (Original Scale):")
        print(f"    MAE: {mae_orig:.4f}, RMSE: {rmse_orig:.4f}, MAPE: {mape_orig:.2f}%")

        return {
            'mae_norm': mae_norm,
            'rmse_norm': rmse_norm,
            'mape_norm': mape_norm,
            'mae_orig': mae_orig,
            'rmse_orig': rmse_orig,
            'mape_orig': mape_orig,
            'predictions': all_predictions,
            'targets': all_targets
        }
    else:
        print(f"  {desc} set is empty")
        return None




def main():

    # ============ Configuration Parameters ============
    SEQ_LEN = 12  # Input sequence length
    PRED_LEN = 12  # Prediction sequence length
    SUBSET_RATIO = 1
    BATCH_SIZE = 256
    EPOCHS = 100

    # ============ File Path Configuration ============
    CURRENT_DIR = os.environ.get('PROJECT_DIR', os.getcwd())
    DATA_PATH = os.path.join(CURRENT_DIR, "data/METR-LA.h5")
    ADJ_PATH = os.path.join(CURRENT_DIR, "data/adj_METR-LA.pkl")



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    try:
        print("Data Loading Phase")

        train_dataset, val_dataset, test_dataset, adj_matrix, data_mean, data_std = load_metr_la_data(
            data_path=DATA_PATH,
            adj_path=ADJ_PATH,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            subset_ratio=SUBSET_RATIO
        )


    except Exception as e:
        print(f"\n data loading failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    num_nodes = adj_matrix.shape[0]

    # Model configuration
    model_config = {
        'num_nodes': num_nodes,
        'input_dim': 1,
        'output_dim': 1,
        'horizon': PRED_LEN,
        'rnn_units': 128,
        'num_layers': 5,
        'cheb_k': 3,
        'ycov_dim': 0,
        'mem_num': 20,
        'mem_dim': 128,
        'use_real_graph': True,
        'use_dynamic_graph': True,
        'transformer_heads': 4,
        'transformer_layers': 2,
        'cl_decay_steps': 2000,
        'use_curriculum_learning': True,
        'adj_matrix': adj_matrix,
        'device': device
    }

    model = GEnSHIN(**model_config)

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = GEnSHIN_Loss(lambda1=0.01, lambda2=0.01)


    print(f"Starting Training (Total {EPOCHS} epochs)")
    # Record training history
    train_history = {
        'total_loss': [],
        'task_loss': [],
        'consistency_loss': [],
        'contrast_loss': []
    }

    val_history = {
        'mae': [],
        'rmse': [],
        'mape': []
    }

    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)

        print("Training phase...")
        train_loss, task_loss, consistency_loss, contrast_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, EPOCHS)

        train_history['total_loss'].append(train_loss)
        train_history['task_loss'].append(task_loss)
        train_history['consistency_loss'].append(consistency_loss)
        train_history['contrast_loss'].append(contrast_loss)

        val_results = evaluate(model, val_loader, device, data_mean, data_std, "Validation")

        if val_results is not None:
            val_mae = val_results['mae_orig']
            val_rmse = val_results['rmse_orig']
            val_mape = val_results['mape_orig']

            # Record validation history
            val_history['mae'].append(val_mae)
            val_history['rmse'].append(val_rmse)
            val_history['mape'].append(val_mape)

            if val_mae < best_val_mae - 1e-4:
                best_val_mae = val_mae
                patience_counter = 0

            else:
                patience_counter += 1
                print(f"  Early stopping counter: {patience_counter}/20")

            if patience_counter >= 20:
                print(
                    f"\nEarly stopping triggered! No improvement in Val MAE for {patience_counter} consecutive epochs")
                break


    test_results = evaluate(model, test_loader, device, data_mean, data_std, "Test")
    print(test_results)
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    try:
        model = main()
    except Exception as e:
        print(f"\nProgram execution error: {e}")
        import traceback
        traceback.print_exc()