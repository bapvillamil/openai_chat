import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)

        self.fc_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Linear transformation
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        # Split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale

        # Apply mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        # Attention scores and weights
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # Weighted sum of values
        x = torch.matmul(attention, value)

        # Concatenate and linearly transform
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)

        output = self.fc_o(x)

        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(1)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

    def forward(self, src, src_mask):
        # Multi-head attention calculation
        attention = self.self_attention(src, src, src, src_mask)

        # Add and Norm
        src = src + self.dropout(attention)
        src = self.norm1(src)

        # Position-wise Feedforward
        ff_output = self.positionwise_feedforward(src)

        # Add and Norm
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # Self-attention on the target
        self_attention = self.self_attention(tgt, tgt, tgt, tgt_mask)

        # Add and Norm
        tgt = tgt + self.dropout(self_attention)
        tgt = self.norm1(tgt)

        # Multi-head attention with the encoder output
        encoder_attention = self.encoder_attention(tgt, memory, memory, memory_mask)

        # Add and Norm
        tgt = tgt + self.dropout(encoder_attention)
        tgt = self.norm2(tgt)

        # Position-wise Feedforward
        ff_output = self.positionwise_feedforward(tgt)

        # Add and Norm
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)

        return tgt



class InsuraTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model,
                 num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(InsuraTransformer, self).__init__()
        self.d_model = d_model

        self.encoder_embedding = nn.Embedding(955, d_model)
        self.decoder_embedding = nn.Embedding(5155, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=50)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, 5155)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # ... (Mask generation code)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) | subsequent_mask(tgt.size(1)).type_as(src_mask)

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Embedding and Positional Encoding for the source and target sequences
        src = self.dropout((self.encoder_embedding(src) * math.sqrt(self.d_model)) + self.positional_encoding(src))
        tgt = self.dropout((self.decoder_embedding(tgt) * math.sqrt(self.d_model)) + self.positional_encoding(tgt))

        # Forward pass through the encoder layers
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Forward pass through the decoder layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        # Final linear layer to produce the output logits
        output = self.fc(tgt)

        return output

# Function to create a mask for the target sequence with subsequent positions masked
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool().to(device)
    return subsequent_mask