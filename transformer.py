import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, 'Dimension check'
        
        self.query = nn.Linear(input_dim, input_dim) #represent the elements that are seeking attention
        self.key = nn.Linear(input_dim, input_dim)   #keys and values represent all elements in the sequence
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()

        queries = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim) 
        keys = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)) #The attention scores are computed as the dot product between the queries and the keys
        attention_weights = torch.softmax(attention_scores, dim=1)
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.view(batch_size, seq_length, -1)

        return attended_values

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_rate = 0.1):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attended = self.attention(x)
        x = x + self.dropout1(attended)
        x = self.layer_norm1(x)

        fed_forward = self.feed_forward(x)
        x = x + self.dropout2(fed_forward)
        x = self.layer_norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(input_dim, num_heads, hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(input_dim, num_heads, hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, num_heads, hidden_dim, num_encoder_layers, dropout_rate)
        self.decoder = TransformerDecoder(input_dim, num_heads, hidden_dim, num_encoder_layers, dropout_rate)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, source, target):
        encoder_output = self.encoder(source)
        decoder_output = self.decoder(target, encoder_output)
        output = self.fc(decoder_output)
        return output
    
input_dim = 64
num_heads = 8
hidden_dim = 256
num_encoder_layers = 2
num_decoder_layers = 2
transformer_model = Transformer(input_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers)

source = torch.rand(1, 10, input_dim)
target = torch.rand(1, 12, input_dim)
output = transformer_model(source, target)
print("Source shape:", source.shape)
print("Target shape:", target.shape)
print("Output shape after Transformer:", output.shape)