import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 3e-4 #1e-3
batch_size = 64
max_seq_len = 64  # needs to be max story length from batch or max sequence length
num_heads = 8
temperature = 1
#d_model = 64  #
embed_size = 64
d_ff = 4 * embed_size    # 4 times model
dropout = 0.01
num_layers = 1



class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        #self.pos_encoding = nn.Embedding(max_seq_len, embed_size)

        #Pytorch Transformer
        #encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead=8).to(device)
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)

        self.layers = nn.Sequential(*(self.TransformerBlock(embed_size, num_heads, d_ff, dropout) for _ in range(num_layers)))
        #for i in range(num_layers):
            #self.layers.append((self.TransformerBlock(embed_size, num_heads, d_ff, dropout)))
        #nn.ModuleList([self.TransformerBlock(embed_size, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_size) # B*T*E
        self.linear = nn.Linear(self.embed_size, self.vocab_size)
        self.to(device)
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(device)
        #Änderung mit Batches
        x = self.embedding(x)  # [batch_size, seq_len, embed_size] B*T*E
        x = self.pos_encoding(x)  # Add positional encoding
        #B,T,E = x.shape   #shape for generate function ?
        #pos_embeds = self.pos_encoding(torch.arange(T).to(device)).to(device)
        #x = x + pos_embeds
        #x = x.to(device)
        #for layer in self.layers:
            #x = layer(x)
        #x = self.norm(x)  # Normalize
        #return self.linear(x)
        un_norm = self.linear(self.norm(self.layers(x)))
        probs = nn.functional.softmax(un_norm, dim =-1).to(device)
        return torch.round(probs)

        #Pytorch Transformer
        #embedding: Tensor = self.embedding(x).to(device)
        #embedding = self.pos_encoding(embedding).to(device)
        #src_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(device)
        #embedding = self.encoder(embedding, mask=src_mask, is_causal=True)


        #embedding: Tensor = self.embedding(x).to(device)
        #embedding = self.pos_encoding(embedding).to(device)
        #return self.linear(embedding).to(device)

    @torch.no_grad()
    def generate_tokens(self, start_token: Tensor | int, length: int) -> list:
        self.eval()
        x = start_token.to(device)
        token_list = [x]
        for _ in range(length):
            logits = self(x)  # Directly use logits to adjust temperature
            probs = F.softmax(logits / temperature, dim=-1).to(device)  # Apply temperature
            pred = torch.multinomial(probs, 1)[0]
            token_list.append(pred)
            x = pred
        return token_list

    class TransformerBlock(nn.Module):
        def __init__(self, mod_dim, num_heads, d_ff, dropout=0.01):
            super().__init__()
            self.attention = self.MultiHeadAttention(mod_dim, mod_dim, num_heads)
            self.norm1 = nn.LayerNorm(mod_dim)
            self.norm2 = nn.LayerNorm(mod_dim)
            #self.ff = self.VanillaNeuralNetwork(mod_dim)
            self.ffn = PositionwiseFeedforward(mod_dim, d_ff, dropout)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor):
            # Apply attention
            attn_output = self.attention(x)
            # Add & norm
            x = self.norm1(x + self.dropout(attn_output))
            # Apply feedforward network
            #ff_output = self.ff(x)
            ffn_output = self.ffn(x)
            # Another add & norm
            #x = self.norm2(x + self.dropout(ff_output))
            x = self.norm2(x + self.dropout(ffn_output))
            return torch.round(x, decimals=4)


        class MultiHeadAttention(nn.Module):
            def __init__(self, embed_dim:int, att_dim: int, num_heads: int):
                super().__init__()
                assert att_dim % num_heads == 0, "att_dim must be divisible by num_heads"
                self.num_heads = num_heads
                self.att_dim = att_dim // num_heads

                self.heads = nn.ModuleList([self.SingleHeadAttention(embed_dim, self.att_dim) for _ in range(num_heads)])
                self.linear = nn.Linear(num_heads * self.att_dim, embed_dim)
                self.norm = nn.LayerNorm(embed_dim)  # Add layer normalization

            def forward(self, embed: Tensor) -> Tensor:
                head_outputs = [head(embed) for head in self.heads] # each element B*T*headsize (att //num_heads)
                concat = torch.cat(head_outputs, dim=-1) # Concatenate to B*T*A
                concat = self.norm(concat)  # Normalize
                result = self.linear(concat)
                return torch.round(result)

            class SingleHeadAttention(nn.Module):
                def __init__(self, embed_dim, att_dim):
                    super().__init__()
                    self.keys = nn.Linear(embed_dim, att_dim, bias=False)
                    self.queries = nn.Linear(embed_dim, att_dim, bias=False)
                    self.values = nn.Linear(embed_dim, att_dim, bias=False)

                def forward(self, embed: Tensor) -> Tensor:
                    #print(f"embed size: {embed.size()}") # B*T*E
                    k = self.keys(embed)
                    #print(f"Keys size: {k.size()}") # B*T*A  A = E/number_heads
                    q = self.queries(embed)
                    v = self.values(embed)

                    #scores = (q @ torch.transpose(k, 1, 2))
                    #B, T, A = k.shape
                    #scores = scores / (A**0.5) # scaled by att_dim
                    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
                    mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).to(device)  # T*T #ToDo Get T to match max_idx from get_batch()
                    mask = mask == 0
                    scores = scores.masked_fill(mask, float('-inf')) # B*T*T
                    attention = F.softmax(scores, dim=-1)
                    output = attention @ v #torch.matmul(attention, v)
                    return torch.round(output, decimals=4)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, dropout: float = 0.07):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encodings once in the right shape.
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        pos_encoding = torch.zeros(max_seq_len, embed_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        # Scale the positional encodings
        #self.scale = math.sqrt(embed_size)

        # Register the positional encodings as a buffer correctly
        self.register_buffer('pe', pos_encoding)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor, shape [batch_size, seq_len, embed_size]
        """
        # Use the registered buffer, ensuring we only apply as many positions as the current x's sequence length
        x = x + self.pe[:x.size(1)] #* self.scale
        return self.dropout(x)

        # Code vorher
        #pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(dim=1)
        #inv_denominator = 10000**(-1/embed_size*torch.arange(0, embed_size, 2, dtype=torch.float))
        #pe_term = pos * inv_denominator
        #self.pos_encoding[:, 0::2] = torch.sin(pe_term)
        #self.pos_encoding[:, 1::2] = torch.cos(pe_term)

    #def forward(self, x: Tensor) -> Tensor:
        #return self.dropout(x + self.pos_encoding[:x.size(0)]).to(device)

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.01):
        super(PositionwiseFeedforward, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(d_model, d_ff)
        # Second fully connected layer that outputs the d_model dimensions
        self.fc2 = nn.Linear(d_ff, d_model)
        # Dropout and activation function
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    from io_utils import load_vocabulary

    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}
    try:
        #model: TransformerModel = torch.load('trained_models/model.pth').to(device)
        model: TransformerModel = torch.load('trained_models/model2.pth').to(device)
    except FileNotFoundError:
        model = TransformerModel(len(vocab))

    input_tensor = torch.tensor(vocab["once"], dtype=torch.int64).unsqueeze(0)
    tl = model.generate_tokens(input_tensor, 60)
    for val in tl:
        print(vocab_rev[val.item()], end=" ")
