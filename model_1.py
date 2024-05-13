import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 1e-3
batch_size = 64
max_seq_len = 256  # needs to be max story length from batch or max sequence length
num_heads = 8
temperature = 1
#d_model = 64  #
embed_size = 128
d_ff = embed_size  * 4    # 4 times model
dropout = 0.07
num_layers = 8



class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)


        self.layers = nn.Sequential(*(self.TransformerBlock(embed_size, num_heads, d_ff, dropout) for _ in range(num_layers)))
        self.norm = nn.LayerNorm(self.embed_size) # B*T*E
        self.linear = nn.Linear(self.embed_size, self.vocab_size)
        self.to(device)
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(device)
        #Änderung mit Batches
        x = self.embedding(x)  # [batch_size, seq_len, embed_size] B*T*E
        x = self.pos_encoding(x)  # Add positional encoding
        un_norm = self.linear(self.norm(self.layers(x)))
        #probs = nn.functional.softmax(un_norm, dim=-1).to(device)
        return un_norm

    @torch.no_grad()
    def generate_tokens(self, start_token: Tensor | int, length: int) -> list:

        self.eval()
        x = torch.as_tensor([start_token], dtype=torch.int64).unsqueeze(0).to(
            device)  # Ensure x is [1, 1] if start_token is int
        token_list = [start_token]  # Assuming start_token is already an integer here!

        for _ in range(length):
            logits = self(x)  # Generate logits
            if logits.dim() == 3 and logits.size(1) == 1:
                logits = logits.squeeze(1)  # Adjust shape from [1, 1, vocab_size] to [1, vocab_size]
                probs = F.softmax(logits / temperature, dim=-1)  # Apply softmax to convert logits to probabilities
                pred = torch.multinomial(probs, 1)  # Sample from the probability distribution
                pred_item = pred.item()  # Convert tensor to integer
                #if pred_item == eos_token_id: #ToDO integrate EOS token
                   # break
                token_list.append(pred_item)  # Add generated token to the list
                x = torch.tensor([[pred_item]], dtype=torch.int64).to(device)  # Prepare input for next generation step

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
            return x


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
                return result

            class SingleHeadAttention(nn.Module):
                def __init__(self, embed_dim, att_dim):
                    super().__init__()
                    self.keys = nn.Linear(embed_dim, att_dim, bias=False)
                    self.queries = nn.Linear(embed_dim, att_dim, bias=False)
                    self.values = nn.Linear(embed_dim, att_dim, bias=False)

                def forward(self, embed: Tensor) -> Tensor:
                    #print(f"embed size: {embed.size()}") # B*T*E,
                    k = self.keys(embed)
                    #print(f"Keys size: {k.size()}") # B*T*A, A = E/number_heads
                    #print(f"Attention Dimension: {k.size(-1)}")
                    q = self.queries(embed)
                    v = self.values(embed)

                    batch_size, max_seq_len, _ = embed.size()

                    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
                    #print(f"Scores size: {k.size()}") # B*T*A, A = E/number_heads
                    mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).to(device)  # T*T #ToDo Get T to match max_idx from get_batch()
                    mask = mask == 0
                    #print(f"Mask size: {mask.size()}")  # T*T
                    scores = scores.masked_fill(mask, float('-inf')) # B*T*T
                    #print(f"Scores size after mask: {mask.size()}")  # B*T*A, A = E/number_heads
                    attention = F.softmax(scores, dim=-1)
                    output = attention @ v #torch.matmul(attention, v)
                    return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, dropout: float = 0.01):
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

    # Create input tensor correctly
    input_tensor = torch.tensor([vocab["once"]], dtype=torch.int64).unsqueeze(0).to(device)
    # Pass the single integer value to generate_tokens
    tl = model.generate_tokens(input_tensor[0][0].item(), 32)
    for val in tl:
        print(vocab_rev[val], end=" ")