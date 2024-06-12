"""  
RNN reference model 
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model_1 import num_special_non_eos_tokens

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate_rnn = 8e-4
batch_size_rnn = 32
max_seq_len_rnn = 100
embed_size_rnn = 256
hidden_size_rnn = 128
num_layers_rnn = 2
dropout_rnn = 0.2
dropout_rnn_2 = 0.2


class RNNModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 256, num_layers: int = 2,
                 dropout_rnn: float = 0.1, dropout_2: float = 0.1, padding_idx: int | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout_rnn)
        self.dropout = nn.Dropout(dropout_2)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        if h is None:
            if x.dim() == 2:  # Batched input
                h = self.init_hidden(x.size(0))
            else:  # Unbatched input
                h = torch.zeros(self.num_layers, self.hidden_size, device=next(self.parameters()).device)
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        out = self.dropout(out)
        out = self.linear(out)
        return out, h

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=next(self.parameters()).device,
                           requires_grad=False)


if __name__ == "__main__":
    from eval import calculate_rouge_scores, get_stories, max_rouge_score
    from io_utils import prompt_model

    prompt = "there"
    length = 100
    num_stories = 1
    
    story = prompt_model("RNN_256_2_128_run_1_32_0.0008_16473.354659267", prompt, length)
    print(story)
    print()
    print(max_rouge_score(story, prompt))

    # stories = [prompt_model("rnn_model", prompt, length) for _ in range(num_stories)]
    # for story in stories:
    #     print(story)
    #     print()
    
    # actual_stories = get_stories(prompt, 10000)

    # print(len(stories))
    # print(len(actual_stories))

    # scores = calculate_rouge_scores(stories, actual_stories[:num_stories])
    # print(scores)
    # print(f"ROUGE-1: {scores['rouge-1']}")
    # print(f"ROUGE-2: {scores['rouge-2']}")
    # print(f"ROUGE-L: {scores['rouge-l']}")
    # print(f"ROUGE-L: {scores['rouge-l']}")

