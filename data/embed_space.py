import torch
from torch import nn
from io_utils import load_vocabulary

if __name__ == '__main__':
    model = torch.load(f'../trained_models/384_6_6_1536.pth').to("cpu")

    vocab = load_vocabulary("../vocabulary.pkl")
    embed = model.embedding
    cos = nn.CosineSimilarity()
    vocab_rev = {k: v for v, k in vocab.items()}

    he = embed(torch.tensor([vocab["he"]]))
    she = embed(torch.tensor([vocab["she"]]))
    mom = embed(torch.tensor([vocab["mom"]]))
    dad = embed(torch.tensor([vocab["dad"]]))
    timmy = embed(torch.tensor([vocab["timmy"]]))
    bear = embed(torch.tensor([vocab["bear"]]))

    similarities = torch.empty(len(vocab), dtype=torch.float)
    approx = bear
    print(embed.weight.data.shape)
    print("Closest tokens in embedding space to 'bear'")
    for i, row in enumerate(embed.weight.data):
        similarities[i] = cos(approx, row)

    sorted = torch.argsort(similarities, descending=True)
    for val in sorted[:10]:
        print(vocab_rev[val.item()], round(similarities[val.item()].item(), 5))
