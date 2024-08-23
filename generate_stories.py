import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model_1 import num_special_non_eos_tokens, device


@torch.no_grad()
def generate_tokens(model: nn.Module, token_tensor: Tensor, length: int = 250, temperature: float = 1.0,
                    eos_token: int = None) -> Tensor:
    model.eval()
    for _ in range(len(token_tensor[0]), length + 1):
        output = model(token_tensor)[:, -1, :-num_special_non_eos_tokens]
        if abs(temperature) < 1e-10:
            pred = torch.argmax(output, dim=1).unsqueeze(0)
        else:
            probs = F.softmax(output * (1 / temperature), dim=-1)
            pred = torch.multinomial(probs, 1)
        token_tensor = torch.cat((token_tensor, pred), 1)
        if pred.item() == eos_token:
            return token_tensor
    return token_tensor


@torch.no_grad()
def generate_tokens_beam(model: nn.Module, input_tensor: Tensor, beam_width: int, length: int = 250,
                         temperature: float = 1.0, eos_token: int = None) -> Tensor:
    model.eval()
    sequences = [(input_tensor.squeeze(0).tolist(), 0.0)]  # List of sequences with their scores
    completed_sequences = []

    for _ in range(length):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == eos_token or len(seq) >= length:  # Cap sequence length
                completed_sequences.append((seq, score))
                continue

            input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            output = model(input_tensor)[:, -1, :-num_special_non_eos_tokens]

            if abs(temperature) < 1e-10:
                # Temperature = 0, select the max logit deterministically
                logits = output
                pred = torch.argmax(logits, dim=-1).item()
                log_prob = F.log_softmax(logits, dim=-1).squeeze(0)[pred].item()
                candidate = (seq + [pred], score + log_prob)
                all_candidates.append(candidate)
            else:
                logits = output / temperature
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_k_log_probs, top_k_tokens = torch.topk(log_probs, beam_width)
                for j in range(beam_width):
                    candidate = (seq + [top_k_tokens[j].item()], score + top_k_log_probs[j])
                    all_candidates.append(candidate)

        if len(all_candidates) == 0:
            break

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

        if all(seq[-1] == eos_token or len(seq) >= length for seq, _ in sequences):
            break

    best_sequence = max(completed_sequences + sequences, key=lambda tup: tup[1])[0]
    return torch.tensor(best_sequence).unsqueeze(0)


@torch.no_grad()
def generate_tokens_beam_multinomial(model: nn.Module, input_tensor: Tensor, beam_width: int, length: int = 250,
                                     temperature: float = 1.0, eos_token: int = None, top_k: int = 50) -> Tensor:
    model.eval()
    sequences = [[(input_tensor.squeeze(0).tolist(), 0.0)]]
    completed_sequences = []

    for step in range(length):
        all_candidates = []

        for beam in sequences:
            for seq, score in beam:
                if seq[-1] == eos_token or len(seq) >= length:  # Cap sequence length
                    completed_sequences.append((seq, score))
                    continue

                input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
                output = model(input_tensor)[:, -1, :-num_special_non_eos_tokens]

                if abs(temperature) < 1e-10:
                    logits = output
                    pred = torch.argmax(logits, dim=-1).item()
                    log_prob = F.log_softmax(logits, dim=-1).squeeze(0)[pred].item()
                    candidate = (seq + [pred], score + log_prob)
                else:
                    logits = output / temperature
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                    top_k_log_probs, top_k_tokens = torch.topk(log_probs, top_k)
                    top_k_probs = F.softmax(top_k_log_probs, dim=-1)
                    pred_idx = torch.multinomial(top_k_probs, 1).item()
                    pred = top_k_tokens[pred_idx].item()
                    candidate = (seq + [pred], score + top_k_log_probs[pred_idx].item())

                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = [ordered[i:i + beam_width] for i in range(0, len(ordered), beam_width)]

        if all(seq[-1] == eos_token or len(seq) >= length for beam in sequences for seq, _ in beam):  # Cap sequence length
            break

    best_sequence = max(completed_sequences, key=lambda tup: tup[1])[0] if completed_sequences else sequences[0][0][0]
    return torch.tensor(best_sequence).unsqueeze(0)


if __name__ == '__main__':
    from io_utils import prompt_model

    method = ''  # Choose the generation method: default, beam, beam_multinomial
    string = ''
    story = prompt_model("transformer_model", string, 255, 0.7, method, beam_width=5)
    print(story)