"""
Functions for generating tokens using a trained model, which are then combined into a story.
For prompting to a model, see section below line 155.
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model_1 import num_special_non_eos_tokens, device


@torch.no_grad()
def generate_tokens(model: nn.Module, token_tensor: Tensor, length: int = 250, temperature: float = 1.0,
                    eos_token: int = None) -> Tensor:
    """
    Generate a sequence of tokens using a given model.

    :param model: The model used for token generation.
    :param token_tensor: The initial tensor containing input tokens.
    :param length: The maximum length of the generated sequence.
    :param temperature: The temperature parameter for sampling.
    :param eos_token: The token representing the end of the sequence.

    :return: token_tensor (Tensor): The tensor containing the generated sequence of tokens.
    """
    model.eval()
    for _ in range(len(token_tensor[0]), length + 1):
        output = model(token_tensor)[:, -1, :-num_special_non_eos_tokens]
        if abs(temperature) < 1e-10:
            pred = torch.argmax(output, dim=1).unsqueeze(0)
        else:
            probs = F.softmax(output * (1 / temperature), dim=-1)
            pred = torch.multinomial(probs, 1)
        if pred.item() == eos_token:
            return token_tensor
        token_tensor = torch.cat((token_tensor, pred), 1)
    return token_tensor


@torch.no_grad()
def generate_tokens_beam(model: nn.Module, input_tensor: Tensor, beam_width: int, length: int = 250,
                         temperature: float = 1.0, eos_token: int = None) -> Tensor:
    """
    Generate a sequence of tokens using beam search with a given model.

    :param model: The model used for token generation.
    :param input_tensor: The initial tensor containing input tokens.
    :param beam_width: The number of beams for beam search.
    :param length: The maximum length of the generated sequence.
    :param temperature: The temperature parameter for sampling.
    :param eos_token: The token representing the end of the sequence.

    :return: best_sequence (Tensor): The tensor containing the best sequence of tokens generated.
    """
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
    """
    Generate a sequence of tokens using beam search with multinomial sampling.

    :param model: The model used for token generation.
    :param input_tensor: The initial tensor containing input tokens.
    :param beam_width: The number of beams for beam search.
    :param length: The maximum length of the generated sequence.
    :param temperature: The temperature parameter for sampling.
    :param eos_token: The token representing the end of the sequence.
    :param top_k: The number of top tokens considered for sampling.

    :return: best_sequence (Tensor): The tensor containing the best sequence of tokens generated.
    """
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

        if all(seq[-1] == eos_token or len(seq) >= length for beam in sequences for seq, _ in beam):
            break

    best_sequence = max(completed_sequences, key=lambda tup: tup[1])[0] if completed_sequences else sequences[0][0][0]
    return torch.tensor(best_sequence).unsqueeze(0)


if __name__ == '__main__':
    from io_utils import prompt_model

    model_name = "transformer_3.7M"  # Name of the model, must be located in trained_models/
    method = 'default'  # Choose the generation method: default, beam, beam_multinomial
    start_string = ''  # Choose start string (an empty string generates a new story from the very beginning)
    story = prompt_model(model_name=model_name, start_str=start_string, length=255, temperature=0.0,
                         method=method, beam_width=5)
    print(story)
