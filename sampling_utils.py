import torch


def top_k_sampling(logits, k):
    top_k = torch.topk(logits, k)
    top_k_indices = top_k.indices
    top_k_values = top_k.values
    probabilities = torch.softmax(top_k_values, dim=-1)
    choice = torch.multinomial(probabilities, num_samples=1)
    token_id = int(top_k_indices[choice])
    return token_id


def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    logits[sorted_indices] = sorted_indices_to_remove.type(logits.dtype) * -1e10
    probabilities = torch.softmax(logits, dim=-1)
    token_id = int(torch.multinomial(probabilities, num_samples=1))

    return token_id


def top_k_p_sampling(logits, k, p):
    # Apply top-k sampling
    top_k = torch.topk(logits, k)
    top_k_indices = top_k.indices
    top_k_values = top_k.values

    # Apply top-p sampling on top-k logits
    sorted_logits, sorted_indices = torch.sort(top_k_values, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    top_k_values[sorted_indices] = sorted_indices_to_remove.type(top_k_values.dtype) * -1e10
    probabilities = torch.softmax(top_k_values, dim=-1)
    choice = torch.multinomial(probabilities, num_samples=1)
    token_id = int(top_k_indices[choice])

    return token_id
