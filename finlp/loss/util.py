import torch.nn.functional as F

def cross_entropy(logits, targets, tag2id):
    """ calc loss
        logits: [batch_size, seq_len, output_size]
        targets: [batch_size, seq_len]
        
    """
    PAD = tag2id.get('<pad>')
    assert PAD is not None

    mask = (targets != PAD)

    targets = targets[mask]
    output_size = logits.size(2)
    logits = logits.masked_select(mask.unsqueeze(2).expand(-1, -1, output_size)).contiguous().view(-1, output_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss