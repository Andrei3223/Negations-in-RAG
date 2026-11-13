"""Two-stage listwise softmax: the Plackett-Luce likelihood of own > partner > rest,
truncated after two positions."""
import torch
import torch.nn.functional as F


def plackett_luce_loss(scores, pos, partner, mask, lam):
    """
    Args:
        scores: (Q, N) query-candidate similarities, already divided by tau.
        pos: (Q,) column of each query's own document.
        partner: (Q,) column of the other document of its pair.
        mask: (Q, N) True where a column is not a candidate for that query (a
            near-copy from the same family). Never True at pos or partner.
        lam: weight of stage 2.

    Returns:
        loss, stage1.mean(), stage2.mean()

    Stage 1 is InfoNCE with the partner as a hard negative: own doc first among
    all candidates. Stage 2 removes the own doc and asks for the partner first
    among the rest, which keeps the partner above unrelated documents. For q2 the
    roles swap (pos = doc2, partner = doc1), so the loss is symmetric when every
    pair contributes both of its queries.
    """
    scores = scores.masked_fill(mask, float('-inf'))
    rows = torch.arange(scores.size(0), device=scores.device)
    stage1 = torch.logsumexp(scores, dim=1) - scores[rows, pos]
    without_pos = scores.masked_fill(F.one_hot(pos, scores.size(1)).bool(), float('-inf'))
    stage2 = torch.logsumexp(without_pos, dim=1) - scores[rows, partner]
    loss = (stage1 + lam * stage2).mean()
    return loss, stage1.mean(), stage2.mean()


def info_nce_loss(scores, pos):
    """Plain InfoNCE for queries with one labelled positive (MS MARCO): own positive
    first among all candidates. Equals plackett_luce_loss with lam=0 and no mask."""
    return F.cross_entropy(scores, pos)
