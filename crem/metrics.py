"""Multi-label evaluation metrics, faithful ports of the files under
CREM/evaluate/ (Ranking_loss.m, coverage.m, One_error.m, MacroAUC.m,
Average_precision.m). All functions take Q x N `Outputs` and Q x N
`test_target` (+/-1), exactly like the MATLAB originals."""
import numpy as np


def _valid_instances(Outputs, test_target):
    """Keep instances that are neither all-positive nor all-negative."""
    num_class = Outputs.shape[0]
    s = test_target.sum(axis=0)
    keep = (s != num_class) & (s != -num_class)
    return Outputs[:, keep], test_target[:, keep]


def ranking_loss(Outputs, test_target):
    """Port of Ranking_loss.m (ties count as errors: pos <= neg)."""
    Outputs, test_target = _valid_instances(Outputs, test_target)
    num_class, num_instance = Outputs.shape
    if num_instance == 0:
        return float("nan")
    rankloss = 0.0
    for i in range(num_instance):
        pos = test_target[:, i] == 1
        neg = ~pos
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos == 0 or n_neg == 0:
            continue
        # pairs where score(pos) <= score(neg)
        temp = np.sum(Outputs[pos, i][:, None] <= Outputs[neg, i][None, :])
        rankloss += temp / (n_pos * n_neg)
    return rankloss / num_instance


def coverage(Outputs, test_target):
    """Port of coverage.m."""
    Outputs, test_target = _valid_instances(Outputs, test_target)
    num_class, num_instance = Outputs.shape
    if num_instance == 0:
        return float("nan")
    cover = 0.0
    for i in range(num_instance):
        # stable ascending sort, like MATLAB's sort
        order = np.argsort(Outputs[:, i], kind="stable")
        rank = np.empty(num_class, dtype=int)
        rank[order] = np.arange(num_class)  # rank[j] = position of j (0=lowest)
        pos = np.where(test_target[:, i] == 1)[0]
        temp_min = rank[pos].min() + 1      # MATLAB loc, 1-based
        cover += num_class - temp_min + 1
    return ((cover / num_instance) - 1.0) / num_class


def one_error(Outputs, test_target):
    """Port of One_error.m."""
    Outputs, test_target = _valid_instances(Outputs, test_target)
    num_class, num_instance = Outputs.shape
    if num_instance == 0:
        return float("nan")
    oneerr = 0
    for i in range(num_instance):
        maximum = Outputs[:, i].max()
        tied = np.where(Outputs[:, i] == maximum)[0]
        if not np.any(test_target[tied, i] == 1):
            oneerr += 1
    return oneerr / num_instance


def macro_auc(Outputs, test_target):
    """Port of MacroAUC.m (per-class AUC, 0.5 credit for ties; classes with
    no positive or no negative instance are excluded from the average)."""
    test_target = np.where(test_target == -1, 0, test_target)
    num_class, num_instance = Outputs.shape
    auc = np.zeros(num_class)
    count_valid_label = 0
    for i in range(num_class):
        pos = test_target[i, :] == 1
        n_p, n_n = pos.sum(), num_instance - pos.sum()
        if n_p == 0 or n_n == 0:
            count_valid_label += 1
        else:
            p_out = Outputs[i, pos]
            n_out = Outputs[i, ~pos]
            gt = np.sum(p_out[:, None] > n_out[None, :])
            eq = np.sum(p_out[:, None] == n_out[None, :])
            auc[i] = (gt + 0.5 * eq) / (n_p * n_n)
    return auc.sum() / (num_class - count_valid_label)


def average_precision(Outputs, test_target):
    """Port of Average_precision.m."""
    Outputs, test_target = _valid_instances(Outputs, test_target)
    num_class, num_instance = Outputs.shape
    if num_instance == 0:
        return float("nan")
    aveprec = 0.0
    for i in range(num_instance):
        order = np.argsort(Outputs[:, i], kind="stable")
        rank = np.empty(num_class, dtype=int)
        rank[order] = np.arange(num_class)
        pos = np.where(test_target[:, i] == 1)[0]
        indicator = np.zeros(num_class)
        indicator[rank[pos]] = 1
        summary = 0.0
        for j in pos:
            loc = rank[j] + 1  # MATLAB 1-based position in ascending order
            summary += indicator[loc - 1:].sum() / (num_class - loc + 1)
        aveprec += summary / len(pos)
    return aveprec / num_instance
