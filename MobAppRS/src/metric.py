import numpy as np
"""
rank:predicted items 
ground_truth:true truth items
"""

def precision(rank, ground_truth):
    hits = [1 if item in ground_truth else 0 for item in rank]
    cum = np.cumsum(hits, dtype=np.float32)
    ar = np.arange(1, len(rank)+1)
    result = np.cumsum(hits, dtype=np.float32)/np.arange(1, len(rank)+1)
    return result


def recall(rank, ground_truth):
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32) / len(ground_truth)
    return result

def map(rank, ground_truth):
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    relevant_num = np.cumsum([1 if item in ground_truth else 0 for item in rank])
    result = [p/r_num if r_num!=0 else 0 for p, r_num in zip(sum_pre, relevant_num)]
    return result
