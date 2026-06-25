from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

def tensor_text_to_video_metrics(sim_tensor, top_k = [1,5,10]):
    """Compute text-to-video retrieval metrics from [text, video_group, video_slot]."""
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    sim_tensor = sim_tensor.clone()
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    group_level_sim = torch.max(sim_tensor, dim=-1).values
    return compute_metrics(group_level_sim.detach().cpu().numpy())

def tensor_video_to_text_sim(sim_tensor):
    """Collapse [text, video_group, video_slot] into [text, video_group]."""
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    sim_tensor = sim_tensor.clone()
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    return torch.max(sim_tensor, dim=-1).values

def tensor_video_to_text_metrics(sim_tensor):
    """Compute video-to-text retrieval metrics from [text, video_group, video_slot]."""
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    sim_tensor = sim_tensor.clone()
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')

    # Each valid grouped video instance retrieves its matching text over the last dimension.
    stacked_sim_matrices = sim_tensor.permute(2, 1, 0)
    first_argsort = torch.argsort(stacked_sim_matrices, dim=-1, descending=True)
    second_argsort = torch.argsort(first_argsort, dim=-1, descending=False)
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1=1, dim2=2))

    diagonal_scores = torch.flatten(torch.diagonal(sim_tensor, dim1=0, dim2=1))
    mask = ~torch.logical_or(torch.isinf(diagonal_scores), torch.isnan(diagonal_scores))
    valid_ranks = ranks[mask]
    if not torch.is_tensor(valid_ranks):
      valid_ranks = torch.tensor(valid_ranks)

    results = {}
    results['R1'] = float(torch.sum(valid_ranks < 1) * 100 / len(valid_ranks))
    results['R5'] = float(torch.sum(valid_ranks < 5) * 100 / len(valid_ranks))
    results['R10'] = float(torch.sum(valid_ranks < 10) * 100 / len(valid_ranks))
    results['MR'] = float(torch.median(valid_ranks + 1))
    results["MedianR"] = results['MR']
    results["MeanR"] = float(np.mean(valid_ranks.detach().cpu().numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.detach().cpu().numpy() + 1))
    return results
