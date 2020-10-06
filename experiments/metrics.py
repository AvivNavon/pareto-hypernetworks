import itertools
from functools import reduce
from pymoo.factory import get_performance_indicator

import torch
import numpy as np


def non_uniformity(losses, ray):
    m = len(losses)
    rl = losses * ray
    normalized_rl = rl / rl.sum()
    non_unif = (normalized_rl * torch.log(normalized_rl * m)).sum()

    return non_unif


def get_area(rectangles):
    """source: https://leetcode.com/problems/rectangle-area-ii/solution/"""
    def intersect(rec1, rec2):
        return [max(rec1[0], rec2[0]),
                max(rec1[1], rec2[1]),
                min(rec1[2], rec2[2]),
                min(rec1[3], rec2[3])]

    def area(rec):
        dx = max(0, rec[2] - rec[0])
        dy = max(0, rec[3] - rec[1])
        return dx * dy

    total_area = 0
    for size in range(1, len(rectangles) + 1):
        for group in itertools.combinations(rectangles, size):
            total_area += (-1) ** (size + 1) * area(reduce(intersect, group))

    return total_area


def cover_metric(losses):
    """Calculate the union area of rectangles of the form [0, 0, 1 / loss1, 1 / loss2]

    :param losses: iterable of (loss1, loss2)
    :return:
    """
    assert len(losses[0]) == 2, "only support 2D problems for now"

    # recs[i] = [x1, y1, x2, y2]
    recs = []
    for loss in losses:
        recs.append([0, 0, 1 / loss[0], 1 / loss[1]])
    return get_area(recs)


def calc_hypervolume(losses, ref_point=None):
    """https://www.pymoo.org/misc/performance_indicator.html#Hypervolume

    :param losses: iterator with elements (loss1, loss2)
    :param ref_point: reference point
    :return:
    """
    # zero_ref_point = np.zeros((len(losses[0]), ))
    # hv = get_performance_indicator("hv", ref_point=zero_ref_point)
    # points = -1 / (losses + 1e-12)

    points = np.array(losses)
    if ref_point is None:
        ref_point = np.array([5.] * len(losses[0]))
    hv = get_performance_indicator("hv", ref_point=ref_point)

    return hv.calc(points)


def find_dominant_points(costs, maximise=False):
    if not isinstance(costs, np.ndarray):
        costs = np.array(costs)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)
    return is_efficient


def find_min_dist_to_ray(points, ray=None):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if ray is None:
        ray = np.array([0.5, 0.5]) * 1e6
    origin = np.zeros(2)
    dist = []
    for p in points:
        dist.append(np.linalg.norm(np.cross(ray - origin, origin - p)) / np.linalg.norm(ray - origin))
    dist = np.array(dist)
    return np.where(dist == dist.min())[0]  # not using argmin in case we have more than one pt with min dist
