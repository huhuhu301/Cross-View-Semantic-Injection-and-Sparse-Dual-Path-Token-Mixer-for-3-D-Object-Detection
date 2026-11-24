# Modified by RV-SDTM contributors; see NOTICE and THIRD_PARTY_NOTICES.md.
import torch
import numpy as np

def limit( ang):
    ang = ang % (2 * np.pi)

    ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

    ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

    return ang


def ang_weight(pred, gt):

    a2 = torch.abs(torch.sin(pred - gt))

    return 1-a2

def compute_iou(x,w,y,l):
    zmax1 = x + w * 0.5
    zmin1 = x - w * 0.5
    zmax2 = y + l * 0.5
    zmin2 = y - l * 0.5
    z_overlap = (torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)).clamp_min(0.)
    all_lap = (torch.max(zmax1, zmax2) - torch.min(zmin1, zmin2)).clamp_min(0.)
    iou = z_overlap / all_lap
    return iou

def bb_loss(pred, target):
    iouw = compute_iou(pred[..., 0], pred[..., 3], target[..., 0], target[..., 3])
    ioul = compute_iou(pred[..., 1], pred[..., 4], target[..., 1], target[..., 4])
    iouh = compute_iou(pred[..., 2], pred[..., 5], target[..., 2], target[..., 5])

    a_p = limit(pred[..., 6])
    a_g = limit(target[..., 6])
    ioua = ang_weight(a_p, a_g)

    iou = iouw*ioul*iouh*ioua

    diff_angle = pred[:, -1] - target[:, -1]
    angle_factor = 1.25 * (1.0 - torch.abs(torch.cos(diff_angle)))

    center_dist_square = torch.pow(target[:, 0:3] - pred[:, 0:3], 2).sum(-1)

    finall_loss = 1-iou + angle_factor + center_dist_square

    return finall_loss*1.5
