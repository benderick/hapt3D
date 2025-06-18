"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
from torch.autograd import Variable


class IoULovaszLoss(torch.nn.Module):
  def __init__(self, variance, embeddings_only, fg_classes):
    super().__init__()
    self.variance = variance
    self.embeddings_only = embeddings_only
    self.fg_classes = fg_classes
    
  def forward(self, points, target, sem_target, embeddings, voxel_resolution=1):
    """ Compute the IoU loss based on embeddings predicted by the network.

    Args:
        target (torch.Tensor): Ground truth instance annotations ... of shape [B x H x W]
        embeddings (torch.Tensor): Embeddings predicted by the network ... of shape [B x 2 x H x W]
        foreground (Optional[torch.Tensor], optional): Mask of pixels belonging to foreground  ... of shape [B x H x W].  Defaults to None.

    Returns:
        torch.Tensor: Value of loss function
    """
    batch_size = len(target)

    iou_losses = []
    for batch_idx in range(batch_size):

      for sem_class in self.fg_classes:
        gt = target[batch_idx] # [H x W]
        sem_mask = torch.zeros_like(gt.squeeze())
        sem_mask[sem_target[batch_idx].squeeze() == sem_class] = 1
        # here we can decide whether to use pts + offs or only offs
        if self.embeddings_only:
          emb = embeddings.features_at(batch_idx)
        else:
          emb = embeddings.features_at(batch_idx) + points.coordinates_at(batch_idx) * voxel_resolution # [2 x H x W]

        for instance_id in torch.unique(gt):

          sem_instance = torch.Tensor(sem_target[batch_idx])[gt == instance_id][0]
          if sem_class != sem_instance.item():
             continue
          
          mask = (gt == instance_id) # [H x W]
          instance_center = emb[mask[:, 0]].mean(dim=0, keepdim=True) # [2 x 1]


          delta = instance_center - emb
          dist = torch.norm(delta, dim=1, p=2.0) # type: ignore - [M]
          soft_mask = torch.exp(- (dist ** 2) / (2 * self.variance))

          soft_mask = soft_mask * sem_mask

          iou_loss = lovasz_softmax_flat(soft_mask.unsqueeze(1), mask.float().squeeze())
          iou_losses.append(iou_loss)
        
    if len(iou_losses):
        res = torch.stack(iou_losses).mean()
    else:
        res = torch.tensor(0.0).cuda()
    return res

  
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.

    class_pred = probas[:, 0]
    errors = (Variable(labels) - class_pred).abs()
    errors_sorted, perm = torch.sort(errors, 0, descending=True)
    perm = perm.data
    fg_sorted = labels[perm]
    loss = torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted)))
    return loss