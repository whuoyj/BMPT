"""
Loss functions for BDC-CLIP training with knowledge distillation.
Transplanted from ViFi-CLIP and adapted for multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(pos_logits, neg_logits, margin=0.1):
    """
    Compute margin-based contrastive loss between positive and negative class logits.

    This encourages the model to:
    1. Assign high probability to positive (correct) classes
    2. Assign low probability to negative (confusing) classes
    3. Maintain a margin between positive and negative predictions

    Args:
        pos_logits (torch.Tensor): Softmax probabilities for positive classes, shape [B]
        neg_logits (torch.Tensor): Softmax probabilities for negative classes, shape [B]
        margin (float): Margin value to separate positive from negative classes (default: 0.1)

    Returns:
        torch.Tensor: Scalar loss value

    Example:
        >>> pos_logits = torch.tensor([0.8, 0.9, 0.7])  # High probability for correct class
        >>> neg_logits = torch.tensor([0.2, 0.1, 0.3])  # Low probability for wrong class
        >>> loss = compute_loss(pos_logits, neg_logits, margin=0.1)
    """
    # The loss encourages: pos_logits > neg_logits + margin
    # If pos_logits - neg_logits > margin, loss = 0
    # Otherwise, loss = margin - (pos_logits - neg_logits)
    loss = torch.clamp(margin - (pos_logits - neg_logits), min=0.0)
    return loss.mean()


def domain_alignment_loss(source_features, target_features, epoch, alpha=1.0, warmup_epochs=5):
    """
    Compute domain alignment loss to align features from different domains.

    This loss minimizes the distribution discrepancy between source domain features
    (e.g., original videos) and target domain features (e.g., augmented/novel videos).
    Uses a warmup strategy to gradually increase the alignment strength.

    Args:
        source_features (torch.Tensor): Features from source domain, shape [B, D]
        target_features (torch.Tensor): Features from target domain, shape [B, D]
        epoch (int): Current training epoch (for warmup scheduling)
        alpha (float): Weight coefficient for domain alignment loss (default: 1.0)
        warmup_epochs (int): Number of epochs for linear warmup (default: 5)

    Returns:
        torch.Tensor: Scalar domain alignment loss

    Example:
        >>> source_feat = torch.randn(32, 512)
        >>> target_feat = torch.randn(32, 512)
        >>> loss = domain_alignment_loss(source_feat, target_feat, epoch=3, alpha=1.0)
    """
    # Normalize features to unit sphere
    source_features = F.normalize(source_features, p=2, dim=-1)
    target_features = F.normalize(target_features, p=2, dim=-1)

    # Compute L2 distance between corresponding features
    distance = torch.norm(source_features - target_features, p=2, dim=-1)

    # Warmup: linearly increase the loss weight from 0 to alpha
    if epoch < warmup_epochs:
        warmup_factor = epoch / warmup_epochs
    else:
        warmup_factor = 1.0

    # Mean distance loss with warmup
    loss = alpha * warmup_factor * distance.mean()
    return loss


def knowledge_alignment_loss(video_features, knowledge_features, temperature=0.07):
    """
    Compute knowledge alignment loss using contrastive learning.

    This loss encourages video features to align with their corresponding knowledge features
    while being discriminative against other samples in the batch.

    Args:
        video_features (torch.Tensor): Video features from visual encoder, shape [B, D]
        knowledge_features (torch.Tensor): Knowledge features from external source, shape [B, D]
        temperature (float): Temperature parameter for contrastive loss (default: 0.07)

    Returns:
        torch.Tensor: Scalar knowledge alignment loss
    """
    # Normalize features
    video_features = F.normalize(video_features, p=2, dim=-1)
    knowledge_features = F.normalize(knowledge_features, p=2, dim=-1)

    # Compute similarity matrix: [B, B]
    similarity = video_features @ knowledge_features.t() / temperature

    # Positive pairs are on the diagonal
    batch_size = video_features.size(0)
    labels = torch.arange(batch_size, device=video_features.device)

    # Cross-entropy loss with diagonal as positive pairs
    loss = F.cross_entropy(similarity, labels)
    return loss


def multi_task_loss(
    logits_cos_vl,
    logits_cls,
    logits_bdc_vl,
    logits_k_t,
    labels,
    criterion,
    weight_cos_vl=1.0,
    weight_cls=1.0,
    weight_bdc_vl=1.0,
    weight_k_t=1.0
):
    """
    Compute multi-task loss for BDC-CLIP with knowledge distillation.

    This function combines losses from four different learning objectives:
    1. Backbone vision-language alignment (logits_cos_vl)
    2. BDC visual classification (logits_cls)
    3. BDC vision-language alignment (logits_bdc_vl)
    4. Knowledge-text alignment (logits_k_t)

    Args:
        logits_cos_vl (torch.Tensor): Cosine similarity logits for V-L alignment, shape [B, num_classes]
        logits_cls (torch.Tensor): Classification logits from BDC visual features, shape [B, num_classes]
        logits_bdc_vl (torch.Tensor): BDC V-L alignment logits, shape [B, num_classes]
        logits_k_t (torch.Tensor): Knowledge-text alignment logits, shape [B, num_classes]
        labels (torch.Tensor): Ground truth labels, shape [B]
        criterion: Loss function (e.g., CrossEntropyLoss)
        weight_cos_vl (float): Weight for backbone V-L loss (default: 1.0)
        weight_cls (float): Weight for visual classification loss (default: 1.0)
        weight_bdc_vl (float): Weight for BDC V-L loss (default: 1.0)
        weight_k_t (float): Weight for knowledge-text loss (default: 1.0)

    Returns:
        tuple: (total_loss, loss_dict) where loss_dict contains individual losses
    """
    # Compute individual losses
    loss_cos_vl = criterion(logits_cos_vl, labels)
    loss_cls = criterion(logits_cls, labels)
    loss_bdc_vl = criterion(logits_bdc_vl, labels)
    loss_k_t = criterion(logits_k_t, labels)

    # Weighted sum of all losses
    total_loss = (
        weight_cos_vl * loss_cos_vl +
        weight_cls * loss_cls +
        weight_bdc_vl * loss_bdc_vl +
        weight_k_t * loss_k_t
    )

    # Return both total loss and individual losses for logging
    loss_dict = {
        'loss_cos_vl': loss_cos_vl.item(),
        'loss_cls': loss_cls.item(),
        'loss_bdc_vl': loss_bdc_vl.item(),
        'loss_k_t': loss_k_t.item(),
        'total_loss': total_loss.item()
    }

    return total_loss, loss_dict