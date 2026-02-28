# pl_model/forward_fn.py
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_cam_1d


def get_forward_func(mil_model):
    """
    Returns a common forward function based on the model name.
    """
    if mil_model in ["meanpooling", "maxpooling", "ABMIL", "GABMIL"]:
        return general_forward
    elif mil_model == "DSMIL":
        return dsmil_forward
    elif mil_model in ["CLAM-SB", "CLAM-MB"]:
        return clam_forward
    elif mil_model in ["TransMIL", "Transformer", "MambaMIL", "RRTMIL", "ILRA"]:
        return transmil_forward
    elif mil_model == "WiKG":
        return wikg_forward
    elif mil_model == "DTFD-MIL":
        return None
    else:
        raise NotImplementedError(f"Unknown MIL model: {mil_model}")


# -----------------------------
# Common forward functions
# -----------------------------

def general_forward(data, classifier, loss_func, num_classes, label=None):
    pred = classifier(data)
    if label is None:
        return pred
    loss = loss_func(pred, label)
    pred_prob = F.softmax(pred, dim=1)
    return pred, loss, pred_prob


def dsmil_forward(data, classifier, loss_func, num_classes, label=None):
    ins_prediction, bag_prediction, _, _ = classifier(data)
    max_prediction, _ = torch.max(ins_prediction, 0)

    loss = None
    if label is not None:
        # Check label shape
        if label.ndim == 1:
            lbl = label
        else:
            lbl = torch.argmax(label, dim=1)

        bag_loss = loss_func(bag_prediction, lbl)
        max_loss = loss_func(max_prediction.unsqueeze(0), lbl)
        loss = 0.5 * bag_loss + 0.5 * max_loss

    Y_prob = torch.softmax(bag_prediction, dim=1)
    return bag_prediction, loss, Y_prob


def clam_forward(data, classifier, loss_func, num_classes, label=None):
    logits, Y_prob, _, _, instance_dict = classifier(
        data, label=label, instance_eval=True
    )
    loss = None
    if label is not None:
        loss = loss_func(logits, label)
        if "instance_loss" in instance_dict:
            instance_loss = instance_dict["instance_loss"]
            loss = classifier.bag_weight * loss + (1 - classifier.bag_weight) * instance_loss
            
    return logits, loss, Y_prob


def transmil_forward(data, classifier, loss_func, num_classes, label=None):
    # TransMIL, Transformer, etc. return (logits, prob, Y_hat)
    res = classifier(data)
    if isinstance(res, tuple):
        logits = res[0]
    else:
        logits = res

    loss = None
    if label is not None:
        loss = loss_func(logits, label)
        
    Y_prob = torch.softmax(logits, dim=1)
    return logits, loss, Y_prob


def wikg_forward(data, classifier, loss_func, num_classes, label=None):
    # 🔥 [Modified] Handle WiKG return values (logits, y_prob, y_hat)
    res = classifier(data)
    if isinstance(res, tuple):
        logits = res[0]
    else:
        logits = res

    loss = None
    if label is not None:
        loss = loss_func(logits, label)
    
    Y_prob = torch.softmax(logits, dim=1)
    return logits, loss, Y_prob


# -----------------------------
# DTFD-MIL 2-tier forward parts
# -----------------------------

def dtfdmil_forward_1st_tier(
    args,
    data,
    classifier,
    attention,
    dimReduction,
    loss_func0,
    label=None,
    get_attention=False,
):
    """
    DTFD-MIL 1st tier.
    """
    instance_per_group = args.total_instance // args.numGroup

    slide_pseudo_feat, slide_sub_preds, slide_sub_labels = [], [], []
    patch_indices, tAA_all = [], []

    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), args.numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

    for tindex in index_chunk_list:
        if label is not None:
            slide_sub_labels.append(label)
            
        subFeat_tensor = torch.index_select(
            data, dim=0, index=torch.LongTensor(tindex).to(data.device)
        )

        # Embedding & attention
        tmidFeat = dimReduction(subFeat_tensor)
        
        # Tuple Unpacking: (logits, attention_score)
        _, tAA = attention(tmidFeat) 
        tAA = tAA.squeeze(0)  # [N]
        
        tAA_all.append(tAA)
        patch_indices.append(tindex)

        # Weighted sum of attention
        tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)

        # Bag-level feature
        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  # [1, D]
        tPredict = classifier(tattFeat_tensor)
        slide_sub_preds.append(tPredict)

        # CAM-based top-k / bottom-k patch selection
        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)
        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)
        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)

        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
        sort_idx = sort_idx.flatten()

        topk_idx_max = sort_idx[:instance_per_group].long()
        topk_idx_min = sort_idx[-instance_per_group:].long()
        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

        MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
        max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
        af_inst_feat = tattFeat_tensor

        if args.distill == "MaxMinS":
            slide_pseudo_feat.append(MaxMin_inst_feat)
        elif args.distill == "MaxS":
            slide_pseudo_feat.append(max_inst_feat)
        elif args.distill == "AFS":
            slide_pseudo_feat.append(af_inst_feat)

    slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
    
    loss = None
    if label is not None:
        slide_sub_preds = torch.cat(slide_sub_preds)
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)
        if slide_sub_labels.ndim == 1:
             slide_sub_labels = slide_sub_labels.long()
        loss = loss_func0(slide_sub_preds, slide_sub_labels).mean()

    if get_attention:
        tAA_tensor = torch.cat(tAA_all, dim=0) if tAA_all else None
        return loss, slide_pseudo_feat, tAA_tensor, patch_indices
        
    return loss, slide_pseudo_feat


def dtfdmil_forward_2nd_tier(
    slide_pseudo_feat,
    UClassifier,
    loss_func1,
    label=None,
):
    """
    DTFD-MIL 2nd tier.
    """
    gSlidePred = UClassifier(slide_pseudo_feat)
    
    loss = None
    if label is not None:
        n_pseudo = slide_pseudo_feat.size(0)
        if label.ndim == 1:
             bag_labels = label.repeat(n_pseudo)
        else:
             bag_labels = label.repeat(n_pseudo, 1)
             
        loss = loss_func1(gSlidePred, bag_labels).mean()
        
    return loss, gSlidePred


# -----------------------------
# Attention (visualization) API
# -----------------------------

def get_attention_func(mil_model):
    if mil_model in ["ABMIL", "GABMIL"]:
        return general_attention_func
    elif mil_model in ["TransMIL", "Transformer", "ILRA"]:
        return transmil_attention_func
    elif mil_model in ["CLAM-SB", "CLAM-MB"]:
        return clam_attention_func
    elif mil_model == "DSMIL":
        return dsmil_attention_func
    elif mil_model == "WiKG":
        return wikg_attention_func
    elif mil_model == "RRTMIL":
        return rrt_attention_func
    else:
        return None 


def general_attention_func(data, classifier, loss_func, num_classes, label=None):
    res = classifier.get_attention_maps(data)
    pred, attention_map = res[0], res[1]

    loss = None
    if label is not None:
        loss = loss_func(pred, label)
    
    pred_prob = F.softmax(pred, dim=1)
    return pred, loss, pred_prob, attention_map


def transmil_attention_func(data, classifier, loss_func, num_classes, label=None):
    out = classifier.get_attention_maps(data)
    logits = out[0]
    attn = out[-1]

    loss = None
    if label is not None:
        loss = loss_func(logits, label)
        
    Y_prob = torch.softmax(logits, dim=1)
    return logits, loss, Y_prob, attn


def clam_attention_func(data, classifier, loss_func, num_classes, label=None):
    logits, Y_prob, Y_hat, attn, instance_dict = classifier(
        data, label=label, instance_eval=True
    )
    loss = None
    if label is not None:
        loss = loss_func(logits, label)
        if "instance_loss" in instance_dict:
            loss = classifier.bag_weight * loss + (1 - classifier.bag_weight) * instance_dict["instance_loss"]
            
    return logits, loss, Y_prob, attn


def dsmil_attention_func(data, classifier, loss_func, num_classes, label=None):
    ins_prediction, bag_prediction, attn, _ = classifier(data)
    
    loss = None
    if label is not None:
        max_prediction, _ = torch.max(ins_prediction, 0)
        lbl = label if label.ndim == 1 else torch.argmax(label, dim=1)
        bag_loss = loss_func(bag_prediction, lbl)
        max_loss = loss_func(max_prediction.unsqueeze(0), lbl)
        loss = 0.5 * bag_loss + 0.5 * max_loss

    Y_prob = torch.softmax(bag_prediction, dim=1)
    pred_label = torch.argmax(bag_prediction, dim=1)
    attn = attn[:, pred_label].squeeze(1)
    
    return bag_prediction, loss, Y_prob, attn


def wikg_attention_func(data, classifier, loss_func, num_classes, label=None):
    out = classifier.get_attention_maps(data)
    logits, attn = out[0], out[-1]
    
    loss = None
    if label is not None:
        loss = loss_func(logits, label)
        
    Y_prob = torch.softmax(logits, dim=1)
    return logits, loss, Y_prob, attn


def rrt_attention_func(*args, **kwargs):
    return None, None, None, None