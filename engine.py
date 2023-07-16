# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import cv2
import numpy as np
import torch
import copy
import os
import util.misc as utils
from datasets.coco_eval import CocoEvaluator, convert_to_xywh
from datasets.panoptic_eval import PanopticEvaluator
from models.fast_detr import contrastive_loss
import torchvision
from util.box_ops import box_cxcywh_to_xyxy
from torch.nn.functional import cross_entropy
import matplotlib.pyplot as plt

base_list = ['person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat', 'bench', 'bird', 
             'horse', 'sheep', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 
             'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
             'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
             'remote', 'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'toothbrush']


def boxto15(bboxes):
        bboxes15 = []
        for bbox in bboxes:
            res = torch.dstack([
                        1.5 * bbox[:, 0] - 0.5 * bbox[:, 2], 
                        1.5 * bbox[:, 1] - 0.5 * bbox[:, 3],
                        1.5 * bbox[:, 2] - 0.5 * bbox[:, 0], 
                        1.5 * bbox[:, 3] - 0.5 * bbox[:, 1]
                        ]).squeeze(0)
            bboxes15.append(res)
        return bboxes15

# adapted from dab-detr
def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    loss_list = []
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v if isinstance(v, (list, dict)) else v.to(device) for k, v in t.items()} for t in targets]

        categories = data_loader.dataset.category_list

        # add pseudo labels
        pseudo_categories = list(set([a for target in targets if 'pseudo_labels' in target for a in target['pseudo_labels']]))
        for target in targets:
            if 'pseudo_labels' not in target:
                continue
            pseudo_label_ids = [pseudo_categories.index(cat) + len(categories) for cat in target['pseudo_labels']]
            target['labels'] = torch.cat([target['labels'], torch.tensor(pseudo_label_ids, device=target['labels'].device, dtype=target['labels'].dtype)])

        outputs = model(samples, categories=categories + pseudo_categories)
        
        features, text_feature, tau = outputs['features'], outputs['text_feature'], outputs['tau']
        
        if args.box_conditioned_pe:
            xywh_gt = torch.cat([target['boxes'] for target in targets])
            box_emb = gen_sineembed_for_position(xywh_gt.unsqueeze(0))[0]
            if args.only_box_size:
                box_emb = box_emb[:,256:]
        else:
            box_emb = None
        gt_boxes = [box_cxcywh_to_xyxy(target['boxes']) for target in targets]
        masks = features[0].decompose()[1]
        sizes = [((1 - m[0].float()).sum(), (1 - m[:,0].float()).sum()) for m in masks]
        for i in range(len(gt_boxes)):
            gt_boxes[i][:,[0,2]] = gt_boxes[i][:,[0,2]] * sizes[i][0]
            gt_boxes[i][:,[1,3]] = gt_boxes[i][:,[1,3]] * sizes[i][1]
        
        # gt_boxes = boxto15(gt_boxes)

        if args.roi_feat == 'layer4':
            if args.backbone == 'clip_RN50x4':
                reso = 9
            else:
                reso = 7
            roi_features = torchvision.ops.roi_align(
                features[0].tensors,
                gt_boxes,
                output_size=(reso, reso),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 7, 7)
        
            if 'module' in dir(model):
                output_feats = model.module.backbone[0].attnpool(roi_features, box_emb)
            else:
                output_feats = model.backbone[0].attnpool(roi_features, box_emb)
                
        elif args.roi_feat == 'layer3':
            if args.backbone == 'clip_RN50x4':
                reso = 18
            else:
                reso = 14
            roi_features = torchvision.ops.roi_align(
                features[0].tensors,
                gt_boxes,
                output_size=(reso, reso),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 7, 7)
            
            # roi_features = model.backbone[0].region_pertur(roi_features)
        
            if 'module' in dir(model):
                output_feats = model.module.backbone[0].layer4(roi_features)
                output_feats = model.module.backbone[0].attnpool(output_feats, box_emb)
            else:
                output_feats = model.backbone[0].layer4(roi_features)
                output_feats = model.backbone[0].attnpool(output_feats, box_emb) 
        output_feats = output_feats / output_feats.norm(dim=-1, keepdim=True)
        logits = (output_feats @ text_feature.t()) * tau
        
        labels = torch.cat([target['labels'] for target in targets])
        
        if labels.numel() == 0:
            loss_cls = logits.sum() * 0.0
        else:
            loss_cls = cross_entropy(logits, labels)
        
        loss_dict_cls = {"cls_loss": loss_cls}
            
        loss_dict = dict()
        weight_dict = dict()
        if args.use_proposal:
            class_agnostic_targets = targets.copy()
            for target in class_agnostic_targets:
                target['labels'] = target['labels'] * 0
            loss_dict = criterion(outputs, class_agnostic_targets)
            weight_dict = criterion.weight_dict
        
        loss_dict.update(loss_dict_cls)
        weight_dict.update(dict(
            cls_loss=1.0
        ))
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # loss_list.append(losses.detach().cpu().numpy())
        # plt.plot(epochs,acc, 'b', label='Training accuracy')
        # plt.plot(loss_list, 'b', label='Training loss')
        # plt.savefig(os.path.join(args.output_dir, 'test.png'))
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}.\n  Training terminated.".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del samples
        del targets
        del loss_dict
        del loss_dict_reduced
        del loss_dict_reduced_unscaled
        del losses
        del losses_reduced_scaled
        
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    if args.export:
        label_map = dict()
        coco_evaluator = None
        panoptic_evaluator = None
    else:
        if args.dataset_file == 'lvis':
            from lvis import LVISEval, LVISResults
            cat2label = data_loader.dataset.cat2label
            label2cat = {v: k for k, v in cat2label.items()}
            panoptic_evaluator = None
            coco_evaluator = None
        else:
            iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
            coco_evaluator = CocoEvaluator(base_ds, iou_types, label2cat=data_loader.dataset.label2catid)

            panoptic_evaluator = None
            if 'panoptic' in postprocessors.keys():
                panoptic_evaluator = PanopticEvaluator(
                    data_loader.dataset.ann_file,
                    data_loader.dataset.ann_folder,
                    output_dir=os.path.join(output_dir, "panoptic_eval"),
                )

    img_root = args.coco_path + '/val2017'
    save_root = os.path.join(args.coco_path, args.visual_dir)

    categories = data_loader.dataset.category_list
    print_freq = 100
    _cnt = 0
    results = []
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v if isinstance(v, (list, dict)) else v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples, categories=data_loader.dataset.category_list)
        features, text_feature, tau = outputs['features'], outputs['text_feature'], outputs['tau']
        
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        if args.eval_box_from == 'GT':
            if args.box_conditioned_pe:
                xywh_gt = torch.cat([target['boxes'] for target in targets])
                box_emb = gen_sineembed_for_position(xywh_gt.unsqueeze(0))[0]
                if args.only_box_size:
                    box_emb = box_emb[:,256:]
            else:
                box_emb = None

            ori_boxes = [box_cxcywh_to_xyxy(target['boxes']) for target in targets]
            box_scores = [1 for box in ori_boxes]
            num_boxes = [target['boxes'].size(0) for target in targets]
        elif args.eval_box_from == 'proposal':
            ori_boxes = [box_cxcywh_to_xyxy(box) for box in outputs['pred_boxes']]
            box_scores = [logit.sigmoid() for logit in outputs['pred_logits']]
            num_boxes = [box.size(0) for box in ori_boxes]
            
        masks = features[0].decompose()[1]
        sizes = [((1 - m[0].float()).sum(), (1 - m[:,0].float()).sum()) for m in masks]
        boxes = [box.clone() for box in ori_boxes]
        for i in range(len(boxes)):
            boxes[i][:,[0,2]] = boxes[i][:,[0,2]] * sizes[i][0]
            boxes[i][:,[1,3]] = boxes[i][:,[1,3]] * sizes[i][1]

        # boxes = boxto15(boxes)
        
        if args.roi_feat == 'layer4':
            if args.backbone == 'clip_RN50x4':
                reso = 9
            else:
                reso = 7
            roi_features = torchvision.ops.roi_align(
                features[0].tensors,
                boxes,
                output_size=(reso, reso),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 7, 7)
        
            if 'module' in dir(model):
                output_feats = model.module.backbone[0].attnpool(roi_features, box_emb)
            else:
                output_feats = model.backbone[0].attnpool(roi_features, box_emb)
                
        elif args.roi_feat == 'layer3':
            if args.backbone == 'clip_RN50x4':
                reso = 18
            else:
                reso = 14
            roi_features = torchvision.ops.roi_align(
                features[0].tensors,
                boxes,
                output_size=(reso, reso),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 7, 7)

            
            # roi_features = model.backbone[0].region_pertur(roi_features)
        
            if 'module' in dir(model):
                output_feats = model.module.backbone[0].layer4(roi_features)
                output_feats = model.module.backbone[0].attnpool(output_feats, box_emb)
            else:
                output_feats = model.backbone[0].layer4(roi_features)
                output_feats = model.backbone[0].attnpool(output_feats, box_emb)
        output_feats = output_feats / output_feats.norm(dim=-1, keepdim=True)
        logits = (output_feats @ text_feature.t()) * tau
        pred_labels = logits.argmax(dim=-1)
        
        labels = torch.cat([target['labels'] for target in targets])


        # visualization
        a = torch.flatten(roi_features, start_dim=2)
        b = output_feats.unsqueeze(2).repeat(1,1,reso**2)
        bb = torch.ones_like(b)
        bb *= 1./1024
        c = torch.sum(bb*a, dim=1).reshape(b.shape[0], reso, reso)
        d = c.detach().cpu().numpy()
        e = cv2.resize(d.transpose(1,2,0), (224, 224), interpolation=cv2.INTER_CUBIC)
        if len(e.shape) == 2:
            e = np.expand_dims(e, axis=2)
        # print(e.shape)
        e = e.transpose(2, 0, 1)
        heatmap = np.zeros((b.shape[0],224,224,3))
        for i in range(e.shape[0]):
            e[i] = (e[i] - e[i].min()) / (e[i].max() - e[i].min())
            e[i] *= 255
            a = e[i].astype('uint8')  
            heatmap[i] = cv2.applyColorMap(a, cv2.COLORMAP_JET)
            heatmap[i][np.where(e[i] <= 125)] = 0

        ori_masks = samples.decompose()[1]
        ori_sizes = [((1 - m[0].float()).sum(), (1 - m[:,0].float()).sum()) for m in ori_masks]
        new_boxes = [box.clone().to(args.device) for box in ori_boxes]
        for i in range(len(boxes)):
            new_boxes[i][:,[0,2]] = new_boxes[i][:,[0,2]] * ori_sizes[i][0]
            new_boxes[i][:,[1,3]] = new_boxes[i][:,[1,3]] * ori_sizes[i][1]
        c = 3
        b,max_h,max_w = ori_masks.shape
        ori_image = np.zeros((b,max_h,max_w,c))
        roi_idx = 0
        scores, idxxx = logits.max(dim=1)
        for idx, target in enumerate(targets):
            # save_dir = os.path.join(save_root, str(int(target['image_id'][0])))
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            path = os.path.join(img_root, data_loader.dataset.coco.imgs[int(target['image_id'][0])]['file_name'])
            img = cv2.imread(path)
            w, h = int(ori_sizes[idx][0]), int(ori_sizes[idx][1])
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            ori_image[idx][:h,:w,:] = img
            temp = copy.deepcopy(ori_image[idx])
            # cv2.imwrite(os.path.join(save_dir, 'original.png'), temp)
            for i, bb in enumerate(new_boxes[idx]):
                x1, y1, x2, y2 = bb
                cv2.rectangle(temp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                roi = ori_image[idx][int(y1):int(y2), int(x1):int(x2), :]

                ### 在224的尺寸级联一下
                # roi_224 = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_CUBIC)
                # padded_roi = cv2.addWeighted(src1=roi_224, alpha=0.8, src2=heatmap[roi_idx], beta=0.3, gamma=0)
                # padded_roi = np.concatenate((padded_roi,roi_224), axis=1)
                # padded_h, padded_w, c = roi_224.shape
                ###

                # padded version
                roi_h, roi_w, roi_c = roi.shape
                if roi_h < 50 or roi_w < 50:
                    roi = cv2.resize(roi, (int(roi_w*4), int(roi_h*4)), interpolation=cv2.INTER_CUBIC)
                roi_h, roi_w, roi_c = roi.shape
                
                resized_heatmap = cv2.resize(heatmap[roi_idx], (int(roi_w), int(roi_h)), interpolation=cv2.INTER_CUBIC)
                weighted_roi = cv2.addWeighted(src1=roi, alpha=0.8, src2=resized_heatmap, beta=0.3, gamma=0)

                padded_h, padded_w = max(224, roi_h), max(224, roi_w)
                padded_roi = np.zeros((padded_h, padded_w, c))
                padded_heatmap = np.zeros((padded_h, padded_w, c))
                padded_roi[:roi_h, :roi_w, :] = roi
                padded_heatmap[:roi_h, :roi_w, :] = weighted_roi
                res = np.concatenate((padded_roi,padded_heatmap), axis=1)

                if categories[labels[roi_idx]] in base_list:
                    save_dir = os.path.join(save_root, 'base')
                else:
                    save_dir = os.path.join(save_root, 'novel')
                    
                if idxxx[roi_idx] != labels[roi_idx]:
                    save_dir = os.path.join(save_dir, 'wrong')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                elif float(scores[roi_idx]) > 20.0:
                    save_dir = os.path.join(save_dir, 'high')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                else:
                    save_dir = os.path.join(save_dir, 'normal')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)


                pred_path = 'pred: ' + '{:.3} '.format(float(scores[roi_idx])) + str(categories[idxxx[roi_idx]])
                if categories[idxxx[roi_idx]] in base_list:
                    pred_path = pred_path + ' (base)'
                else:
                    pred_path = pred_path + ' (novel)'
                cv2.putText(res, pred_path, (1, padded_h-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
                cv2.putText(res, 'gt: ' + '{:.3} '.format(float(logits[roi_idx][labels[roi_idx]])) + str(categories[labels[roi_idx]]), (1, padded_h-25), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 100), 2)
                cv2.putText(res, 'ID: ' + str(int(target['image_id'][0])) + '_' + str(i), (1, padded_h-40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 100), 2)
                # cv2.imwrite(os.path.join(save_dir, str(i) + '.png'), roi)
                cv2.imwrite(os.path.join(save_dir, str(int(target['image_id'][0])) + '_' + str(i) + '_padded_cam_group_cls6.png'), res)
                # cv2.imwrite(os.path.join(save_dir, str(i) + '_attention_prompt.png'), padded_roi)
                roi_idx += 1
            # cv2.imwrite(os.path.join(save_dir, str(int(target['image_id'][0])) + '_' + 'labeled.png'), temp)

        if args.export:
            pred_labels = logits.argmax(dim=-1)
            box_ids = torch.cat([target['box_ids'] for target in targets])
            for id, label in zip(box_ids, pred_labels):
                label_map[id.item()] = data_loader.dataset.label2catid[label.item()]
        
        # loss = cross_entropy(logits, labels)
        
        # loss_dict = {"ce_loss": loss}

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = loss_dict_reduced
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        if args.dataset_file == 'coco':
            results = []
        logits = logits.softmax(dim=-1)
        if args.dataset_file == 'lvis':
            for logit, box, scale, box_score, target in zip(logits.split(num_boxes), ori_boxes, scale_fct, box_scores, targets):
                logit = logit * box_score
                scores, indices = logit.flatten().topk(k=min(300, logit.numel()))
                box_id = torch.div(indices, logit.size(1), rounding_mode='floor')
                cls_id = indices % logit.size(1)
                pred_boxes = box[box_id]
                image_id = target['image_id'].item()
                out_boxes = pred_boxes * scale[None]
                out_boxes = convert_to_xywh(out_boxes)
                
                for ind in range(len(scores)):
                    temp = {
                        "image_id": image_id,
                        "score": scores[ind].item(),
                        "category_id": cls_id[ind].item(),
                        "bbox": out_boxes[ind].tolist(),
                    }
                    if args.label_map:
                        temp["category_id"] = label2cat[temp["category_id"]]

                    results.append(temp)
        else:
            for logit, box, scale, box_score in zip(logits.split(num_boxes), ori_boxes, scale_fct, box_scores):
                logit = logit * box_score
                scores, indices = logit.flatten().topk(k=min(100, logit.numel()))
                box_id = torch.div(indices, logit.size(1), rounding_mode='floor')
                cls_id = indices % logit.size(1)
                pred_boxes = box[box_id]
                results.append(dict(
                    scores=scores,
                    labels=cls_id,
                    boxes=pred_boxes * scale[None],
                ))
        
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
            panoptic_evaluator.update(res_pano)
            
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.export:
        import json
        with open(f'logs/export_label_{utils.get_rank()}.json', 'w') as f:
            json.dump(label_map, f)
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    
    if args.dataset_file == 'lvis':
        rank = utils.get_rank()
        torch.save(results, output_dir + f"/pred_{rank}.pth")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if rank == 0:
            world_size = utils.get_world_size()
            for i in range(1, world_size):
                temp = torch.load(output_dir + f"/pred_{i}.pth")
                results += temp


        lvis_results = LVISResults(base_ds, results, max_dets=300)
        lvis_eval = LVISEval(base_ds, lvis_results, "bbox")
        lvis_eval.run()
        lvis_eval.print_results()
    
    del samples
    del targets
    # del loss_dict
    # del loss_dict_reduced
    # del loss_dict_reduced_unscaled

    torch.cuda.empty_cache()

    return stats, coco_evaluator


