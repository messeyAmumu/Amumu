import torch
import clip
import argparse
from PIL import Image
from pathlib import Path
from pycocotools import coco
from datasets import build_dataset, get_coco_api_from_dataset, DistributedWeightedSampler
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from engine import evaluate, train_one_epoch
from util.box_ops import box_cxcywh_to_xyxy
import torchvision
from models.classifier import build_classifier
import copy
import os
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_args_parser():
    parser = argparse.ArgumentParser('CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching', add_help=False)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/home/ccc/CORA-region/data/coco')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--device', default='cuda:1', help='device to use for training / testing. We must use cuda.')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--label_type', default='')
    parser.add_argument('--clip_aug', action='store_true', default=False)
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--use_caption', action='store_true')
    parser.add_argument('--pseudo_threshold', default=0.2, type=float)
    parser.add_argument('--pseudo_box', default='', type=str)
    parser.add_argument('--no_overlapping_pseudo', action='store_true')
    parser.add_argument('--base_sample_prob', default=1.0, type=float)
    parser.add_argument('--repeat_factor_sampling', action='store_true')
    parser.add_argument('--repeat_threshold', default=0.001, type=float)
    parser.add_argument('--remove_base_pseudo_label', action='store_true')
    parser.add_argument('--text_len', default=25, type=int)
    parser.add_argument('--text_adapter', action='store_true')
    parser.add_argument('--text_prompt', action='store_true')
    parser.add_argument('--adapter_dim', default=256, type=int)
    parser.add_argument('--lr_language', default=0.0, type=float)
    parser.add_argument('--ovd', action='store_true', default=True)
    parser.add_argument('--no_clip_init', action='store_true')
    parser.add_argument('--no_clip_init_text', action='store_true')
    parser.add_argument('--no_clip_init_image', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--output_dir', default='/home/ccc/CORA-region/logs/clip_test', help='path where to save, empty for no saving')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--backbone', default='clip_RN50', type=str,
                        help="Name of the convolutional backbone to use")

    return parser

def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only."
    print(args)

    device = torch.device(args.device)
    classifier = build_classifier(args)

    dataset_val = build_dataset(image_set='val', args=args)
    dataset_train = build_dataset(image_set='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)
    base_ds = get_coco_api_from_dataset(dataset_val)

    
    text_features_for_classes = []

    img_root = args.coco_path + '/val2017'
    save_root = args.coco_path + '/visualize'
    
    header = 'Test:'
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=args.device)

    categories=data_loader_val.dataset.category_list

    metric_logger = utils.MetricLogger(delimiter="  ")
    for samples, targets in metric_logger.log_every(data_loader_val, 10, header):
        samples = samples.to(args.device)
        
        # torch_resize = Resize([224,224]) 
        # resized_samples = torch_resize(samples.tensors)
        # image = preprocess(Image.open("/home/ccc/cat1.jpg")).unsqueeze(0).to(device)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(args.device)
        # change to the category list

        with torch.no_grad():

            box_emb = None
            ori_boxes = [box_cxcywh_to_xyxy(target['boxes']) for target in targets]
            box_scores = [1 for box in ori_boxes]
            num_boxes = [target['boxes'].size(0) for target in targets]
            masks = samples.decompose()[1]
            sizes = [((1 - m[0].float()).sum(), (1 - m[:,0].float()).sum()) for m in masks]
            boxes = [box.clone().to(args.device) for box in ori_boxes]
            for i in range(len(boxes)):
                boxes[i][:,[0,2]] = boxes[i][:,[0,2]] * sizes[i][0]
                boxes[i][:,[1,3]] = boxes[i][:,[1,3]] * sizes[i][1]


            roi_features = torchvision.ops.roi_align(
                samples.tensors,
                boxes,
                output_size=(224, 224),
                spatial_scale=1.0,
                aligned=True)
            
            
            features = model.encode_image(roi_features).type(torch.float32)
            
            labels = torch.cat([target['labels'] for target in targets])
            text_feature = classifier(categories).to(args.device)
            
            logits = (features @ text_feature.t())
            pred_labels = logits.argmax(dim=-1)
            print(labels)
            print(pred_labels)
            # text_features = model.encode_text(text)
            
            # logits_per_image, logits_per_text = model(image, text)
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            b,c,max_h,max_w = samples.tensors.shape
            ori_image = np.zeros((b,max_h,max_w,c))
            roi_idx = 0
            scores, idxxx = logits.max(dim=1)
            for idx, target in enumerate(targets):
                save_dir = os.path.join(save_root, str(int(target['image_id'][0])))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                path = os.path.join(img_root, data_loader_val.dataset.coco.imgs[int(target['image_id'][0])]['file_name'])
                img = cv2.imread(path)
                w, h = int(sizes[idx][0]), int(sizes[idx][1])
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
                ori_image[idx][:h,:w,:] = img
                temp = copy.deepcopy(ori_image[idx])
                cv2.imwrite(os.path.join(save_dir, 'original.png'), temp)
                for i, bb in enumerate(boxes[idx]):
                    x1, y1, x2, y2 = bb
                    cv2.rectangle(temp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                    roi = ori_image[idx][int(y1):int(y2), int(x1):int(x2), :]
                    roi_h, roi_w, roi_c = roi.shape
                    if roi_h < 50 or roi_w < 50:
                        roi = cv2.resize(roi, (int(roi_h*4), int(roi_w*4)), interpolation=cv2.INTER_CUBIC)
                    roi_h, roi_w, roi_c = roi.shape
                    padded_h, padded_w = max(224, roi_h), max(224, roi_w)
                    padded_roi = np.zeros((padded_h, padded_w, c))
                    padded_roi[:roi_h, :roi_w, :] = roi
                    cv2.putText(padded_roi, 'pred: ' + '{:.3} '.format(float(scores[roi_idx])) + str(categories[idxxx[roi_idx]]), (1, padded_h-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
                    cv2.putText(padded_roi, 'gt: ' + '{:.3} '.format(float(logits[roi_idx][labels[roi_idx]])) + str(categories[labels[roi_idx]]), (1, padded_h-25), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 100), 2)
                    cv2.putText(padded_roi, 'ID: ' + str(int(target['image_id'][0])) + '_' + str(i), (1, padded_h-40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 100), 2)
                    # cv2.imwrite(os.path.join(save_dir, str(i) + '.png'), roi)
                    cv2.imwrite(os.path.join(save_dir, str(i) + '_224.png'), padded_roi)
                    roi_idx += 1
                # cv2.imwrite(os.path.join(save_dir, 'labeled.png'), temp)

        # print("Label probs:", probs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CLIP_test", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
