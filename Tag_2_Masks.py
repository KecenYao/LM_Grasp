import argparse
import os
import glob

import numpy as np
import json
import torch
import torchvision
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
import sys
sys.path.append('Tag2Text')
from Tag2Text.models import tag2text
from Tag2Text import inference_ram
import torchvision.transforms as TS

# Prompting use GPT
from action_prompt import *

# GRConvNet
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from GRConvNet.RoboticGrasping.hardware.device import get_device
from GRConvNet.RoboticGrasping.inference.post_process import post_process_output
from GRConvNet.RoboticGrasping.utils.data.camera_data import CameraData
from GRConvNet.RoboticGrasping.utils.visualisation.plot import plot_results, save_results

logging.basicConfig(level=logging.INFO)


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)


def mask_on_tag(masks, image, depth_img, rgb_id, depth_id, output_dir):
    color = np.array([255, 255, 255]) #use white as the background mask color
    h, w = masks[0].shape[-2:]

    # generate per object masked image
    masked_depths, masked_imgs = [], []
    i = 0
    for mask in masks:
        # initialize background
        mask_env = np.ones((h, w, 1))
        depth_mask = np.zeros((h, w, 1))

        mask = mask.cpu().numpy()
        mask = mask.reshape(h, w, 1)
        depth_mask = np.logical_or(depth_mask, mask)
        mask_env = ~mask * mask_env

        # generate masked depth
        #depth_mask = depth_mask.astype(bool)
        masked_depth = depth_mask.reshape(h, w) * depth_img
        masked_depth = masked_depth.astype(np.uint8)
        masked_depths.append(masked_depth)
        cv2.imwrite(os.path.join(output_dir, f"depth/masked_{depth_id}_{i}.png"), masked_depth)

        # generate masked rgb image
        img = np.array(image) * depth_mask
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(os.path.join(output_dir, "test.jpg"), img)
        mask_env = mask_env * color.reshape(1, 1, -1)
        masked_img = img + mask_env
        cv2.imwrite(os.path.join(output_dir, f"rgb/masked_{rgb_id}_{i}.png"), masked_img)
        masked_img = masked_img.astype(np.uint8)
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB) # convert rgb channel back for matplotlib imshow display
        masked_imgs.append(masked_img)

        i += 1
    
    return masked_imgs, masked_depths


def Command2Mask(command, args, image_path, depth_path):
    # cfg
    #config_file = args.config  # change the path of the model config file
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    #ram_checkpoint = args.ram_checkpoint  # change the path of the model
    ram_checkpoint = "./Tag2Text/ram_swin_large_14m.pth"
    #grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    #sam_checkpoint = args.sam_checkpoint
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    split = args.split
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device

    # Acquire actions and targets from commands
    actions_and_tags = instruction_to_action(command)
    tags = ''
    for action_and_tag in actions_and_tags:
        if action_and_tag[0] == 'Grasp':    #check for grab action
            tags = action_and_tag[1]
    print(tags)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    depth_img = Image.open(depth_path)
    depth_img = np.array(depth_img)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))


    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, tags, box_threshold, text_threshold, device=device
    )


    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
        
    rgb_id = os.path.splitext(os.path.basename(image_path))[0]
    depth_id = os.path.splitext(os.path.basename(image_path))[0]
    masked_rgbs, masked_depths = mask_on_tag(masks, image, depth_img, rgb_id, depth_id, output_dir)

    return masked_rgbs, masked_depths
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Tag_to_Mask Demo", add_help=True)
    """
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    """
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--input_depth", type=str, required=True, help="path to depth file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    #parser.add_argument("--tags", default=None, type=str, required=True, help="tag for the object to mask out")
    #parser.add_argument("--command", default=None, type=str, required=True, help="commanmd for the robot")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    image_dir = args.input_image
    depth_dir = args.input_depth

    #make output folder
    output_dir = args.output_dir
    try:
        os.makedirs(os.path.join(output_dir, 'rgb'))
        os.makedirs(os.path.join(output_dir, 'depth'))
    except:
        pass

    # Define the file extensions for RGB and depth images
    image_extension = ".png"
    depth_extension = ".png"

    # Get the list of RGB and depth image files in the input directory
    image_files = glob.glob(os.path.join(image_dir, f"*{image_extension}"))
    depth_files = glob.glob(os.path.join(depth_dir, f"*{depth_extension}"))

    # Sort the file lists to ensure corresponding images are paired correctly
    image_files.sort()
    depth_files.sort()

    command = f"""Grab the shampoo from the table"""
    print("\ncommand: {} \n".format(command))

    for image_file, depth_file in zip(image_files, depth_files):
        #print(image_file, depth_file, id)
        Command2Mask(command, args, image_file, depth_file) # run the main function
        break
        