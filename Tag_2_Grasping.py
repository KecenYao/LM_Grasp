import argparse
import os
import glob

import numpy as np
import json
import torch
import torchvision
from PIL import Image

from Tag_2_Masks import Command2Mask
from run_GRConvNet import run_GRConvNet 
from whisper_command import speech_recognition


def Command_2_Grasping(command, args, image_path, depth_path):
    masked_rgbs, masked_depths = Command2Mask(command, args, image_path, depth_path)
    q_imgs, ang_imgs, width_imgs = [], [], []
    id = 0
    for masked_rgb, masked_depth in zip(masked_rgbs, masked_depths):
        q_img, ang_img, width_img = run_GRConvNet(masked_rgb, masked_depth, id=id)
        q_imgs.append(q_img)
        ang_imgs.append(ang_img)
        masked_depths.append(masked_depth)
        id += 1
    
    return q_imgs, ang_imgs, width_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tag_to_Mask Demo", add_help=True)
    
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--input_depth", type=str, required=True, help="path to depth file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--use_audio", type=bool, default=False, help="use audio form input command")
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
    image_paths = glob.glob(os.path.join(image_dir, f"*{image_extension}"))
    depth_paths = glob.glob(os.path.join(depth_dir, f"*{depth_extension}"))

    # Sort the file lists to ensure corresponding images are paired correctly
    image_paths.sort()
    depth_paths.sort()

    speech_file = "scissor.mp3"
    if args.use_audio:
        command, language = speech_recognition(speech_file)
        print("Audio language: {}".format(language))
    else:
        command = f"""Grab the rhinoceros from the table"""
    print("\ncommand: {} \n".format(command))

    for image_path, depth_path in zip(image_paths, depth_paths):
        Command_2_Grasping(command, args, image_path, depth_path)
        break