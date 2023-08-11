import os

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


def run_GRConvNet(rgb, depth, id=0):
    # Load image
    logging.info('Loading image...')
    #pic = Image.open(args.rgb_path, 'r')
    rgb = np.array(rgb)
    #pic = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(depth), axis=2)

    # Load Network
    network = "GRConvNet/RoboticGrasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98" # set default network path
    logging.info('Loading model...')
    net = torch.load(network)
    logging.info('Done')

    # Get the compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_depth, use_rgb = True, True  # set default as using both depth and rgb image
    img_data = CameraData(include_depth=use_depth, include_rgb=use_rgb)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        save = True # save the files by default
        if save:
            try:
                os.makedirs(os.path.join('results', str(id)))
            except:
                pass
                
            save_results(
                rgb_img=img_data.get_rgb(rgb, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=1,
                grasp_width_img=width_img,
                save_path=os.path.join('results', str(id))
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            plot_results(fig=fig,
                         rgb_img=img_data.get_rgb(rgb, False),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=1,
                         grasp_width_img=width_img)
            fig.savefig('img_result.pdf')

    return q_img, ang_img, width_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run grasping network")
    parser.add_argument('--rgb_path', type=str, default='GRConvNet/RoboticGrasping/cornell/08/pcd0845r.png',
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='GRConvNet/RoboticGrasping/cornell/08/pcd0845d.tiff',
                        help='Depth Image path')
    args = parser.parse_args()

    rgb = Image.open(args.rgb_path, 'r')
    depth = Image.open(args.depth_path, 'r')

    run_GRConvNet(rgb, depth)