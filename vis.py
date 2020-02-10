# Created by yongxinwang at 2020-02-09 22:55
# Created by yongxinwang at 2019-09-16 19:07
import os
import numpy as np
import cv2
import argparse
from PIL import Image

from utils.utils import random_colors, visualize_boxes


def visualize(image_dir, result_path, **kwargs):
    vid = kwargs.pop("video_name", "")
    # tracks = np.loadtxt(result_path)
    save_dir = os.path.join(os.path.dirname(result_path), "images")
    os.makedirs(save_dir, exist_ok=True)

    video_name = "{}-{}".format(os.path.basename(result_path).replace(".txt", ""))
    tracks = np.loadtxt(result_path, delimiter=',')
    max_color = 20
    colors = random_colors(max_color, bright=True)
    total_frames = len(os.listdir(image_dir))

    for i, frame in enumerate(range(1, total_frames + 1)):
        curr_tracks = tracks[(tracks[:, 0] == frame)]
        boxes = curr_tracks[:, 2:6]
        # image = cv2.imread(os.path.join(image_dir, "%06d.jpg" % frame))
        # import ipdb
        # ipdb.set_trace()
        # image = load_image_single_detection(os.path.join(image_dir, "%06d.jpg" % frame), square_size=513)
        image = Image.open(os.path.join(image_dir, "%06d.jpg" % frame))
        for j, box in enumerate(boxes):
            id = curr_tracks[j, 1]
            color = tuple([int(tmp * 255) for tmp in colors[int(id % max_color)]])

            image = visualize_boxes(image, [box], texts=[str(id)], width=3, outline=color)
            # cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color=color, thickness=3)
            # cv2.putText(image, str(id), (int(x0), int(y0-10)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        image.save(os.path.join(save_dir, "track-%06d.jpg" % frame))
        # cv2.imwrite(os.path.join(exp_dir, "images", "track-%06d.jpg" % frame), image)

    os.system('ffmpeg -framerate 15 -i {}/track-%06d.jpg -c:v libx264 '
              '-profile:v high -crf 20 -pix_fmt yuv420p -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" {}/{}-det.avi'.format(save_dir, save_dir, video_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument parser for visualizing tracking reseults")
    parser.add_argument("--video_name", type=str, help="saving video name")
    args = parser.parse_args()

    image_dir = "/hdd/yongxinw/MOT17/MOT17/train/MOT17-09-SDP/img1/"
    result_path = "./results/TowardsRTT/MOT17-09-SDP.txt"
    visualize(image_dir, result_path, video_name=args.video_name)
