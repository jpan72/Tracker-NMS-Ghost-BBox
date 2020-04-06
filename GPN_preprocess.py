import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms as T
import json
from tensorboardX import SummaryWriter

import torch
from tracker.preprocess_multitracker import JDETracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from utils.datasets import JointDataset, collate_fn
from utils.utils import *
from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def write_results(filename, results, data_type, dataset):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                if dataset in ['mot15_train', 'mot15_test'] and h < 59:
                    continue
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, evaluator, writer, n_iter, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate, train=False)
    timer = Timer()
    results = []
    frame_id = 0
    plot_arguments = []
    ghost_sequence = []
    ghost_match_ious = []

    for path, img, img0 in dataloader:
        if frame_id % 200 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        if opt.ghost_stats:
            online_targets, n_iter, ghosts, ghost_match_iou = tracker.update(blob, img0, opt, evaluator, writer, n_iter, path)
            ghost_tlwhs = [g.tlwh for g in ghosts]
            ghost_sequence.append((ghost_tlwhs, frame_id))
            ghost_match_ious.extend(ghost_match_iou)
        elif opt.vis_FN:
            online_targets, n_iter, FN_tlbrs_selected = tracker.update(blob, img0, opt, evaluator, writer, n_iter, path)
        else:
            online_targets, n_iter = tracker.update(blob, img0, opt, evaluator, writer, n_iter, path)

        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # if t.ghost == 1:
            # 	print('ghost == 1')
            # 	tid = -1
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        if opt.save_debug or opt.count_fn:
            plot_arguments.append((img0, online_tlwhs, online_ids, frame_id, 1. / timer.average_time))

        if opt.vis_FN:
            plot_arguments.append((img0, online_tlwhs, online_ids, frame_id, 1. / timer.average_time, FN_tlbrs_selected))

        frame_id += 1

    # save results
    write_results(result_filename, results, data_type, opt.dataset)
    if opt.save_debug or opt.count_fn:
        return frame_id, timer.average_time, timer.calls, n_iter, plot_arguments
    elif opt.vis_FN:
        return frame_id, timer.average_time, timer.calls, n_iter, plot_arguments
    if opt.ghost_stats:
        return frame_id, timer.average_time, timer.calls, n_iter, ghost_sequence, ghost_match_ious
    return frame_id, timer.average_time, timer.calls, n_iter


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    # result_root = os.path.join(data_root, '..', 'results', exp_name)
    result_root = os.path.join(opt.result_dir, exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # Read config
    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    if opt.count_fn or opt.ghost_stats:
        fig, axs = plt.subplots(2, 4, sharey=True, tight_layout=True)

    # ------- Setup training ------- #
    # Get dataloader
    transforms = T.Compose([T.ToTensor()])

    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]
    dataset = JointDataset(dataset_root, trainset_paths, img_size, augment=True, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                             num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    writer = SummaryWriter('../exp-ghost-bbox')
    n_iter = 0

    # for seq in seqs:
    for i, seq in enumerate(seqs):

        det_area_sum = 0
        det_count = 0
        fn_areas = []

        ghost_fn_overlaps = []
        fn_closest_ghost_overlaps = []
        num_fn = 0

        # output_dir = os.path.join(opt.output_dir, exp_name, seq) if save_images or save_videos else None
        output_dir = os.path.join(opt.output_images, opt.dataset, opt.dataset)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        logger.info('\nstart seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        try:
            meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
            frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        except:
            frame_rate = 30

        evaluator = Evaluator(data_root, seq, data_type)
        # import pdb; pdb.set_trace()
        if opt.save_debug or opt.count_fn:
            nf, ta, tc, n_iter, plot_arguments = eval_seq(opt, dataloader, data_type, result_filename, evaluator, writer, n_iter,
                                                  save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        elif opt.vis_FN:
            print('here')
            nf, ta, tc, n_iter, plot_arguments = eval_seq(opt, dataloader, data_type, result_filename, evaluator, writer, n_iter,
                                                  save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
            print('after')
        elif opt.ghost_stats:
            nf, ta, tc, n_iter, ghost_sequence, ghost_match_ious = eval_seq(opt, gpn, dataloader, data_type, result_filename, evaluator, writer, n_iter,
                                                                    save_dir=output_dir, show_image=show_image,
                                                                    frame_rate=frame_rate)
        else:
            nf, ta, tc, n_iter = eval_seq(opt, dataloader, data_type, result_filename, evaluator, writer, n_iter,
                                  save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        # evaluator = Evaluator(data_root, seq, data_type)
        acc = evaluator.eval_file(result_filename)
        accs.append(acc)

        if save_videos:
            output_video_folder = osp.join(opt.output_videos, opt.dataset)
            if not osp.exists(output_video_folder):
                os.makedirs(output_video_folder)
            output_video_path = osp.join(output_video_folder, '{}.mp4'.format(seq))

            cmd_str = 'ffmpeg -y -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
            os.system("rm -R {}".format(output_dir))

        if opt.save_debug:
            debug_dir = os.path.join(opt.debug_images, opt.dataset)
            if not osp.exists(debug_dir):
                os.makedirs(debug_dir)
            for img0, online_tlwhs, online_ids, frame_id, fps in plot_arguments:
                debug_im = vis.plot_tracking_debug(img0, online_tlwhs, online_ids, acc.mot_events.loc[frame_id], seq,
                                                   evaluator,
                                                   frame_id=frame_id, fps=fps)
                cv2.imwrite(os.path.join(debug_dir, '{:05d}.jpg'.format(frame_id)), debug_im)

            debug_video_folder = osp.join(opt.debug_videos, opt.dataset)
            if not osp.exists(debug_video_folder):
                os.makedirs(debug_video_folder)
            debug_video_path = osp.join(debug_video_folder, '{}_debug.mp4'.format(seq))

            cmd_str = 'ffmpeg -y -f image2 -i {}/%05d.jpg -c:v copy {}'.format(debug_dir, debug_video_path)
            os.system(cmd_str)
            os.system("rm -R {}".format(debug_dir))

        if opt.vis_FN:

            FN_dir = os.path.join(opt.FN_images, opt.dataset)
            if not osp.exists(FN_dir):
                os.makedirs(FN_dir)

            for img0, online_tlwhs, online_ids, frame_id, fps, FN_tlbrs_selected in plot_arguments:
                try:
                    FN_im = vis.plot_FN(img0, online_tlwhs, online_ids, acc.mot_events.loc[frame_id], seq,
                                             evaluator,
                                             frame_id=frame_id, fps=fps, FN_tlbrs_selected=FN_tlbrs_selected)

                    cv2.imwrite(os.path.join(FN_dir, '{:05d}.jpg'.format(frame_id)), FN_im)

                except:
                    cv2.imwrite(os.path.join(FN_dir, '{:05d}.jpg'.format(frame_id)), img0)

            FN_video_folder = osp.join(opt.FN_videos, opt.dataset)
            if not osp.exists(FN_video_folder):
                os.makedirs(FN_video_folder)
            FN_video_path = osp.join(FN_video_folder, '{}_FN.mp4'.format(seq))

            cmd_str = 'ffmpeg -y -f image2 -i {}/%05d.jpg -c:v copy {}'.format(FN_dir, FN_video_path)
            os.system(cmd_str)
            os.system("rm -R {}".format(FN_dir))


        if opt.count_fn:
            for img0, online_tlwhs, online_ids, frame_id, fps in plot_arguments:
                try:
                    a, b, c = vis.count_fn_debug(img0, online_tlwhs, online_ids, acc.mot_events.loc[frame_id], seq,
                                                 evaluator,
                                                 frame_id=frame_id, fps=fps)
                    det_area_sum += a
                    det_count += b
                    fn_areas.extend(c)
                except:
                    continue

            det_area = float(det_area_sum) / det_count
            num_fn = len(fn_areas)
            fn_areas = np.array(fn_areas)
            fn_area = float(sum(fn_areas)) / num_fn
            fn_area_ratio = fn_areas / det_area
            print('Average detections area: {} px'.format(det_area))
            print('Average FN area: {} px. There are {} FNs in total.'.format(fn_area, num_fn))

            hist = np.histogram(fn_areas, bins=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5]) * 5000)
            hist_ratio = np.histogram(fn_area_ratio[fn_area_ratio < 3],
                                      bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5])
            print('Histogram in px: {}'.format(hist))
            print('Histogram in ratio: {}'.format(hist_ratio))

            axs[i // 4, i % 4].hist(fn_area_ratio[fn_area_ratio < 2], bins=10)
            axs[i // 4, i % 4].set_title(seq)
            axs[i // 4, i % 4].set_xlabel('FN area')
            axs[i // 4, i % 4].set_ylabel('Count')

            plt.savefig('FN_histograms.jpg')

        if opt.ghost_stats:

            for ghost_tlwhs, frame_id in ghost_sequence:
                try:
                    # print(frame_id)
                    # print('ghost_tlwhs')
                    # print(ghost_tlwhs)
                    # print()
                    ghost_fn_overlap, num_fn_i, fn_closest_ghost_overlap = vis.get_overlap(ghost_tlwhs,
                                                                                           acc.mot_events.loc[frame_id],
                                                                                           seq, evaluator,
                                                                                           frame_id=frame_id)
                    ghost_fn_overlaps.extend(ghost_fn_overlap)
                    num_fn += num_fn_i
                    fn_closest_ghost_overlaps.extend(fn_closest_ghost_overlap)
                except:
                    continue

            # axs[i//4, i%4].hist(ghost_fn_overlaps, bins=10)
            # axs[i//4, i%4].set_title(seq)
            # axs[i//4, i%4].set_xlabel('Overlap')
            # axs[i//4, i%4].set_ylabel('Count')
            # plt.ylim(0,1000)
            # plt.savefig('Ghost_FN_overlap.jpg')
            # print('Number of FN in this sequence: {}'.format(num_fn))

            # axs[i//4, i%4].hist(ghost_fn_overlaps, bins=10, density=True)
            # axs[i//4, i%4].set_title(seq)
            # axs[i//4, i%4].set_xlabel('Overlap')
            # axs[i//4, i%4].set_ylabel('Count%')
            # plt.xlim(0,1)
            # plt.savefig('Ghost_FN_overlap_percentage.jpg')
            # print('Number of FN in this sequence: {}'.format(num_fn))

            # axs[i//4, i%4].hist(fn_closest_ghost_overlaps, bins=list(np.arange(0,11)), density=True)
            # axs[i//4, i%4].set_title(seq)
            # axs[i//4, i%4].set_xlabel('Overlap')
            # axs[i//4, i%4].set_ylabel('Count%')
            # plt.xlim(0,1)
            # plt.savefig('Ghost_FN_closest_overlap.jpg')
            # print('Number of FN in this sequence: {}'.format(num_fn))

            # axs[i//4, i%4].hist(ghost_match_ious, bins=list(np.arange(0,11)), density=True)
            # axs[i//4, i%4].set_title(seq)
            # axs[i//4, i%4].set_xlabel('Overlap')
            # axs[i//4, i%4].set_ylabel('Count%')
            # # plt.xlim(0,1)
            # axs[i//4, i%4].set_xlim(0,1)
            # plt.savefig('Ghost_match_overlap.jpg')         

            axs[i // 4, i % 4].hist(fn_closest_ghost_overlaps, bins=np.linspace(0, 1, 11), alpha=0.5, label='unmatched',
                                    color='lightblue')
            axs[i // 4, i % 4].hist(ghost_match_ious, bins=np.linspace(0, 1, 11), alpha=0.5, label='matched', color='r')
            axs[i // 4, i % 4].set_title(seq)
            axs[i // 4, i % 4].set_xlabel('Overlap')
            axs[i // 4, i % 4].set_ylabel('Count')
            axs[i // 4, i % 4].set_xlim(0, 1)
            plt.legend(loc='upper right')
            plt.savefig('Ghost_stats.jpg')

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--result-dir', type=str, default='results', help='path to result files')
    parser.add_argument('--output-images', type=str, default='output_images', help='path to output path')
    parser.add_argument('--output-videos', type=str, default='output_videos', help='path to output path')
    parser.add_argument('--debug-images', type=str, default='../exp/debug_images', help='path to debug vis result path')
    parser.add_argument('--debug-videos', type=str, default='../exp/debug_videos', help='path to debug vis result path')
    parser.add_argument('--FN-images', type=str, default='../exp/FN_images', help='path to FN vis result path')
    parser.add_argument('--FN-videos', type=str, default='../exp/FN_videos', help='path to FN vis result path')
    parser.add_argument('--dataset', type=str, default='mot17_train', help='dataset to test on')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--ghost-feature-thres', type=float, default=0.6, help='feature threshold for ghost matching')
    parser.add_argument('--ghost-iou-thres', type=float, default=0.4, help='iou threshold for ghost matching')
    parser.add_argument('--ghost-occ-thres', type=float, default=0.2, help='occlusion threshold for ghost matching')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--G', type=int, default=1, help='number of additional ghost boxes for each detection')
    parser.add_argument('--save-thres', type=float, default=0.5,
                        help='number of additional ghost boxes for each detection')
    parser.add_argument('--KF-var-mult', type=float, default=1, help='multiplier of KF variance for ghost matches')
    parser.add_argument('--N', type=int, default=1, help='number of ghost track copies for each unmatched track')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--ld', type=float, default=0.5, help='lambda for regression loss')
    parser.add_argument('--lr', type=float, default=0.1, help='GPN learning rate')
    parser.add_argument('--occ-reason-thres', type=float, default=0.5, help='GPN learning rate')

    # parser.add_argument('--test-mot17', action='store_true', help='tracking buffer')
    parser.add_argument('--save-images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save-videos', action='store_true', help='save tracking results (video)')
    parser.add_argument('--save-debug', action='store_true', help='save tracking results (video)')
    parser.add_argument('--count-fn', action='store_true', help='count and categorize FN')
    parser.add_argument('--ghost', action='store_true', help='add ghost bounding boxes')
    parser.add_argument('--two-stage', action='store_true', help='include ghost bbox in the additional matching stage')
    parser.add_argument('--update-ghost-feat', action='store_true',
                        help='when matched with ghost, update track feature with ghost feature')
    parser.add_argument('--update-ghost-coords', action='store_true',
                        help='when matched with ghost, update track coords with ghost bbox')
    parser.add_argument('--feat-ghost-match', action='store_true', help='perform feature match for ghosts')
    parser.add_argument('--iou-ghost-match', action='store_true', help='perform iou match for ghosts')
    parser.add_argument('--occ-ghost-match', action='store_true', help='perform occlusion match for ghosts')
    parser.add_argument('--small-ghost', action='store_true', help='propose smaller ghost box as well')
    parser.add_argument('--ghost-stats', action='store_true',
                        help='get IoU statistics between all proposed ghosts and FNs')
    parser.add_argument('--save-lt', action='store_true',
                        help='use ghost only for reactivate lost tracks; for each lost track, see if there is a ghost where IoU > threshold')
    parser.add_argument('--ghost-track', action='store_true', help='use ghost track instead ghost detection')
    parser.add_argument('--thresholding-occ-reason', action='store_true', help='using threshold/min-distance for occlusion reasoning when generating GT ghost box')
    parser.add_argument('--use-featmap', action='store_true', help='use feature map for GRN, instead of feature vector')
    parser.add_argument('--vis-FN', action='store_true', help='visualize matched FNs and all FNs')

    opt = parser.parse_args()
    print(opt, end='\n\n')

    if opt.dataset == 'mot17_train':
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP
                    '''
        data_root = '/hdd/yongxinw/MOT17/MOT17/train'

    elif opt.dataset == 'mot15_train':
        seqs_str = '''ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Bahnhof
                      ETH-Pedcross2
                      ETH-Sunnyday
                      KITTI-13
                      KITTI-17
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      Venice-2
                    '''
        data_root = '/hdd/yongxinw/2DMOT2015/train/'


    elif opt.dataset == 'mot15_train_unique':
        seqs_str = '''ADL-Rundle-6
                      ADL-Rundle-8
                      KITTI-13
                      KITTI-17
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      Venice-2
                    '''
        data_root = '/hdd/yongxinw/2DMOT2015/train/'

    elif opt.dataset == 'mot15_train_useful':
        seqs_str = '''ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                    '''
        # seqs_str = "PETS09-S2L1"
        data_root = '/hdd/yongxinw/2DMOT2015/train/'

    elif opt.dataset == 'mot17_test':
        seqs_str = '''MOT17-01-SDP
                     MOT17-03-SDP
                     MOT17-06-SDP
                     MOT17-07-SDP
                     MOT17-08-SDP
                     MOT17-12-SDP
                     MOT17-14-SDP'''
        data_root = '/hdd/yongxinw/MOT17/MOT17/test'

    elif opt.dataset == 'mot15_test':
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1
                    '''
        data_root = '/hdd/yongxinw/2DMOT2015/test/'

    elif opt.dataset == 'mot17':
        seqs_str = '''MOT17-01-SDP
                      MOT17-02-SDP
                      MOT17-03-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-12-SDP
                      MOT17-13-SDP
                      MOT17-14-SDP
                    '''


    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         # exp_name=opt.weights.split('/')[-2],
         exp_name=opt.dataset,
         show_image=False,
         save_images=opt.save_images,
         save_videos=opt.save_videos)
