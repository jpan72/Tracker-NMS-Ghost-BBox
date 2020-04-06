import numpy as np
import cv2
from utils.evaluation import Evaluator
from utils.io import unzip_objs
from tracker import matching

def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    if idx == -3:
        print('!!!!!!!!')
        color = (0,0,0)

    return color


def debug_color(mot_type): # B G R
    color_dict = {'MATCH': (0, 255, 0), # green
                'MISS': (0, 255, 255),  # yellow
                'FP': (0, 0, 255),      # red
                'SWITCH': (255, 0, 255)} # purple

    if mot_type not in color_dict:
        # print(mot_type)
        return (0,0,0)
    return color_dict[mot_type]


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    # to plot ghost bboxes in black
    for i, tlwh in enumerate(tlwhs):
        obj_id = int(obj_ids[i])
        if obj_id == -1:
            color = get_color(abs(obj_id))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)

    return im


def plot_tracking_debug(image, tlwhs, obj_ids, acc_frame, seq, evaluator, scores=None, frame_id=0, fps=0., ids2=None):

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))*2

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness

        # draw MATCH, FP, SWITCH (i.e. boxes that are tracked)

        mot_type = acc_frame[acc_frame.HId.eq(obj_id)].Type.values[0]
        color = debug_color(mot_type)

        if mot_type == 'FP':
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=int(line_thickness/2))
        else:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    # draw MISS (i.e. boxes that are missed)

    # evaluator = Evaluator('/hdd/yongxinw/MOT17/MOT17/train', seq, data_type='mot')

    gt_objs = evaluator.gt_frame_dict.get(frame_id+1, [])
    gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

    miss_rows = acc_frame[acc_frame.Type.eq('MISS')]
    miss_OIds = miss_rows.OId.values
    for miss_OId in miss_OIds:

        x1, y1, w, h = gt_tlwhs[gt_ids==miss_OId][0] # 2d array -> 1d array
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        color = debug_color('MISS')

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, '{}'.format(miss_OId), (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    return im


def count_fn_debug(image, tlwhs, obj_ids, acc_frame, seq, evaluator, u8scores=None, frame_id=0, fps=0., ids2=None):

    det_area_sum_frame = 0
    det_count_frame = 0
    fn_areas_frame = []

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))*2

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)


    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        det_area_sum_frame += w * h
        det_count_frame += 1

    # count and categorize MISS (i.e. boxes that are missed)

    gt_objs = evaluator.gt_frame_dict.get(frame_id+1, [])
    gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

    miss_rows = acc_frame[acc_frame.Type.eq('MISS')]
    miss_OIds = miss_rows.OId.values
    for miss_OId in miss_OIds:

        try:
            x1, y1, w, h = gt_tlwhs[gt_ids==miss_OId][0] # 2d array -> 1d array
            box_size = w * h
            fn_areas_frame.append(box_size)
        except:
            continue

    return det_area_sum_frame, det_count_frame, fn_areas_frame


def plot_FN(image, tlwhs, obj_ids, acc_frame, seq, evaluator, scores=None, frame_id=0, fps=0., ids2=None, FN_tlbrs_selected=None):

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))*2

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    # draw match, FP, SWITCH
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness

        # draw MATCH, FP, SWITCH (i.e. boxes that are tracked)

        # mot_type = acc_frame[acc_frame.HId.eq(obj_id)].Type.values[0]
        # color = debug_color(mot_type)

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=(255,0,0), thickness=int(line_thickness/2))

        # if mot_type == 'FP':
        #     cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=int(line_thickness/2))
        # else:
        #     cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    # draw FN and selected FN

    gt_objs = evaluator.gt_frame_dict.get(frame_id+1, [])
    gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

    miss_rows = acc_frame[acc_frame.Type.eq('MISS')]
    miss_OIds = miss_rows.OId.values
    for miss_OId in miss_OIds:
        try:
            x1, y1, w, h = gt_tlwhs[gt_ids==miss_OId][0] # 2d array -> 1d array
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            color = debug_color('MISS')

            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, '{}'.format(miss_OId), (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)

            print()
            print('yellow:')
            print(intbox)
        except:
            pass

    for i, tlbr in enumerate(FN_tlbrs_selected):
        x1, y1, x2, y2 = tlbr
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        print('orange:')
        print(intbox)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=(0, 140, 255), thickness=int(line_thickness)) # orange


    return im

def get_overlap(ghost_tlwhs, acc_frame, seq, evaluator, frame_id=0):

    gt_objs = evaluator.gt_frame_dict.get(frame_id+1, [])
    gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

    miss_rows = acc_frame[acc_frame.Type.eq('MISS')]
    miss_OIds = miss_rows.OId.values

    fn_tlbrs = []
    ghost_tlbrs = []
    for miss_OId in miss_OIds:

        fn_tlwh = gt_tlwhs[gt_ids==miss_OId][0] # 2d array -> 1d array
        fn_tlbr = np.asarray(fn_tlwh).copy()
        fn_tlbr[2:] += fn_tlbr[:2]
        fn_tlbrs.append(fn_tlbr)

    for ghost_tlwh in ghost_tlwhs:
        ghost_tlbr = np.asarray(ghost_tlwh).copy()
        ghost_tlbr[2:] += ghost_tlbr[:2]
        ghost_tlbrs.append(ghost_tlbr)        


    iou_matrix = 1 - matching.iou_distance(fn_tlbrs, ghost_tlbrs)
    ghost_fn_overlaps = iou_matrix.flatten()

    # if len(ghost_fn_overlaps) > 0:
    #     import pdb
    #     pdb.set_trace()

    fn_closest_ghost_overlap = iou_matrix.max(axis=1)

    return ghost_fn_overlaps, len(miss_OIds), fn_closest_ghost_overlap


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im
