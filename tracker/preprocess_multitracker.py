import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
import copy

from utils.io import unzip_objs
from utils.utils import *
from utils.log import logger
from utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30, img_patch=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
        self.ghost = False
        self.tlwh_buffer = deque([], maxlen=buffer_size)
        
        self.img_patch = img_patch
        self.buffer_size = buffer_size
    
    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat 
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha *self.smooth_feat + (1-self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i,st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def sample_ghost_tracks(stracks, N):
        ghost_tracks = []

        for st in stracks:
            for i in range(N):
                st_copy = copy.deepcopy(st)
                st_copy.mean = np.random.multivariate_normal(st.mean, st.covariance)
                if st.mean[4] == 0:
                    st_copy.mean[4:] = [0, 0, 0, 0]
                ghost_tracks.append(st_copy)

        return ghost_tracks

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tlwh_buffer = deque([], maxlen=self.buffer_size)
        self.tlwh_buffer.append(new_track.tlwh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True, update_kf=True, var_multiplier=1):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        if update_kf:
            new_tlwh = new_track.tlwh
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), var_multiplier)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)
        self.tlwh_buffer.append(new_track.tlwh)
        self.img_patch = new_track.img_patch


    def update_FN(self, FN_tlwh, frame_id, FN_img_patch, update_feature=False, update_kf=True, var_multiplier=1):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        if update_kf:
            new_tlwh = FN_tlwh
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), var_multiplier)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.tlwh_buffer.append(FN_tlwh)

        self.score = 1
        self.img_patch = FN_img_patch #?


    def update_ghost(self, ghost_tlwh, frame_id, update_feature=True, var_multiplier=1):
        """
        Update a matched track with GPN regressed coords
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = ghost_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), var_multiplier)
        self.state = TrackState.Tracked
        self.is_activated = True

        # self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)
        self.tlwh_buffer.append(ghost_tlwh)


    def extend(self, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.state = TrackState.Tracked
        self.is_activated = True




    @property
    #@jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    #@jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    #@jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    #@jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    #@jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



class JDETracker(object):
    def __init__(self, opt, frame_rate=30, train=False):
        self.opt = opt
        self.model = Darknet(opt.cfg)
        # load_darknet_weights(self.model, opt.weights)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
        self.model.cuda().eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        # self.gpn = GPN().cuda()
        # self.loss_reg = nn.SmoothL1Loss().cuda()
        # self.loss_conf = nn.MSELoss().cuda()
        # self.optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.gpn.parameters()), lr=opt.lr, momentum=.9)
        # if train:
        #     self.gpn.train()

    def update_ghost_track(self, im_blob, img0, opt, evaluator, writer, n_iter, path):

        G = opt.G
        two_stage = opt.two_stage
        iou_ghost_match = opt.iou_ghost_match
        ghost_iou_ths = opt.ghost_iou_thres
        update_ghost_feat = opt.update_ghost_feat
        var_multiplier = opt.KF_var_mult
        N = opt.N
        occ_reason_thres = opt.occ_reason_thres
        thresholding_occ_reason = opt.thresholding_occ_reason
        use_featmap = opt.use_featmap

        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        ghost_match_iou = []


        t1 = time.time()
        ''' Step 1: Network forward, get detections & embeddings'''

        if use_featmap:
            with torch.no_grad():
                h0, w0, _ = img0.shape
                pred, conv5_out = self.model(im_blob)

                pred = pred[pred[:, :, 4] > self.opt.conf_thres]
                conv5_out = conv5_out
                if len(pred) > 0:
                    dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres,
                                               self.opt.nms_thres)[0]
                    scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
                    # dets, embs = dets[:, :5].cpu().numpy(), dets[:, 6:].cpu().numpy()
                    dets, embs = dets[:, :5].cpu().detach().numpy(), dets[:, 6:].cpu().detach().numpy()
                    bbox = np.rint(dets[:,:4]).astype(int)
                    bbox_x, bbox_y, bbox_w, bbox_h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]

                    bbox_x[bbox_x > w0] = w0
                    bbox_x[bbox_x < 0] = 0
                    bbox_y[bbox_y > h0] = h0
                    bbox_y[bbox_y < 0] = 0

                    h_rs = 224
                    w_rs = 224
                    img_patches = []
                    for x,y,w,h in zip(bbox_x, bbox_y, bbox_w, bbox_h):
                        tmp = img0[y:y+h, x:x+w, :]
                        tmp = np.ascontiguousarray(tmp)
                        import cv2
                        tmp = cv2.resize(tmp, (h_rs, w_rs))
                        img_patches.append(tmp)

                    # img_patches = np.stack(img_patches)

                    '''Detections'''
                    detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30, p) for
                                  (tlbrs, f, p) in zip(dets, embs, img_patches)]
                else:
                    detections = []

        else:
            with torch.no_grad():
                pred, conv5_out = self.model(im_blob)
                pred = pred[pred[:, :, 4] > self.opt.conf_thres]
                if len(pred) > 0:
                    dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres,
                                               self.opt.nms_thres)[0]
                    scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
                    # dets, embs = dets[:, :5].cpu().numpy(), dets[:, 6:].cpu().numpy()
                    dets, embs = dets[:, :5].cpu().detach().numpy(), dets[:, 6:].cpu().detach().numpy()
                    '''Detections'''
                    detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                                  (tlbrs, f) in zip(dets, embs)]
                else:
                    detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        ghost_dets = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                track.ghost = False
                activated_stracks.append(track)

                # *** For each feature matched detection, create a ghost detection ***
                ghost_det = copy.deepcopy(det)
                ghost_det.ghost = True
                ghost_dets.append(ghost_det)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        detections = [detections[i] for i in u_detection]


        ''' Step 3: Second association, with IOU'''

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state==TrackState.Tracked ]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                track.ghost = False
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        detections = [detections[i] for i in u_detection]



        if G > 0:
            # remove tracks that are out of frame (i.e. don't match ghosts with such tracks)
            for it in u_track:
                track = r_tracked_stracks[it]
                tlbr = track.tlbr
                if self.frame_id > 10 and tlbr[0] < 0 or tlbr[1] < 0 or tlbr[2] > 1088 or tlbr[3] > 608:
                    track.mark_lost()
                    lost_stracks.append(track)

            detections_g = ghost_dets
            newly_matched = []

        '''Mark unmatched tracks as lost'''
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # logger.debug('===========Frame {}=========='.format(self.frame_id))
        # logger.debug('Activated: {}'.format([track.track_id for track in activated_stracks]))
        # logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        #

        ''' Collect data for GPN'''

        prefix = path.split('img1')[0]
        if opt.use_featmap:
            dataset_root = '../preprocess-ghost-bbox-th{}-map-more-filter/'.format(opt.occ_reason_thres)
        else:
            dataset_root = '../preprocess-ghost-bbox-th0.6/'
        save_dir = osp.join(prefix, 'preprocess').replace('/hdd/yongxinw/', dataset_root)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_path = path.replace('/hdd/yongxinw/', dataset_root).replace('img1', 'preprocess').replace('.png', '').replace('.jpg', '')


        # Use motmetrics to find FNs

        trk_tlwhs = [track.tlwh for track in output_stracks]
        trk_ids = [track.track_id for track in output_stracks]
        # trk_ids = np.arange(len(trk_tlwhs))
        evaluator.eval_frame(self.frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        FN_tlbrs_selected = []
        tracks_selected = []


        before_boxes = []
        after_boxs = []

        if self.frame_id > 2:
            try:
                # Match unmatched tracks with matched dets
                unmatched_tracks = [r_tracked_stracks[it] for it in u_track]
                dists = matching.iou_distance(unmatched_tracks, detections_g)
                if thresholding_occ_reason:
                    if len(unmatched_tracks) > 0 and len(detections_g) > 0:
                        um_det_matches = list(zip(range(len(unmatched_tracks)), dists.argmin(axis=1)))
                        dists_min = dists.min(axis=1)
                        um_det_matches = np.array(um_det_matches)[dists_min <= occ_reason_thres,:]
                    else:
                        um_det_matches = []
                else:
                    um_det_matches, u_track, u_detection = matching.linear_assignment(dists, thresh=occ_reason_thres)

                map1 = {}
                for um, det in um_det_matches:
                    map1[unmatched_tracks[um].track_id] = detections_g[det]


                # todo: fix the bug in try part
                acc_frame = evaluator.acc.mot_events.loc[self.frame_id-1]
                miss_rows = acc_frame[acc_frame.Type.eq('MISS')]
                miss_OIds = miss_rows.OId.values

                gt_objs = evaluator.gt_frame_dict.get(self.frame_id, [])
                gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

                acc_frame_p = evaluator.acc.mot_events.loc[self.frame_id-2]

                for miss_OId in miss_OIds:
                    # print()
                    # print(self.frame_id)
                    # import pdb; pdb.set_trace()

                    FN_tlwh = gt_tlwhs[gt_ids==miss_OId][0]
                    FN_tlbrs_selected.append(STrack.tlwh_to_tlbr(FN_tlwh))

                    miss_HId_p = acc_frame_p[acc_frame_p.OId.eq(miss_OId)].HId.values
                    if len(miss_HId_p) == 0:
                        # print('cannot find miss OId from tracks in previous frame')
                        continue
                    else:
                        miss_HId_p = miss_HId_p[0]

                    track_id = miss_HId_p

                    track = None
                    for x in r_tracked_stracks:
                        if x.track_id == track_id:
                            track = x

                    if track == None or track.track_id not in map1:
                        # print('cannot find track ID from lost tracks in current frame')
                        continue

                    det = map1[track_id]

                    # print(self.frame_id)
                    # print(FN_tlwh, track.mean[:4].astype(np.int), det.tlbr)
                    target_delta_bbox = FN_tlwh - track.tlwh

                    before_boxes.append(track.tlwh)
                    after_boxs.append(FN_tlwh)

                    np.savez(save_path, track_feat=track.img_patch, det_feat=det.img_patch,
                             track_tlbr=track.tlbr, det_tlbr=det.tlbr, tlwh_history=track.tlwh_buffer,
                             target_delta_bbox=target_delta_bbox)

                    # print('except, updating with FN tlwh')
                    # add FN to the tracked pool
                    x,y,w,h = FN_tlwh.astype(int)
                    track.update_FN(FN_tlwh, self.frame_id, img0[y:y+h, x:x+w, :])
                    activated_stracks.append(track)
            except:
                pass


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)


        # update output_stracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]


        if opt.ghost_stats:
            return output_stracks, n_iter, last_ghosts, ghost_match_iou
        if opt.vis_FN:
            return output_stracks, n_iter, FN_tlbrs_selected, tracks_selected
        if opt.vis_UR:
            return output_stracks, n_iter, before_boxes, after_boxs
        return output_stracks, n_iter

    def update(self, im_blob, img0, opt, evaluator, writer, n_iter, path):

        if opt.ghost_track:
            return self.update_ghost_track(im_blob, img0, opt, evaluator, writer, n_iter, path)
        raise ValueError('In GPN branch, should always use ghost_track option')
        return None


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb
            

