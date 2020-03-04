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

from utils.utils import *
from utils.log import logger
from utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

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
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True, update_kf=True):
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
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)


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
    def __init__(self, opt, frame_rate=30):
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

    def update(self, im_blob, img0, ghost, G, save_lt, two_stage, small_ghost,
        feat_ghost_match, iou_ghost_match, occ_ghost_match, ghost_feature_ths, ghost_iou_ths, ghost_occ_ths, save_thres,
        update_ghost_feat, update_ghost_coords, ghost_stats=False):

        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        ghost_match_iou = []

        if save_lt:

            t1 = time.time()
            ''' Step 1: Network forward, get detections & embeddings'''
            with torch.no_grad():
                pred = self.model(im_blob)
            pred = pred[pred[:, :, 4] > self.opt.conf_thres]
            if len(pred) > 0:
                dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, 
                                           self.opt.nms_thres)[0]
                scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
                dets, embs = dets[:, :5].cpu().numpy(), dets[:, 6:].cpu().numpy()
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

            ghosts = []
            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    track.ghost = False
                    activated_stracks.append(track)

                    # For each matched detection, create a ghost detection
                    ghost= copy.deepcopy(det)
                    ghost.ghost = True
                    ghosts.append(ghost)
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
                # remove tracks that are out of frame (i.e. don't mask ghosts with such tracks)
                for it in u_track:
                    track = r_tracked_stracks[it]
                    tlbr = track.tlbr
                    if self.frame_id > 10 and tlbr[0] < 0 or tlbr[1] < 0 or tlbr[2] > 1088 or tlbr[3] > 608:
                        track.mark_lost()
                        lost_stracks.append(track)


            ''' !!! Ghost association !!! '''
            for iteration in range(G):

                newly_matched = []
                # detections.extend(ghosts) # deprecated
                detections_g = ghosts

                # IoU match for ghosts
                if iou_ghost_match:

                    r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state==TrackState.Tracked ]
                    dists = matching.iou_distance(r_tracked_stracks, detections_g)

                    if len(r_tracked_stracks) > 0:
                        matches = list(zip(range(len(r_tracked_stracks)), dists.argmin(axis=1)))
                        dists_min = dists.min(axis=1)
                        matches = np.array(matches)[dists_min <= save_thres,:]
                        u_track = np.where(dists_min > save_thres)[0]
                    else:
                        matches = []
                        u_track = range(len(r_tracked_stracks))


                    for itracked, idet in matches:
                        track = r_tracked_stracks[itracked]
                        det = detections_g[idet]
                        if track.state == TrackState.Tracked:
                            if ghost_stats:
                                ghost_match_iou.append(1-dists[itracked, idet])
                            # print(1 - dists[itracked, idet])
                            # if iteration == 0:
                            #     print('!!! IoU match')
                            # else:
                            #     print('!!! small ghost, IoU match')

                            track.extend(self.frame_id)
                            track.ghost = True
                            activated_stracks.append(track)
                            newly_matched.append(copy.deepcopy(det))   
                        else:
                            if det.ghost:
                                continue
                            track.re_activate(det, self.frame_id, new_id=False)
                            refind_stracks.append(track)


                # if no ghost is matched in this iteration, we no longer create ghost duplicates, so we stop ghost matching
                if len(newly_matched) == 0:
                    print("Break because no ghost is matched in current iteration")
                    break
                ghosts = newly_matched

            ''' End of Ghost association'''

            '''Mark unmatched tracks as lost'''
            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

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



        elif two_stage:

            # print('Performing two-stage association (ghost bbox)')
            t1 = time.time()
            ''' Step 1: Network forward, get detections & embeddings'''
            with torch.no_grad():
                pred = self.model(im_blob)
            pred = pred[pred[:, :, 4] > self.opt.conf_thres]
            if len(pred) > 0:
                dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, 
                                           self.opt.nms_thres)[0]
                scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
                dets, embs = dets[:, :5].cpu().numpy(), dets[:, 6:].cpu().numpy()
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

            ghosts = []
            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    track.ghost = False
                    activated_stracks.append(track)

                    # *** For each feature matched detection, create a ghost detection ***
                    ghost= copy.deepcopy(det)
                    ghost.ghost = True
                    ghosts.append(ghost)
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


            ''' !!! Ghost association !!! '''
            for iteration in range(G):

                # adjust ghost box area by 50% (i.e. decrease box size length by 30%)
                if small_ghost and iteration == 1:
                    for i in range(len(ghosts)):
                        ghosts[i]._tlwh[0] += ghosts[i].tlwh[2] / 2 * 0.3
                        ghosts[i]._tlwh[1] += ghosts[i].tlwh[3] / 2 * 0.3
                        ghosts[i]._tlwh[2] *= 0.7
                        ghosts[i]._tlwh[3] *= 0.7

                newly_matched = []
                # detections.extend(ghosts) # deprecated
                detections_g = ghosts

                if feat_ghost_match:

                    r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state==TrackState.Tracked]
                    dists = matching.embedding_distance(r_tracked_stracks, detections_g)
                    dists = matching.fuse_motion(self.kalman_filter, dists, r_tracked_stracks, detections_g)
                    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=ghost_feature_ths)

                    for itracked, idet in matches:
                        track = r_tracked_stracks[itracked]
                        det = detections_g[idet]
                        if track.state == TrackState.Tracked:
                            print('!!! feature match')
                            track.update(detections_g[idet], self.frame_id, update_ghost_feat, update_ghost_coords)
                            track.ghost = True
                            activated_stracks.append(track)
                            newly_matched.append(copy.deepcopy(det))
                        else:
                            # don't initiate track for a ghost detection box
                            if det.ghost:
                                continue
                            track.re_activate(det, self.frame_id, new_id=False)
                            refind_stracks.append(track)
                    detections_g = [detections_g[i] for i in u_detection]


                if iou_ghost_match:

                    r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state==TrackState.Tracked ]
                    dists = matching.iou_distance(r_tracked_stracks, detections_g)
                    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=ghost_iou_ths)

                    for itracked, idet in matches:
                        track = r_tracked_stracks[itracked]
                        det = detections_g[idet]
                        if track.state == TrackState.Tracked:
                            if ghost_stats:
                                ghost_match_iou.append(1-dists[itracked, idet])
                            # print(1 - dists[itracked, idet])
                            # if iteration == 0:
                            #     print('!!! IoU match')
                            # else:
                            #     print('!!! small ghost, IoU match')

                            track.update(det, self.frame_id, update_ghost_feat, update_ghost_coords)
                            track.ghost = True
                            activated_stracks.append(track)
                            newly_matched.append(copy.deepcopy(det))   
                        else:
                            if det.ghost:
                                continue
                            track.re_activate(det, self.frame_id, new_id=False)
                            refind_stracks.append(track)
                    detections_g = [detections_g[i] for i in u_detection]

                if occ_ghost_match:

                    r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state==TrackState.Tracked ]
                    dists = matching.occ_distance(r_tracked_stracks, detections_g)
                    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=ghost_occ_ths)

                    for itracked, idet in matches:
                        track = r_tracked_stracks[itracked]
                        det = detections_g[idet]
                        if track.state == TrackState.Tracked:
                            print('!!! Occlusion match')
                            track.update(det, self.frame_id, update_ghost_feat, update_ghost_coords)
                            track.ghost = True
                            activated_stracks.append(track)
                            newly_matched.append(copy.deepcopy(det))   
                        else:
                            if det.ghost:
                                continue
                            track.re_activate(det, self.frame_id, new_id=False)
                            refind_stracks.append(track)
                    detections_g = [detections_g[i] for i in u_detection]


                # if no ghost is matched in this iteration, we no longer create ghost duplicates, so we stop ghost matching
                if len(newly_matched) == 0:
                    print("Break because no ghost is matched in current iteration")
                    break
                ghosts = newly_matched

            ''' End of Ghost association'''
            
            '''Mark unmatched tracks as lost'''
            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

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


        else: # Not used because naively adding ghosts causes too many FPs. Basically deprecated

            print('Performing naive association with ghost bboxes')

            t1 = time.time()
            ''' Step 1: Network forward, get detections & embeddings'''
            with torch.no_grad():
                pred = self.model(im_blob)
            pred = pred[pred[:, :, 4] > self.opt.conf_thres]
            original_num_dets = 0
            if len(pred) > 0:
                dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, 
                                           self.opt.nms_thres)[0]
                original_num_dets = len(dets)
                scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
                dets, embs = dets[:, :5].cpu().numpy(), dets[:, 6:].cpu().numpy()
                '''Detections'''
                detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                              (tlbrs, f) in zip(dets, embs)]
                if ghost and self.frame_id > 1:
                    detections =[copy.deepcopy(d) for d in detections for _ in range(G+1)] # order: 1 2 3 1 2 3 1 2 3 ...
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

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_stracks.append(track)
                else:
                    if idet > original_num_dets:
                        continue
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            ''' Step 3: Second association, with IOU'''
            # detections = [detections[i] for i in u_detection]
            detections = [detections[i] for i in u_detection if i < original_num_dets]
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state==TrackState.Tracked ]
            dists = matching.iou_distance(r_tracked_stracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
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

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_stracks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        # print('===========Frame {}=========='.format(self.frame_id))
        # print('Activated: {}'.format([track.track_id for track in activated_stracks]))
        # print('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # print('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # print('Removed: {}'.format([track.track_id for track in removed_stracks]))

        if ghost_stats:
            return output_stracks, ghosts, ghost_match_iou

        return output_stracks


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
            

