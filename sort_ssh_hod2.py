"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

import math
import pickle as pkl
import pandas as pd


np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)

  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
  return(o)


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))



# ------------------------------------------------------------------------------------------
#    Main function
#
# ------------------------------------------------------------------------------------------

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument('--display2', dest='display2', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)

    # data argument
    parser.add_argument("--hod_path",
                        help="Path to detections of hands.",
                        type=str, required=True)
    parser.add_argument("--dataset_path",
                        help="Path to image dataset.",
                        type=str, required=True)
    parser.add_argument("--class_anno_path",
                        help="Path to class annotation of image dataset.",
                        type=str, required=True)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    return args


def decision2(trackerss):
    """
    (1)検出が3つ以上ある場合は、2つ以外は誤検出の可能性が高い
    (2)誤検出のトラッカーidは正常なものよりも出現頻度が少ないと考えられるので、出現頻度上位2つを正常なトラッカーidに採用
    """
    id2count = {}
    for tracker in trackerss:
        for d in tracker:
            d[4] = int(d[4])
            id2count[d[4]] = id2count[d[4]]+1 if (d[4] in id2count) else 1

    id2count = sorted(id2count.items(), key=lambda x:x[1], reverse=True)
    if len(id2count) == 0:
        trk1_id, trk2_id = -1, -1
    elif len(id2count) == 1:
        trk1_id, trk2_id = id2count[0][0], -1
    else:
        trk1_id, trk2_id = id2count[0][0], id2count[1][0]

    return trk1_id, trk2_id


def distribute(hand1, hand2, trk1_id, trk2_id, trackers, start_index, frame):
    # get bbox(4), center(2)
    trk1_h, trk2_h = [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]
    for d in trackers:
        if d[4] == trk1_id:
            trk1_h = [d[0], d[1], d[2], d[3], (d[0]+d[2])/2, (d[1]+d[3])/2, d[4]]
        elif d[4] == trk2_id:
            trk2_h = [d[0], d[1], d[2], d[3], (d[0]+d[2])/2, (d[1]+d[3])/2, d[4]]

    # distribute
    if (len(hand1)==0) or (len(hand1)>0 and hand1[4:]==[-1,-1,-1] and hand2[4:]==[-1,-1,-1]):
        """
        case1: 履歴が空
        case2: 履歴が存在するが, 両方とも全てが未検出状態
        """
        hand1.append(trk1_h)
        hand2.append(trk2_h)
    else:
        """
        case1-4: 現在のtrk_idのうち、少なくとも片方が未検出状態、少なくとも片方がhand履歴に存在（存在する方を手がかり）
        case6:   現在のtrk_idのうち、少なくとも片方が未検出状態、両方ともhand履歴に存在しない（現在のtrk_idとhand履歴の距離を手がかり）
        case5:   現在のtrk_idのうち、両方とも未検出状態（特に考慮するべき点はない）
        """
        range_hand1 = np.array(hand1[start_index:])
        range_hand2 = np.array(hand2[start_index:])
        if trk1_h[6] in range_hand1[:, 6] and trk1_h[6] != -1:   # (1)trk_id1がhand1履歴に存在
            hand1.append(trk1_h)
            hand2.append(trk2_h)
        elif trk2_h[6] in range_hand1[:, 6] and trk2_h[6] != -1: # (2)trk_id2がhand1履歴に存在
            hand1.append(trk2_h)
            hand2.append(trk1_h)
        elif trk1_h[6] in range_hand2[:, 6] and trk1_h[6] != -1: # (3)trk_id1がhand2履歴に存在
            hand2.append(trk1_h)
            hand1.append(trk2_h)
        elif trk2_h[6] in range_hand2[:, 6] and trk2_h[6] != -1: # (4)trk_id2がhand2履歴に存在
            hand2.append(trk2_h)
            hand1.append(trk1_h)
        elif trk1_h[6] == -1 and trk2_h[6] == -1:                # (5)両方未検出なら考慮不要
            hand1.append(trk1_h)
            hand2.append(trk2_h)
        else:                                                    # (6)現在の両方のtrk_idが, 両方のhand履歴に存在しない
            hand1_sum, hand2_sum = 0, 0
            hand1_mean, hand2_mean = [1, 1], [1, 1] # [x, y], forbid zero division.
            for i in range(len(range_hand1)):
                if range_hand1[i][4] != -1 and range_hand1[i][5] != -1:
                    hand1_sum += 1
                    hand1_mean[0] += range_hand1[i][4]
                    hand1_mean[1] += range_hand1[i][5]
                if range_hand2[i][4] != -1 and range_hand2[i][5] != -1:
                    hand2_sum += 1
                    hand2_mean[0] += range_hand2[i][4]
                    hand2_mean[1] += range_hand2[i][5]
            if hand1_sum>0: # hand履歴が全て未検出のときは、平均値は初期のまま
                hand1_mean = [hand1_mean[0]/hand1_sum, hand1_mean[1]/hand1_sum]
            if hand2_sum>0:
                hand2_mean = [hand2_mean[0]/hand2_sum, hand2_mean[1]/hand2_sum]
            d11 = math.sqrt((hand1_mean[0]-trk1_h[4])**2 + (hand1_mean[1]-trk1_h[5])**2) #math.sqrt(math.dist(hand1_mean, trk1_h[]))
            d12 = math.sqrt((hand1_mean[0]-trk2_h[4])**2 + (hand1_mean[1]-trk2_h[5])**2) #math.sqrt(math.dist(hand1_mean, trk2_h[]))
            if d11<d12: #ユークリッド距離が小さい方を追加
                hand1.append(trk1_h)
                hand2.append(trk2_h)
            else:
                hand1.append(trk2_h)
                hand2.append(trk1_h)
            print('distance: d11={}, d12={}, frame={}'.format(d11, d12, int(frame.replace('frame_00000', '').replace('.jpg', ''))))

    return hand1, hand2


def complement(hand, n=10):
    """
        現在が未検出のときは、前後n枚の平均値をとることで補正する
        前後n枚が全て未検出なら、現在も未検出で確定とする
    """
    for i, d in enumerate(hand):
        if d[:4] != [-1,-1,-1,-1]: # 平均化する必要がない
            continue

        min_index = 0 if (i<n) else i-n #はみ出る時は最初から
        max_index = len(hand)-1 if (i+n > len(hand)-1) else i+n #はみ出る時は最後まで
        range_hand = hand[min_index:max_index+1]

        count, mean = 0, [0, 0] #[x, y]
        for t in range_hand: # 平均値計算
            if t[4] != -1 and t[5] != -1:
                count += 1
                mean[0] += t[4]
                mean[1] += t[5]
        if count>0:
            mean = [mean[0]/count, mean[1]/count]
            hand[i][4:6] = mean

    return hand


def makeleftright(hand1, hand2):
    """
    x座標の平均値の大小で、左右を決める
    """
    hand1_sum, hand2_sum, hand1_mean_x, hand2_mean_x = 0, 0, 0, 0
    for i in range(len(hand1)):
        if hand1[i][4] != -1:
            hand1_sum += 1
            hand1_mean_x += hand1[i][4]
        if hand2[i][4] != -1:
            hand2_sum += 1
            hand2_mean_x += hand2[i][4]
    if hand1_sum > 0:
        hand1_mean_x = hand1_mean_x/hand1_sum
    if hand2_sum > 0:
        hand2_mean_x = hand2_mean_x/hand2_sum

    #{0:'Left', 1:'Right'} hand1 == Left(0)
    if hand1_mean_x > hand2_mean_x:
        return hand2, hand1
    return hand1, hand2


def tracking(hod_data, basename, framedir, anno_data):
    """
    通常のSortに仮定を追加
    追加仮定: 動画内では、物体検出数は2であり、その物体は同一人物の右手と左手である。
    TODO: 物体が長時間出現していないときの処理。hand履歴を書き換えてしまうので、常に手が存在していることになってしまう。
          対処法としては、hand履歴を書き換えない形にする。デフォルトのhand履歴を残しておく。
    """
    print("Processing [%s]."%(basename))

    # sort.pyの通常処理
    trackerss = []
    mot_tracker = Sort(max_age=args.max_age,
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker

    for i in range(len(hod_data)):
        frame = hod_data[i]['frame_index']
        dets = hod_data[i]['hand_dets'] #[boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
        if dets is None:
            trackers = []
            trackerss.append(trackers)
            continue
        else:
            dets = dets[:, :5]
            trackers = mot_tracker.update(dets)
            trackerss.append(trackers)

    # 追加仮定を用いた処理
    hand1, hand2 = [], [] #[[x1, y1, x2, y2, center_x, center_y], ...]
    n = 60
    for i, trackers in enumerate(trackerss):
        frame=hod_data[i]['frame_index']
        min_index = 0 if (i<n) else i-n
        max_index = len(trackerss)-1 if (i+n > len(trackerss)-1) else i+n
        start_index = min_index
        range_trkss = trackerss[min_index:max_index+1]

        trk1_id, trk2_id = decision2(range_trkss) # 出現頻度上位2つを選択
        hand1, hand2 = distribute(hand1, hand2, trk1_id, trk2_id, trackers, start_index, frame) #hand履歴に振り分け

    hand1 = complement(hand1, n=15) # 予測モデル, sortでの検出抜けの補正
    hand2 = complement(hand2, n=15)
    hand1, hand2 = makeleftright(hand1, hand2) # 左右の決定
    _ = False # TODO: 軌道の平均化(滑らかにする)(近似曲線?)

    # 保存
    if args.save_path:
        print('save {}.pkl'.format(basename))
        f = open('{}/{}.pkl'.format(args.save_path, basename),'wb')
        pkl.dump({'hand_L': hand1, 'hand_R': hand2}, f)
        f.close

    # 結果出力
    if args.display or args.display2:
        if not os.path.exists('hod_output_hypo'):
            os.mkdir('hod_output_hypo')

        # get label
        t = basename.split('_')
        label = anno_data.loc[
            (anno_data['video_id'] == '{}_{}'.format(t[0], t[1])) &
            (anno_data['start_frame'] == t[2]) & (anno_data['stop_frame'] == t[3])]
        if label.empty == True:
            label_verb, label_noun = 'No label', 'No label'
        else:
            label_verb, label_noun = label['verb'].values[0], label['noun'].values[0]

    if args.display: # individual image
        if not os.path.exists('hod_output_hypo/{}'.format(basename)):
            os.mkdir('hod_output_hypo/{}'.format(basename))
        for i in range(len(hod_data)):
            frame = hod_data[i]['frame_index']
            im = io.imread(os.path.join(framedir, frame))
            ax.imshow(im)
            if hand1[i][4:] != [-1, -1, -1]: # Left bbox
                ax.add_patch(patches.Rectangle((hand1[i][0], hand1[i][1]), hand1[i][2]-hand1[i][0], hand1[i][3]-hand1[i][1], fill=False, lw=3, ec=colours[0+1,:]))
            if hand2[i][4:] != [-1, -1, -1]: # Right bbox
                ax.add_patch(patches.Rectangle((hand2[i][0], hand2[i][1]), hand1[i][2]-hand1[i][0], hand1[i][3]-hand1[i][1], fill=False, lw=3, ec=colours[1+1,:]))
            if hand1[i][4] != -1 and hand1[i][5] != -1:    # Left point
                ax.add_patch(patches.Circle(xy=(hand1[i][4], hand1[i][5]), radius=1, fc=colours[0+1,:], ec=colours[0+1,:]))
            if hand2[i][4] != -1 and hand2[i][5] != -1:    # Right point
                ax.add_patch(patches.Circle(xy=(hand2[i][4], hand2[i][5]), radius=1, fc=colours[1+1,:], ec=colours[1+1,:]))
            fig.canvas.flush_events()
            plt.savefig("hod_output_hypo/{}/{}".format(basename, frame))
            plt.draw()
            ax.cla()

    if args.display2: # overall trajectory
        if not os.path.exists('hod_output_hypo/overall_trajectory'):
            os.mkdir('hod_output_hypo/overall_trajectory')
        if len(hod_data) == 0:
            print('hod_data is empty')
            return
        if 'frame_index' not in hod_data[0]:
            print('invalid hod_data')
            return

        im = io.imread(os.path.join(framedir, hod_data[0]['frame_index']))
        im_white = np.ones(im.shape, np.uint8)*255
        ax.imshow(im_white)
        # draw label
        x = 40
        y = -10
        s = 'label: verb={}, noun={}'.format(label_verb, label_noun)
        ax.text(x, y, s)
        # draw point
        for i in range(len(hod_data)):
            if hand1[i][4] != -1 and hand1[i][5] != -1: # Left point
                ax.add_patch(patches.Circle(xy=(hand1[i][4], hand1[i][5]), radius=1, fc=colours[0+1,:], ec=colours[0+1,:]))
            if hand2[i][4] != -1 and hand2[i][5] != -1: # Right point
                ax.add_patch(patches.Circle(xy=(hand2[i][4], hand2[i][5]), radius=1, fc=colours[1+1,:], ec=colours[1+1,:]))
        # save
        fig.canvas.flush_events()
        plt.savefig('hod_output_hypo/overall_trajectory/{}.png'.format(basename))
        plt.draw()
        ax.cla()


if __name__ == '__main__':
    args = parse_args()
    colours = np.random.rand(32, 3)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    print('save_path: {}'.format(args.save_path))

    with open(args.class_anno_path, 'rb') as f:
        # annotation file
        anno_data = pkl.load(f)

        # processing
        patt1 = os.path.join(args.hod_path, 'P*')
        for box_dir in glob.glob(patt1):
            patt2 = os.path.join(box_dir, '*.pkl')
            for box_pkl in glob.glob(patt2):
                with open(box_pkl, 'rb') as f:
                    hod_data = pkl.load(f)
                    basename = os.path.basename(box_pkl).replace('.pkl', '')
                    framedir = os.path.join(args.dataset_path, basename[0:3], 'rgb_frames', basename[0:6])
                    tracking(hod_data, basename, framedir, anno_data)
