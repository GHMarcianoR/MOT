from track import Track
from dataset import Dataset
from visualization import *
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import motmetrics as mm
from metric import iou
import numpy as np


class Core:

    def __init__(self, folder, draw_, min_iou_, max_miss_, conf_v, eval_=False, ):

        self.dataset = Dataset(folder, eval_)
        self.Tracks = OrderedDict()
        self.num_frames = self.dataset.getNumberFrames()
        self.next_id = 0
        self.draw_objects, self.show_objects, self.draw_video = draw_
        self.eval = eval_
        self.min_iou = min_iou_
        self.max_miss = max_miss_
        self.conf_value = conf_v
        self._create_new_tracks(self.dataset.getDetections(1))
        if self.draw_objects:
            self.list_frames = [self._drawObjects(1)]

        if self.eval:
            self.acc = mm.MOTAccumulator(auto_id=True)
            self.mh = mm.metrics.create()
            self._eval(1)

    def _assigment(self, frame):
        tracks_alive = self._get_tracks_alive()
        detections = [bb for bb in self.dataset.getDetections(frame) if bb.getConfScore() >= self.conf_value]
        cost_mat = np.zeros((len(tracks_alive), len(detections)))
        missed_det = []

        for idx, track in enumerate(tracks_alive):
            for jdx, bb in enumerate(detections):
                cost_mat[idx][jdx] = iou(track.last().minmax(),
                                         bb.minmax())

        track_ind, det_ind = linear_sum_assignment(cost_mat, maximize=True)

        for i, j in zip(track_ind, det_ind):
            if cost_mat[i][j] > self.min_iou:
                tracks_alive[i].append(detections[j])
            else:
                tracks_alive[i].missed()
                if not tracks_alive[i].is_alive():
                    missed_det.append(detections[j])

        if len(tracks_alive) < len(detections):
            missed_det = [detections[i] for i in range(len(detections)) if i not in det_ind]
            self._create_new_tracks(missed_det)

    def next_step(self):
        for track in self._get_tracks_alive():
            track.project_position()

    def _create_new_tracks(self, lbbs):
        for bb in lbbs:
            track = Track(bb, self.next_id, self.max_miss)
            self.Tracks[self.next_id] = track
            self.next_id += 1

    def main_loop(self):

        for current_frame in tqdm(range(2, self.num_frames)):

            self._assigment(current_frame)

            if self.eval:
                self._eval(current_frame)

            if self.draw_objects:
                image = self._drawObjects(current_frame)
                self.list_frames.append(image)

        if self.draw_video:
            createVideo(self.list_frames,
                        str(self.dataset.getDatasetName()),
                        self.dataset.getFrameRate(),
                        self.dataset.getShapeFrames())

    def _drawObjects(self, frame):
        image = cv.imread(self.dataset.getImagePath() + self.dataset.getListFrames()[frame])
        info_objects = []
        for track in self._get_tracks_alive():
            info_objects.append((track.last(), track.getID(), track.getcolor()))

        return draw(info_objects, image, self.show_objects)

    def _get_tracks_alive(self):
        return [track for key, track in self.Tracks.items() if track.is_alive()]

    def _get_tracks_not_alive(self):
        return [track for key, track in self.Tracks.items() if not track.is_alive()]

    def _get_bb_last_frame(self):
        return [track.last() for key, track in self.Tracks.items() if track.is_alive()]

    def _eval(self, frame):
        hip_bb, hip_ids = [], []
        for track in self._get_tracks_alive():
            hip_bb.append(track.last().standard())
            hip_ids.append(track.getID())
        if len(hip_bb) > 0:
            hip_bb = np.stack(hip_bb, axis=0)
        stack_gts = []
        gt_ids = []
        for bb in self.dataset.getDetections(frame, gt=True):
            stack_gts.append(np.array(bb.standard()))
            gt_ids.append(bb.getIDObject())

        stack_gts = np.stack(stack_gts, axis=0)

        distance = mm.distances.iou_matrix(stack_gts, hip_bb, max_iou=0.5)
        self.acc.update(gt_ids, hip_ids, distance)

    def train_results(self):

        summary = self.mh.compute(self.acc,
                                  metrics=mm.metrics.motchallenge_metrics,
                                  name=['MOTMetrics'])

        strsummary = mm.io.render_summary(
            summary,
            formatters=self.mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        return strsummary

    def test_results(self, name):
        result = open('results/'+str(name) + '.txt', 'w')
        for key, track in self.Tracks.items():
            for bb in track.getTrack():
                x, y, w, h = bb.standard()
                result.write(str(bb.getIDFrame()) + ',' +
                             str(track.getID()+1) + ',' +
                             str(x) + ',' +
                             str(y) + ',' +
                             str(w) + ',' +
                             str(h) + ',' +
                             str(1) + ',' +
                             str(-1) + ',' +
                             str(-1) + ',' +
                             str(-1) + ',' +
                             '\n')
