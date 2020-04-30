from kalman_filter import KalmanFilter
from boudingbox import BoudingBox
import random


def _restoreStandardValue(arr_):
    arr = arr_.copy()
    arr[2] *= arr[3]
    arr[:2] -= arr[2:] / 2
    return arr


class Track:
    def __init__(self, bb, id_, max_miss_):
        self.track = [bb]
        self.alive = True
        self.kf = KalmanFilter()
        self.id_track = id_
        self.mean, self.cov = self.kf.initiate(bb.asp_ratio())
        self.num_miss = 0
        self.num_max_miss = max_miss_
        self.project_mean = 0
        self.color = (int(random.random() * 256),
                      int(random.random() * 256),
                      int(random.random() * 256))

    def last(self):
        return self.track[-1]

    def get_last_mean(self):
        return self.mean[0:4]

    def append(self, bb):
        self.kalman_steps(bb.asp_ratio())
        aproxBB = BoudingBox(coord=_restoreStandardValue(self.get_last_mean()),
                             frame=bb.getIDFrame())
        self.track.append(aproxBB)

    def kalman_steps(self, dets):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, dets)

    def getTrack(self):
        return self.track

    def getBBFrame(self, frame_id):
        for bb in self.track:
            if bb.getIDFrame() == frame_id:
                return bb

    def __len__(self):
        return len(self.track)

    def is_alive(self):
        return self.alive

    def ressurect(self):
        self.alive = True

    def project_position(self):
        self.append(BoudingBox(_restoreStandardValue(self.mean[:4]),
                               self.last().getIDFrame() + 1))

    def missed(self):
        self.num_miss += 1
        self.project_position()
        if self.num_miss > self.num_max_miss:
            self.kill()

    def kill(self):
        self.alive = False

    def getID(self):
        return self.id_track

    def getcolor(self):
        return self.color