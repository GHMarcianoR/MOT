import pandas as pd
import configparser
from os import listdir
from os.path import isfile, join
from boudingbox import BoudingBox
import numpy as np


class Dataset:
    """Class to gather information about the dataset
    create primarily for datasets present in MOTChallenge.
    For usage in other datasets, modification may be required
    """

    def __init__(self, folder, evaluation):
        det_path = folder + 'det/det.txt'
        gt_path = folder + 'gt/gt.txt'
        seq_path = folder + 'seqinfo.ini'
        self.imgs_path = folder + 'img1/'

        # Must fix the columns names
        columns = ['frame_id', 'object_id', 'x', 'y', 'w', 'h',
                   'conf_score', 'class', 'visibility', '0']

        self.detections = pd.read_csv(det_path, header=None)
        qtd_col = len(self.detections.columns)
        self.detections.columns = columns[:qtd_col]
        if evaluation:
            self.groud_truth = pd.read_csv(gt_path, header=None)
            self.groud_truth.columns = columns[0:9]

        self.frames = [f for f in listdir(self.imgs_path) if isfile(join(self.imgs_path, f))]
        self.frames.sort(reverse=False)
        self.sequence_info = configparser.ConfigParser()
        self.sequence_info.read(seq_path)

    def getDetections(self, frame_id, gt=False):
        list_detections = []
        if gt:
            aux = self.groud_truth[self.groud_truth['frame_id'] == frame_id]
        else:
            aux = self.detections[self.detections['frame_id'] == frame_id]

        for i in range(len(aux)):
            info_bb = np.array(aux.iloc[i])
            detection = BoudingBox(coord=info_bb[2:6],
                                   frame=frame_id,
                                   object_id=info_bb[1],
                                   conf=info_bb[6])

            list_detections.append(detection)
        return list_detections

    def getNumberFrames(self):
        return self.sequence_info.getint('Sequence', 'seqLength')

    def getFrameRate(self):
        return self.sequence_info.getint('Sequence', 'frameRate')

    def getShapeFrames(self):
        return self.sequence_info.getint('Sequence', 'imWidth'), self.sequence_info.getint('Sequence', 'imHeight')

    def getListFrames(self):
        return self.frames

    def getImagePath(self):
        return self.imgs_path

    def getDatasetName(self):
        return self.sequence_info.get('Sequence', 'name')
