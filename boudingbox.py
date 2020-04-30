import numpy as np


class BoudingBox:
    # Representacao bouding box
    # inicializa no formato [x y w h]
    def __init__(self, coord, frame, object_id=-1, conf=-1):
        self.coord = np.array(coord, dtype=np.int)
        self.frame = frame
        self.object_id = object_id
        self.conf_score = conf

    def getConfScore(self):
        return self.conf_score

    def getIDFrame(self):
        return self.frame

    def getIDObject(self):
        return self.object_id

    def standard(self):
        return self.coord

    def minmax(self):
        # converte bouding box do formato padrao para:
        # [max_x, max_y, min_x, min_y]
        c = self.coord.copy()
        c[2:] += c[:2]
        return c

    def asp_ratio(self):
        c = self.coord.copy()
        c = c.astype(np.float)
        c[:2] += c[2:] / 2
        c[2] /= c[3]
        return c
