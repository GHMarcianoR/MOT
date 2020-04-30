import cv2 as cv
from tqdm import tqdm


def draw(info, image, show_objects):
    for bb, id_ , color in info:
        x, y, w, h = bb.minmax()
        image = cv.rectangle(image, (x, y), (w, h), color, thickness=4)
        cv.putText(image, str(id_), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if show_objects:
        cv.imshow('mot', image)
        # cv.waitKey()
    return image


def createVideo(listframes, filename, frame_rate, shape):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    videowriter = cv.VideoWriter(filename='videos/' + filename + '.mp4',
                                 fourcc=fourcc,
                                 fps=frame_rate,
                                 frameSize=shape)
    for frame in tqdm(listframes):
        videowriter.write(frame)
    videowriter.release()
