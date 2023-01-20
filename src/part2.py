import numpy as np
import cv2
from cv2 import aruco
from utils import solve_homography, warping

def planarAR(REF_IMAGE_PATH, VIDEO_PATH):
    """
    Reuse the previously written function "solve_homography" and "warping" to implement this task
    :param REF_IMAGE_PATH: path/to/reference/image
    :param VIDEO_PATH: path/to/input/seq0.avi
    """
    # initialize the video stream and ref image
    video = cv2.VideoCapture(VIDEO_PATH)
    ref_image = cv2.imread(REF_IMAGE_PATH)
    # get the width and height of the film and image
    h, w, c = ref_image.shape
    film_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    film_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # fps: Frame rate of the output video.
    film_fps = video.get(cv2.CAP_PROP_FPS)
    # fourcc: 4-character code of codec used to compress the frames.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # initiate a video writer
    videowriter = cv2.VideoWriter("output2.mp4", fourcc, film_fps, (film_w, film_h))
    # initiate a corner detector
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    ref_corns = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # TODO: find homography per frame and apply backward warp
    frame_idx = 0
    while (video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {:d}'.format(frame_idx))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            # TODO: 1.find corners with aruco
            # function call to aruco.detectMarkers()
            corners, markerIds, _ = aruco.detectMarkers(frame, arucoDict, parameters=arucoParameters)
            corner = corners[0][0].astype(int)
            # TODO: 2.find homograpy
            # function call to solve_homography()
            H = solve_homography(ref_corns, corner)
            # TODO: 3.apply backward warp
            # function call to warping()
            xmin, ymin = np.min(corner, axis=0)
            xmax, ymax = np.max(corner, axis=0)
            frame = warping(ref_image, frame, H, ymin, ymax, xmin, xmax, direction='b')
            # write the images into film with the video writer
            videowriter.write(frame)
            frame_idx += 1

        else:
            break
    #  close the video file and the video writer
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # ================== Part 2: Marker-based planar AR========================
    VIDEO_PATH = '../resource/seq0.mp4'
    # TODO: you can change the reference image to whatever you want
    REF_IMAGE_PATH = '../resource/img5.png'
    planarAR(REF_IMAGE_PATH, VIDEO_PATH)