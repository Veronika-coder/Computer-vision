import cv2
import numpy as np
from cv2 import aruco
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.022
MARKER_LENGTH = 0.014
FRAME_SKIP = 30 

def calibrate_camera_aruco(video_path):
    logger.info("Initializing camera calibration...")
    
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard((SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file.")
        return False, None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_charuco_ids = []
    all_charuco_corners = []

    process_bar = tqdm(total=total_frames, desc='Processing frames', position=0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        process_bar.update(1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if len(marker_corners) > 0:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board)

            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) == 24:
                all_charuco_ids.append(charuco_ids)
                all_charuco_corners.append(charuco_corners)

    cap.release()
    

    if all_charuco_corners and all_charuco_ids:
        logger.info("Performing camera calibration...")
        ret, mtx, dist, _, _ = aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board,
                                                            (width, height), None, None)
        if ret:
            logger.info("Calibration successful!")
            logger.info("Camera matrix:\n%s", mtx)
            logger.info("Distortion coefficients:\n%s", dist)
            return True, mtx, dist
        else:
            logger.error("Calibration failed.")
            return False, None, None
    else:
        logger.error("No valid Charuco corners detected.")
        return False, None, None

def main():
    ret, mtx, dist = calibrate_camera_aruco(r'D:/8semestr/comp zir/lab3/1_calibration.mp4')

    if ret:
        np.savez('D:/8semestr/comp zir/lab3/calibration_results.npz', mtx=mtx, dist=dist)

if __name__ == "__main__":
    main()
