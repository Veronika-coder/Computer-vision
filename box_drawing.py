import cv2
import numpy as np

calibration_data = np.load('D:/8semestr/comp zir/lab3/calibration_results.npz')
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters_create()

video_path = 'D:/8semestr/comp zir/lab3/2_pose_estimation.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

object_height = 12
marker_length = 7 

distance_between_markers = 12

output_video_path = 'D:/8semestr/comp zir/lab3//output_video.avi'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        object_center = np.mean(tvecs, axis=0)[0]
        object_orientation = np.mean(rvecs, axis=0)[0]
        
        length = distance_between_markers
        width = marker_length
        height = object_height
        
        rotation_matrix, _ = cv2.Rodrigues(object_orientation)
        object_points = np.array([
            [-width/2, -length/2, 0],
            [ width/2, -length/2, 0],
            [ width/2,  length/2, 0],
            [-width/2,  length/2, 0],
            [-width/2, -length/2, -height],
            [ width/2, -length/2, -height],
            [ width/2,  length/2, -height],
            [-width/2,  length/2, -height]
        ])
        
        transformed_points = cv2.projectPoints(
            object_points,
            object_orientation,
            object_center,
            camera_matrix,
            dist_coeffs
        )[0].reshape(-1, 2)
        
        for point in transformed_points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
        
        for i in range(4):
            i_next = (i + 1) % 4
            pt1 = (int(transformed_points[i][0]), int(transformed_points[i][1]))
            pt2 = (int(transformed_points[i_next][0]), int(transformed_points[i_next][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            pt3 = (int(transformed_points[i][0]), int(transformed_points[i][1]))
            pt4 = (int(transformed_points[i + 4][0]), int(transformed_points[i + 4][1]))
            cv2.line(frame, pt3, pt4, (0, 255, 0), 2)
            
            pt5 = (int(transformed_points[i_next][0]), int(transformed_points[i_next][1]))
            pt6 = (int(transformed_points[i_next + 4][0]), int(transformed_points[i_next + 4][1]))
            cv2.line(frame, pt5, pt6, (0, 255, 0), 2)
    
    out.write(frame)
    
    cv2.imshow('Frame with ArUco markers', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
