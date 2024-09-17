import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2


def get_head_rotation(
    landmarks: landmark_pb2.NormalizedLandmarkList,
    frame: np.ndarray[np.uint8],
) -> tuple[np.ndarray[np.uint8], np.ndarray]:
    """This funtion is used to estimate head pose and rotation angles given a set of points, their image projections,
    the camera intrinsec matrix and the distorsion matrix. This result is obtained
    by solving PnP (Perspective n-points) problem.
    For more details: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html

    Args:
        landmarks (landmark_pb2.NormalizedLandmarkList): Normalized Landmarks object obtained in a given frame.
        frame (np.ndarray[np.uint8]): Frame of the video/cam.

    Returns:
        frame (np.ndarray): The frame with the head rotation lines drawn.
        nose_end_point_2d (np.ndarray): The coordinates of the nose end point

    """
    height, width, _channels = frame.shape
    focal_length = 1 * width
    cam_matrix = np.array(
        [[focal_length, 1, height / 2], [0, focal_length, height / 2], [0, 0, 1]]
    )
    mdists = np.zeros((4, 1), dtype=np.float64)

    face_3d_ref, nose_3d_rref = create_3d_reference()
    face_2d_ref, nose_2d_ref = create_2d_reference(landmarks, frame)

    _success, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d_ref, face_2d_ref, cam_matrix, mdists
    )
    nose_end_point_2d, _jacobian = cv2.projectPoints(
        nose_3d_rref,
        rotation_vector,
        translation_vector,
        cam_matrix,
        mdists,
    )

    frame = draw_projection_lines(frame, nose_2d_ref, nose_end_point_2d)

    return frame, nose_end_point_2d


def draw_projection_lines(
    frame: np.ndarray[np.uint8], nose2D: list[float], nose_end_point: np.ndarray
) -> np.ndarray[np.uint8]:
    """Draws projection lines on the given frame from the nose reference point to the projected nose end points.

    Args:
        frame (np.ndarray[np.uint8]): Frame of the video/cam.
        nose2D (list[float]): nose2D refence point.
        nose_end_points (np.array): Output array of image points obtained by solving OpenCV
                                    projectPoints() problem.

    Returns:
        np.ndarray[np.uint8]: Frame of the video/cam with the lines drawn.

    """

    p1 = (int(nose2D[0]), int(nose2D[1]))
    p2 = (int(nose_end_point[0, 0, 0]), int(nose_end_point[0, 0, 1]))

    color = (255, 102, 102)
    thickness = 2

    cv2.line(frame, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)

    cv2.circle(frame, p2, 10, (0, 0, 255), -1)

    return frame


def create_2d_reference(
    landmarks: landmark_pb2.NormalizedLandmarkList, frame: np.ndarray[np.uint8]
) -> tuple[np.ndarray, list[float]]:
    """Creates a 2D reference to solve PnP problem from the position of the landmarks in a given frame.

    Args:
        landmarks (landmark_pb2.NormalizedLandmarkList): Normalized Landmarks object obtained in a given frame.

    Returns:
        tuple[np.ndarray, list[float]]: Face2D reference array and nose2D reference point.

    """
    height, width, _channels = frame.shape
    face_2d_ref = [
        [int(landmarks[1].x * width), int(landmarks[1].y * height)],
        [int(landmarks[199].x * width), int(landmarks[199].y * height)],
        [int(landmarks[33].x * width), int(landmarks[33].y * height)],
        [int(landmarks[263].x * width), int(landmarks[263].y * height)],
        [int(landmarks[61].x * width), int(landmarks[61].y * height)],
        [int(landmarks[291].x * width), int(landmarks[291].y * height)],
    ]
    nose_2d_ref = (landmarks[1].x * width, landmarks[1].y * height)

    return (np.array(face_2d_ref, dtype=np.float64), nose_2d_ref)


def create_3d_reference() -> tuple[np.array, np.array]:
    """Creates a 3D reference to solve PnP problemn from the position of the landmarks in a given frame.
    Returns:

        tuple[np.array, np.array]: Face3D reference array and nose3D reference array.

    """

    face_3d_ref = [
        [0.0, 0.0, 0.0],  # Tip of the Nose [mediapipe: 1]
        [0.0, -330.0, -65.0],  # Chin [mediapipe: 199]
        [-225.0, 170.0, -135.0],  # Left corner of the eye [mediapipe: 33]
        [225.0, 170.0, -135.0],  # Right corner of th eye [mediapipe: 263]
        [-150.0, -150.0, -125.0],  # Left corner of the mouth [mediapipe: 61]
        [150.0, -150.0, -125.0],  # Right corner of the mouth [mediapipe: 291]
    ]

    nose_3d_ref = np.float64(
        [
            [0, 0, 1000],  # Z axis
        ],
    )

    return (np.array(face_3d_ref, dtype=np.float64), nose_3d_ref)

