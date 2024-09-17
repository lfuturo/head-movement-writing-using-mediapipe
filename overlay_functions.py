import cv2
import numpy as np

overlays = [
    [["Q", "W", "E"], ["A", "S", "D"], ["Z", "X", "C"]],
    [["R", "T", "Y"], ["F", "G", "H"], ["V", "B", "N"]],
    [["U", "I", "O"], ["P", "J", "K"], ["L", "M", "~N"]],
]

quadrant_labels = [["UL", "UM", "UR"], ["ML", "MM", "MR"], ["DL", "DM", "DR"]]

key_mapping = {
    ("UL", 1): "Q",
    ("UM", 1): "W",
    ("UR", 1): "E",
    ("ML", 1): "A",
    ("MM", 1): "S",
    ("MR", 1): "D",
    ("DL", 1): "Z",
    ("DM", 1): "X",
    ("DR", 1): "C",
    ("UL", 2): "R",
    ("UM", 2): "T",
    ("UR", 2): "Y",
    ("ML", 2): "F",
    ("MM", 2): "G",
    ("MR", 2): "H",
    ("DL", 2): "V",
    ("DM", 2): "B",
    ("DR", 2): "N",
    ("UL", 3): "U",
    ("UM", 3): "I",
    ("UR", 3): "O",
    ("ML", 3): "P",
    ("MM", 3): "J",
    ("MR", 3): "K",
    ("DL", 3): "L",
    ("DM", 3): "M",
    ("DR", 3): "Ñ",
    ("Space", 1): " ",
    ("Space", 2): " ",
    ("Space", 3): " ",
}


def draw_grid(
    frame: np.ndarray[np.uint8],
    overlay: list[list[str]],
    screen_width: int,
    screen_height: int,
) -> np.ndarray[np.uint8]:
    """
    Draws a grid on the camera frame with content from a provided overlay.

    Args:
        frame (np.ndarray[np.uint8]): The camera frame where the grid will be drawn.
        overlay (list[list[str]]): Matrix of characters that defines the content to display in each cell of the grid.
        screen_width (int): Width of the screen.
        screen_height (int): Height of the screen.

    Returns:
        np.ndarray[np.uint8]: The camera frame with the grid and overlay content drawn on it.
    """
    rows, cols = len(overlay), len(overlay[0])
    grid_width = screen_width // cols
    grid_height = screen_height // rows

    for i in range(rows):
        for j in range(cols):
            x = j * grid_width
            y = i * grid_height

            cv2.rectangle(
                frame, (x, y), (x + grid_width, y + grid_height), (255, 255, 255), 2
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            thickness = 3
            text_size = cv2.getTextSize(overlay[i][j], font, font_scale, thickness)[0]
            text_x = x + (grid_width - text_size[0]) // 2
            text_y = y + (grid_height + text_size[1]) // 2
            cv2.putText(
                frame,
                overlay[i][j],
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
            )

    return frame


def detect_quadrant(
    nose_end_point: tuple[int, int], screen_width: int, screen_height: int
) -> str:
    """
    Detects which quadrant the nose end point falls into on the screen grid.

    Args:
        nose_end_point (tuple[int, int]): The (x, y) coordinates of the nose end point projected on the screen.
        screen_width (int): The width of the screen.
        screen_height (int): The height of the screen.

    Returns:
        str: The label of the quadrant where the nose end point is located.
    """

    cols = 3
    rows = 3
    grid_width = screen_width // cols
    grid_height = screen_height // rows

    x, y = nose_end_point

    col = min(int(x // grid_width), cols - 1)
    row = min(int(y // grid_height), rows - 1)

    quadrant_label = quadrant_labels[row][col]

    return quadrant_label
