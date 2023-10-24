import cv2
import napari
import numpy as np
import pyvirtualcam
from magicgui import magic_factory
from napari.qt.threading import thread_worker

import time
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)


# Define a class to hold the state
class WebcamState:
    def __init__(self):
        self.capturing = False
        self.cam = None
        self.update_mode = {"filter": "None"}
        self.trail_param = 0.1


# Create an instance of the state
state = WebcamState()


def kaleidoscope_filter(image):
    h, w, _ = image.shape

    # Define the size of the central segment (for example, 1/4 of the image's height and width)
    segment_size = (h // 4, w // 4)

    # Calculate the starting and ending coordinates for the segment
    start_y = h // 2 - segment_size[0] // 2
    end_y = start_y + segment_size[0]
    start_x = w // 2 - segment_size[1] // 2
    end_x = start_x + segment_size[1]

    # Crop the central segment
    segment = image[start_y:end_y, start_x:end_x]

    # Mirror horizontally
    hor_mirror = np.fliplr(segment)

    # Concatenate horizontally mirrored segment with the original segment
    top = np.concatenate((segment, hor_mirror), axis=1)

    # Mirror the top vertically
    bottom = np.flipud(top)

    # Concatenate vertically to get the kaleidoscope pattern
    kaleidoscope = np.concatenate((top, bottom), axis=0)

    # Resize the kaleidoscope pattern to fit the entire image
    output = cv2.resize(kaleidoscope, (w, h), interpolation=cv2.INTER_NEAREST)

    return output


def gameboy_filter(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Downscale the image to Gameboy's resolution
    small = cv2.resize(gray, (160, 144), interpolation=cv2.INTER_LINEAR)

    # Threshold the image to get the 4 Gameboy shades
    _, black_and_white = cv2.threshold(small, 128, 255, cv2.THRESH_BINARY)
    _, dark_gray = cv2.threshold(small, 64, 255, cv2.THRESH_BINARY)
    _, light_gray = cv2.threshold(small, 192, 255, cv2.THRESH_BINARY)

    # Combine the shades
    gameboy_gray = np.zeros_like(small, dtype=np.uint8)
    gameboy_gray += 255 - black_and_white
    gameboy_gray += (255 - dark_gray) // 2
    gameboy_gray += (255 - light_gray) // 3

    # Map grayscale shades to Gameboy greenish palette
    gameboy_color = np.zeros(
        (small.shape[0], small.shape[1], 3), dtype=np.uint8
    )

    gameboy_color[gameboy_gray == 255] = [155, 188, 15]  # Off-white color
    gameboy_color[gameboy_gray == 170] = [139, 172, 15]  # Light gray
    gameboy_color[gameboy_gray == 85] = [48, 98, 48]  # Dark gray
    gameboy_color[gameboy_gray == 0] = [15, 56, 15]  # Black

    # Resize to original image size
    output = cv2.resize(
        gameboy_color,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return output


def make_layer(layer_name="Conference", viewer=None):
    global state
    LOGGER.info("Entering make_layer function")

    def update_layer(new_frame):
        global state
        try:
            viewer.layers[layer_name].data = new_frame
        except KeyError:
            LOGGER.info("Adding new layer")
            viewer.add_image(new_frame, name=layer_name)

        screen = viewer.screenshot(flash=False, canvas_only=False)

        # Close and reopen the camera with the new frame size if it has changed
        if state.cam is not None and (screen.shape[1], screen.shape[0]) != (
            state.cam.width,
            state.cam.height,
        ):
            state.cam.close()
            state.cam = None

        if state.cam is None:
            LOGGER.info("Initializing virtual camera")
            state.cam = pyvirtualcam.Camera(
                width=screen.shape[1], height=screen.shape[0], fps=20
            )

        state.cam.send(screen[:, :, :3])
        state.cam.sleep_until_next_frame()

    @thread_worker
    def frame_updater():
        global state
        LOGGER.info("Opening webcam")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            LOGGER.error("Cannot open webcam")
            raise IOError("Cannot open webcam")

        prev_frame = None
        LOGGER.info("Starting frame capturing loop")
        while state.capturing:
            # LOGGER.info("Reading frame from webcam")
            ret, frame = cap.read()
            if not ret:
                LOGGER.error("Failed to read frame from webcam")
                continue

            frame = cv2.resize(
                frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )
            frame = frame[:, :, ::-1]
            np_frame = np.array(frame)
            if prev_frame is None:
                prev_frame = np_frame

            if state.update_mode["filter"] == "Blur":
                frame = cv2.blur(frame, (5, 5))
            elif state.update_mode["filter"] == "Laplacian":
                frame = cv2.Laplacian(frame, cv2.CV_8U)
            elif state.update_mode["filter"] == "Gameboy":
                frame = gameboy_filter(frame)
            elif state.update_mode["filter"] == "Kaleidoscope":
                frame = kaleidoscope_filter(frame)

            frame = np.array(
                prev_frame * state.trail_param
                + frame * (1.0 - state.trail_param),
                dtype=frame.dtype,
            )
            prev_frame = np.array(frame)
            yield frame

            # Sleep for 20FPS
            time.sleep(1 / 20)
        cap.release()
        LOGGER.info("Capture device released.")

    LOGGER.info("Starting frame updater worker")
    worker = frame_updater()
    worker.yielded.connect(update_layer)
    worker.start()

    return worker


@magic_factory(
    call_button="Update",
    dropdown={
        "choices": ["None", "Blur", "Laplacian", "Gameboy", "Kaleidoscope"]
    },
)
def conference_widget(
    viewer: "napari.viewer.Viewer",
    layer_name="Napari Conference",
    dropdown="None",
    running=False,
    trails_param=0.1,
):
    global state

    state.capturing = running
    state.update_mode["filter"] = dropdown
    state.trail_param = trails_param

    if state.cam is None and state.capturing:
        LOGGER.info("Creating layer")
        make_layer(layer_name, viewer=viewer)


if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.resize(800, 600)

    widget = conference_widget()

    viewer.window.add_dock_widget(widget, name="Conference")

    napari.run()
