import cv2
import napari
import numpy as np
import yaml
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari_conference.filters import kaleidoscope_filter, gameboy_filter, sketch_filter, neon_filter, thermal_filter, vhs_filter, glitch_filter
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from qtpy import QtCore
import time
import logging
import os

from napari.layers import Shapes

# Configure logging
logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)

# Define a class to hold the state
class WebcamState:
    def __init__(self):
        self.capturing = False
        self.video_capturing = False
        self.update_mode = {"filter": "None"}
        self.trail_param = 0.1
        self.media_files = []
        self.media_descriptions = []
        self.current_media_index = 0
        self.webcam_shown = True
        self.showing_image = False
        self.current_video_worker = None
        self.picture_in_picture = False
        self.layout_config = 1

# Create an instance of the state
state = WebcamState()

def load_media_files(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('media_files', [])

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
        ret, frame = cap.read()
        if not ret:
            LOGGER.error("Failed to read frame from webcam")
            continue

        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, ::-1]  # Convert BGR to RGB
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
        elif state.update_mode["filter"] == "Sketch":
            frame = sketch_filter(frame)
        elif state.update_mode["filter"] == "Neon":
            frame = neon_filter(frame)
        elif state.update_mode["filter"] == "Thermal":
            frame = thermal_filter(frame)
        elif state.update_mode["filter"] == "VHS":
            frame = vhs_filter(frame)
        elif state.update_mode["filter"] == "Glitch":
            frame = glitch_filter(frame)            

        frame = np.array(
            prev_frame * state.trail_param
            + frame * (1.0 - state.trail_param),
            dtype=frame.dtype,
        )
        prev_frame = np.array(frame)
        yield frame

        time.sleep(1 / 20)

    cap.release()
    LOGGER.info("Capture device released.")

@thread_worker
def video_frame_updater(viewer):
    global state
    media_file = state.media_files[state.current_media_index]
    LOGGER.info(f"Opening media file: {media_file}")

    if media_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
        state.showing_image = True
        image = cv2.imread(media_file)
        if image is None:
            LOGGER.error("Cannot open image file")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image
        while state.video_capturing and state.showing_image:
            time.sleep(0.1)  # Keep the image in view
    else:
        state.showing_image = False
        cap = cv2.VideoCapture(media_file)
        if not cap.isOpened():
            LOGGER.error("Cannot open video file")
            raise IOError("Cannot open video file")

        LOGGER.info("Starting video frame capturing loop")
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = original_width / original_height
        new_height = 480
        new_width = int(new_height * aspect_ratio)

        while state.video_capturing:
            ret, frame = cap.read()
            if not ret:
                LOGGER.info("End of video file, looping back")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
                continue

            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frame = frame[:, :, ::-1]  # Convert BGR to RGB
            yield frame

            time.sleep(1 / 20)

        cap.release()
        LOGGER.info("Video file capture device released.")


def update_layer(viewer, layer_name, frame, set_active=False):
    if layer_name not in viewer.layers:
        viewer.add_image(frame, name=layer_name)
        configure_layout(viewer)
    else:
        viewer.layers[layer_name].data = frame
    if set_active:
        viewer.layers.selection.active = viewer.layers[layer_name]

def adjust_zoom(viewer):
    layer = viewer.layers["media"]
    viewer.camera.center = (layer.extent.world[1][0] + layer.extent.world[1][1]) / 2, (layer.extent.world[0][0] + layer.extent.world[0][1]) / 2
    viewer.camera.zoom = 0.5 * max(layer.extent.world[1][1] - layer.extent.world[1][0], layer.extent.world[0][1] - layer.extent.world[0][0])

def show_next_media(viewer):
    if state.current_video_worker is not None:
        state.current_video_worker.quit()
    state.video_capturing = False
    state.showing_image = False
    state.current_media_index = (state.current_media_index + 1) % len(state.media_files)
    state.video_capturing = True
    video_worker = video_frame_updater(viewer)
    video_worker.yielded.connect(lambda frame: update_layer(viewer, "media", frame, set_active=True))
    video_worker.start()
    state.current_video_worker = video_worker
    if state.picture_in_picture:
        rescale_webcam(viewer, adjust_zoom_p=False)
    update_text_layer(viewer)

def show_previous_media(viewer):
    if state.current_video_worker is not None:
        state.current_video_worker.quit()
    state.video_capturing = False
    state.showing_image = False
    state.current_media_index = (state.current_media_index - 1) % len(state.media_files)
    state.video_capturing = True
    video_worker = video_frame_updater(viewer)
    video_worker.yielded.connect(lambda frame: update_layer(viewer, "media", frame, set_active=True))
    video_worker.start()
    state.current_video_worker = video_worker
    if state.picture_in_picture:
        rescale_webcam(viewer, adjust_zoom_p=False)
    update_text_layer(viewer)

def toggle_webcam_layer(viewer):
    state.webcam_shown = not state.webcam_shown
    if state.webcam_shown:
        viewer.layers["Kyle Harrington"].visible = True
    else:
        viewer.layers["Kyle Harrington"].visible = False

def rescale_webcam(viewer, adjust_zoom_p=True):
    layer = viewer.layers["Kyle Harrington"]
    media_layer = viewer.layers["media"]
    media_extent = media_layer.extent
    media_width = media_extent[1][1] - media_extent[1][0]
    media_height = media_extent[0][1] - media_extent[0][0]

    if state.layout_config == 1:
        width_ratio = media_height / layer.data.shape[0]
        layer.scale = (width_ratio, width_ratio)
    else:
        layer.scale = (0.125, 0.125)

    if adjust_zoom_p:
        adjust_zoom(viewer)

def render_text_to_image(text, layout_config, total_width=800, height=200, font_scale=1, thickness=2, padding=20):
    """Render text to an image using OpenCV with padding and multiline support."""

    # Adjust width based on layout configuration
    if layout_config == 1:  # Side-by-side layout
        adjusted_width = total_width * 2  # Assuming half the width is allocated for text
    else:  # Picture-in-picture layout
        adjusted_width = total_width - padding * 2  # Full width minus padding

    # Split text into lines based on word wrapping
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = text.split(' ')
    lines = []
    current_line = ''

    for word in words:
        if current_line == '':
            current_line = word
        else:
            text_size = cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0]
            if text_size[0] + 2 * padding <= adjusted_width:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word

    if current_line:
        lines.append(current_line)

    max_text_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines])
    max_text_height = max([cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in lines])

    # Calculate required image size
    img_width = min(max_text_width + 2 * padding, adjusted_width)
    img_height = (max_text_height + padding) * len(lines) + padding

    # Create a blank image
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Draw each line of text
    y = padding + max_text_height
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = (img_width - text_size[0]) // 2
        cv2.putText(img, line, (text_x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += max_text_height + padding

    return img

def update_text_layer(viewer):
    description = state.media_descriptions[state.current_media_index]

    # Render text to an image with padding and multiline support, based on layout configuration
    text_image = render_text_to_image(description, state.layout_config)

    # Update or add the text image layer
    if "Text Layer" in viewer.layers:
        viewer.layers["Text Layer"].data = text_image
    else:
        viewer.add_image(text_image, name="Text Layer")

    # Position the text image layer based on the layout configuration
    if state.layout_config == 1:
        viewer.layers["Text Layer"].translate = [540, 0]
    else:
        viewer.layers["Text Layer"].translate = [540, 0]

def toggle_layout(viewer):
    state.layout_config = 3 - state.layout_config  # Toggle between 1 and 2
    configure_layout(viewer)

def configure_layout(viewer):
    if "Kyle Harrington" in viewer.layers and "media" in viewer.layers and "Text Layer" in viewer.layers:
        if state.layout_config == 1:
            viewer.layers["Kyle Harrington"].translate = [0, 0]
            viewer.layers["Kyle Harrington"].scale = (1, 1)
            viewer.layers["media"].translate = [0, 640]
            viewer.layers["Text Layer"].translate = [540, 0]
        else:
            viewer.layers["Kyle Harrington"].translate = [0, 0]
            viewer.layers["Kyle Harrington"].scale = (0.25, 0.25)
            viewer.layers["media"].translate = [0, 0]
            viewer.layers["Text Layer"].translate = [540, 0]

        viewer.camera.center = (0.0, 295.9729546866133, 639.9765603361523)
        viewer.camera.zoom = 0.6090584267788942
            
        if state.picture_in_picture:
            rescale_webcam(viewer, adjust_zoom_p=False)
        update_text_layer(viewer)

def add_buttons_and_keybindings(viewer):
    widget = QWidget()
    layout = QVBoxLayout()

    next_button = QPushButton("Next Media")
    next_button.clicked.connect(lambda: show_next_media(viewer))
    layout.addWidget(next_button)

    prev_button = QPushButton("Previous Media")
    prev_button.clicked.connect(lambda: show_previous_media(viewer))
    layout.addWidget(prev_button)

    toggle_button = QPushButton("Toggle Webcam")
    toggle_button.clicked.connect(lambda: toggle_webcam_layer(viewer))
    layout.addWidget(toggle_button)

    rescale_button = QPushButton("Rescale Webcam")
    rescale_button.clicked.connect(lambda: rescale_webcam(viewer))
    layout.addWidget(rescale_button)

    toggle_layout_button = QPushButton("Toggle Layout")
    toggle_layout_button.clicked.connect(lambda: toggle_layout(viewer))
    layout.addWidget(toggle_layout_button)

    widget.setLayout(layout)
    viewer.window.add_dock_widget(widget, area="right")

    viewer.bind_key("n", lambda _: show_next_media(viewer))
    viewer.bind_key("p", lambda _: show_previous_media(viewer))
    viewer.bind_key("t", lambda _: toggle_webcam_layer(viewer))
    viewer.bind_key("r", lambda _: rescale_webcam(viewer))
    viewer.bind_key("l", lambda _: toggle_layout(viewer))

@magic_factory(
    call_button="Update",
    dropdown={
        "choices": ["None", "Blur", "Laplacian", "Gameboy", "Kaleidoscope", "Sketch", "Neon", "Thermal", "Glitch"]
    },
)
def conference_widget(
    viewer: "napari.viewer.Viewer",
    dropdown="None",
    running=False,
    video_running=False,
    trails_param=0.1,
    yaml_file="podcast_media_files_v02.yaml",
):
    global state

    state.capturing = running
    state.video_capturing = video_running
    state.update_mode["filter"] = dropdown
    state.trail_param = trails_param

    media_data = load_media_files(yaml_file)
    state.media_files = [item['file'] for item in media_data]
    state.media_descriptions = [item['description'] for item in media_data]

    if state.capturing:
        LOGGER.info("Creating webcam layer")
        worker = frame_updater()
        worker.yielded.connect(lambda frame: update_layer(viewer, "Kyle Harrington", frame))
        worker.start()

    if state.video_capturing and state.media_files:
        if state.current_video_worker is not None:
            state.current_video_worker.quit()
        LOGGER.info("Creating video layer")
        video_worker = video_frame_updater(viewer)
        video_worker.yielded.connect(lambda frame: update_layer(viewer, "media", frame, set_active=False))
        video_worker.start()
        state.current_video_worker = video_worker

    update_text_layer(viewer)

    for dock_widget in list(viewer.window._dock_widgets.values()) + [viewer.window.qt_viewer.dockLayerList, viewer.window.qt_viewer.dockLayerControls]:
        dock_widget.setFloating(True)

if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.resize(800, 600)

    widget = conference_widget()
    add_buttons_and_keybindings(viewer)

    viewer.window.add_dock_widget(widget, name="Conference")

    napari.run()
