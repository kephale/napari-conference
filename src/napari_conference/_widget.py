"""
Webcam filter application using napari for image processing and virtual camera output.
Provides real-time filters and effects for video conferencing.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import cv2
import napari
import numpy as np
import pyvirtualcam
from magicgui import magic_factory
from napari.qt.threading import thread_worker
import time

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class WebcamState:
    """Holds the state of the webcam application."""
    capturing: bool = False
    cam: Optional[pyvirtualcam.Camera] = None
    update_mode: Dict[str, str] = None
    trail_param: float = 0.1

    def __post_init__(self):
        if self.update_mode is None:
            self.update_mode = {"filter": "None"}

class ImageFilters:
    """Collection of image filtering methods."""
    
    @staticmethod
    def kaleidoscope_filter(image: np.ndarray) -> np.ndarray:
        """Apply kaleidoscope effect to the image.
        
        Args:
            image: Input image array (HxWx3)
            
        Returns:
            Processed image with kaleidoscope effect
        """
        h, w, _ = image.shape
        segment_size = (h // 4, w // 4)
        
        # Calculate segment coordinates
        start_y = h // 2 - segment_size[0] // 2
        end_y = start_y + segment_size[0]
        start_x = w // 2 - segment_size[1] // 2
        end_x = start_x + segment_size[1]
        
        # Create kaleidoscope pattern
        segment = image[start_y:end_y, start_x:end_x]
        hor_mirror = np.fliplr(segment)
        top = np.concatenate((segment, hor_mirror), axis=1)
        bottom = np.flipud(top)
        kaleidoscope = np.concatenate((top, bottom), axis=0)
        
        return cv2.resize(kaleidoscope, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def gameboy_filter(image: np.ndarray) -> np.ndarray:
        """Apply Gameboy-style retro filter to the image.
        
        Args:
            image: Input image array (HxWx3)
            
        Returns:
            Processed image with Gameboy effect
        """
        # Convert to grayscale and resize to Gameboy resolution
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 144), interpolation=cv2.INTER_LINEAR)
        
        # Create threshold layers
        _, black_and_white = cv2.threshold(small, 128, 255, cv2.THRESH_BINARY)
        _, dark_gray = cv2.threshold(small, 64, 255, cv2.THRESH_BINARY)
        _, light_gray = cv2.threshold(small, 192, 255, cv2.THRESH_BINARY)
        
        # Combine shades
        gameboy_gray = np.zeros_like(small, dtype=np.uint8)
        gameboy_gray += 255 - black_and_white
        gameboy_gray += (255 - dark_gray) // 2
        gameboy_gray += (255 - light_gray) // 3
        
        # Map to Gameboy color palette
        gameboy_color = np.zeros((small.shape[0], small.shape[1], 3), dtype=np.uint8)
        palette = {
            255: [155, 188, 15],  # Off-white
            170: [139, 172, 15],  # Light gray
            85: [48, 98, 48],     # Dark gray
            0: [15, 56, 15]       # Black
        }
        
        for gray_value, color in palette.items():
            gameboy_color[gameboy_gray == gray_value] = color
            
        return cv2.resize(gameboy_color, (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)

class WebcamProcessor:
    """Handles webcam capture and processing."""
    
    def __init__(self, state: WebcamState, viewer: napari.Viewer, layer_name: str):
        self.state = state
        self.viewer = viewer
        self.layer_name = layer_name
        self.filters = ImageFilters()

    def update_layer(self, new_frame: np.ndarray) -> None:
        """Update the napari layer with new frame data."""
        try:
            self.viewer.layers[self.layer_name].data = new_frame
        except KeyError:
            logger.info("Adding new layer")
            self.viewer.add_image(new_frame, name=self.layer_name)

        screen = self.viewer.screenshot(flash=False, canvas_only=False)
        self._update_virtual_camera(screen)

    def _update_virtual_camera(self, screen: np.ndarray) -> None:
        """Update virtual camera output."""
        new_size = (screen.shape[1], screen.shape[0])
        
        if self.state.cam is not None and new_size != (self.state.cam.width, self.state.cam.height):
            self.state.cam.close()
            self.state.cam = None

        if self.state.cam is None:
            logger.info("Initializing virtual camera")
            self.state.cam = pyvirtualcam.Camera(
                width=new_size[0], height=new_size[1], fps=20
            )

        self.state.cam.send(screen[:, :, :3])
        self.state.cam.sleep_until_next_frame()

    @thread_worker
    def frame_updater(self):
        """Capture and process webcam frames."""
        logger.info("Opening webcam")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            raise IOError("Cannot open webcam")

        prev_frame = None
        try:
            while self.state.capturing:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from webcam")
                    continue

                frame = self._process_frame(frame, prev_frame)
                prev_frame = np.array(frame)
                yield frame
                time.sleep(1/20)  # 20 FPS
        finally:
            cap.release()
            logger.info("Capture device released.")

    def _process_frame(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> np.ndarray:
        """Apply selected filters and effects to the frame."""
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame = frame[:, :, ::-1]  # BGR to RGB
        
        if prev_frame is None:
            prev_frame = np.array(frame)

        # Apply selected filter
        filter_map = {
            "Blur": lambda f: cv2.blur(f, (5, 5)),
            "Laplacian": lambda f: cv2.Laplacian(f, cv2.CV_8U),
            "Gameboy": self.filters.gameboy_filter,
            "Kaleidoscope": self.filters.kaleidoscope_filter
        }
        
        selected_filter = self.state.update_mode["filter"]
        if selected_filter in filter_map:
            frame = filter_map[selected_filter](frame)

        # Apply motion trail effect
        frame = np.array(
            prev_frame * self.state.trail_param + frame * (1.0 - self.state.trail_param),
            dtype=frame.dtype
        )
        
        return frame

@magic_factory(
    call_button="Update",
    dropdown={"choices": ["None", "Blur", "Laplacian", "Gameboy", "Kaleidoscope"]}
)
def conference_widget(
    viewer: "napari.viewer.Viewer",
    layer_name: str = "Napari Conference",
    dropdown: str = "None",
    running: bool = False,
    trails_param: float = 0.1,
) -> None:
    """Create and update the conference widget with selected parameters."""
    state.capturing = running
    state.update_mode["filter"] = dropdown
    state.trail_param = trails_param

    if state.cam is None and state.capturing:
        logger.info("Creating layer")
        processor = WebcamProcessor(state, viewer, layer_name)
        worker = processor.frame_updater()
        worker.yielded.connect(processor.update_layer)
        worker.start()

# Initialize global state
state = WebcamState()

__all__ = ["WebcamProcessor", "conference_widget", "ImageFilters", "WebcamState"]

if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.resize(800, 600)
    widget = conference_widget()
    viewer.window.add_dock_widget(widget, name="Conference")
    napari.run()