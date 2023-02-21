"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari


import cv2
import numpy as np

import napari
from napari.qt.threading import thread_worker
import pyvirtualcam
from magicgui import magic_factory

global capturing
capturing = False

global cam
cam = None

global update_mode
update_mode = {"filter": "None"}

global trail_param
trail_param = 0.1
    
def make_layer(layer_name="Conference", viewer=None):
    # Make the virtual camera
    # cam = pyvirtualcam.Camera(width=2908, height=2032, fps=20)
    # cam = pyvirtualcam.Camera(width=960, height=540, fps=20)

    # This function is called on every frame of the real webcam
    def update_layer(new_frame):
        global cam

        try:
            viewer.layers[layer_name].data = new_frame
        except KeyError:
            viewer.add_image(new_frame, name=layer_name)

        screen = viewer.screenshot(flash=False, canvas_only=False)
        if cam is None:
            cam = pyvirtualcam.Camera(
                width=screen.shape[1], height=screen.shape[0], fps=20
            )

        # Send to virtual camera
        # cam.send(screen)
        cam.send(screen[:, :, :3])
        # cam.sleep_until_next_frame()


    @thread_worker(connect={"yielded": update_layer})
    def frame_updater():
        global update_mode, capturing, cam, trail_param
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        prev_frame = None
        
        while capturing:
            ret, frame = cap.read()
            frame = cv2.resize(
                frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )
            frame = frame[:, :, ::-1]

            if prev_frame is None:
                prev_frame = np.array(frame)

            # frame = cv2.blur(frame, (5,5))

            if update_mode["filter"] == "Blur":
                frame = cv2.blur(frame, (5, 5))
            elif update_mode["filter"] == "Laplacian":
                frame = cv2.Laplacian(frame, cv2.CV_8U)

            elif update_mode["filter"] == "Canny":
                frame = cv2.Canny(frame, 100, 200)
            # elif update_mode["filter"] == "AddNoise":
            #     row, col, ch = frame.shape
            #     gauss = np.random.randn(row, col, ch)
            #     gauss = gauss.reshape(row, col, ch)
            #     frame = frame + frame * gauss

            # Make this an option
            frame = np.array(prev_frame * trail_param + frame * (1.0 - trail_param), dtype=frame.dtype)
            prev_frame = np.array(frame)
                
            yield frame
            
        cap.release()
        print("Capture device released.")

        # cam.close()
        # print("Virtual camera released.")

    return frame_updater()



@magic_factory(
    call_button="Update",
    dropdown={"choices": ["None", "Blur", "Laplacian"]},
)
def conference_widget(viewer: "napari.viewer.Viewer", layer_name="Napari Conference", dropdown="None", running=False, trails_param=0.1):
    global update_mode, capturing, cam, trail_param

    capturing = running
    update_mode["filter"] = dropdown

    trail_param = new_trail_param
    
    if cam is None:
        make_layer(layer_name, viewer=viewer)
    

if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.resize(800, 600)


    widget = conference_widget()
    
    viewer.window.add_dock_widget(widget, name="Conference")
    # widget_demo.show()

    worker = make_layer("Kyle", viewer=viewer)
    
    try:
        worker.start()
    except Exception:
        print("barf")


    napari.run()


"""
Unhandled:

 WARNING: Traceback (most recent call last):
  File "/Users/kharrington/nesoi/examples/webcam_fun.py", line 32, in update_layer
    screen = viewer.screenshot(flash=False, canvas_only=False)
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/viewer.py", line 129, in screenshot
    return self.window.screenshot(
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/_qt/qt_main_window.py", line 1350, in screenshot
    img = QImg2array(self._screenshot(size, scale, flash, canvas_only))
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/_qt/qt_main_window.py", line 1316, in _screenshot
    img = self._qt_window.grab().toImage()
RuntimeError: wrapped C/C++ object of type _QtMainWindow has been deleted
"""
