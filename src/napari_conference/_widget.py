"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import cv2
import napari
import numpy as np
import pyvirtualcam
from magicgui import magic_factory
from napari.qt.threading import thread_worker

global capturing
capturing = False

global cam
cam = None

global update_mode
update_mode = {"filter": "None"}

global trail_param
trail_param = 0.1

group_name = "webcam"


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
    # Make the virtual camera
    # cam = pyvirtualcam.Camera(width=2908, height=2032, fps=20)
    # cam = pyvirtualcam.Camera(width=960, height=540, fps=20)
    global frame_count, out_zarr

    out_zarr = None

    frame_count = 0
    max_frames = 1000

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

        cam.sleep_until_next_frame()

    @thread_worker(connect={"yielded": update_layer})
    def frame_updater():
        global update_mode, capturing, cam, trail_param
        global frame_count, out_zarr
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

            np_frame = np.array(frame)

            if prev_frame is None:
                prev_frame = np_frame

            # Apply a filter
            if update_mode["filter"] == "Blur":
                frame = cv2.blur(frame, (5, 5))
            elif update_mode["filter"] == "Laplacian":
                frame = cv2.Laplacian(frame, cv2.CV_8U)
            elif update_mode["filter"] == "Gameboy":
                frame = gameboy_filter(frame)

            # This adds trails if trail param is 1, then this will never update
            frame = np.array(
                prev_frame * trail_param + frame * (1.0 - trail_param),
                dtype=frame.dtype,
            )
            prev_frame = np.array(frame)

            yield frame

        cap.release()
        print("Capture device released.")

        # cam.close()
        # print("Virtual camera released.")

    return frame_updater()


@magic_factory(
    call_button="Update",
    dropdown={"choices": ["None", "Blur", "Laplacian", "Gameboy"]},
)
def conference_widget(
    viewer: "napari.viewer.Viewer",
    layer_name="Napari Conference",
    dropdown="None",
    running=False,
    trails_param=0.1,
):
    global update_mode, capturing, cam, trail_param

    capturing = running
    update_mode["filter"] = dropdown

    trail_param = trails_param

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
