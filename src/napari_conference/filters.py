import numpy as np
import cv2

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
