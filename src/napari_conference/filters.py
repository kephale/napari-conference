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

def sketch_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def neon_filter(image):
    edges = cv2.Canny(image, 100, 200)
    neon = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    neon = cv2.addWeighted(image, 0.5, neon, 0.5, 0)
    return neon

def thermal_filter(image):
    thermal = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return thermal

def vhs_filter(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    vhs = cv2.add(image, noise)
    return cv2.cvtColor(vhs, cv2.COLOR_BGR2GRAY)

def glitch_filter(image):
    rows, cols, _ = image.shape
    glitch = np.copy(image)
    num_swaps = np.random.randint(5, 20)
    
    for _ in range(num_swaps):
        # Define the first random region
        x1 = np.random.randint(0, cols)
        y1 = np.random.randint(0, rows)
        
        max_w1 = min(50, cols - x1)
        max_h1 = min(50, rows - y1)
        
        if max_w1 <= 5 or max_h1 <= 5:
            continue
        
        w1 = np.random.randint(5, max_w1)
        h1 = np.random.randint(5, max_h1)
        
        # Ensure the second region is within bounds and of the same size
        x2 = np.random.randint(0, cols - w1)
        y2 = np.random.randint(0, rows - h1)
        
        # Extract the regions
        region1 = glitch[y1:y1+h1, x1:x1+w1].copy()
        region2 = glitch[y2:y2+h1, x2:x2+w1].copy()
        
        # Swap the regions
        glitch[y1:y1+h1, x1:x1+w1] = region2
        glitch[y2:y2+h1, x2:x2+w1] = region1
        
    return glitch
