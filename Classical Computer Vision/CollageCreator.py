import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.filters import gaussian

def CollageCreate(AddressofFolder):
    """
    - Randomly selects 5 frames from Folder
    - Fade beackground edges of each frame
    - Sort by a colour and edge
    - add lest variation in the centre and rest on the outside

    Sorting for middle 
    - Colour variation - standard deviation of the rgb pixel-intensity
      histogram higher std-dev = more colour diversity.


    Fade edges:
    - Create a binary mask where border pixels are 0 and interior pixels are 1.
    - gausasian blur it
    - try alpha masking
    """

    # get 5 random image
    valid_exts = ('.png', '.jpg')
    all_files  = [os.path.join(AddressofFolder, f)
                  for f in sorted(os.listdir(AddressofFolder))
                  if f.lower().endswith(valid_exts)]
    selected_paths = random.sample(all_files, 5)

    THUMB_H, THUMB_W = 270, 360

    def load(path, h, w):
        img = io.imread(path)
        img = (resize(img, (h, w), anti_aliasing=True) * 255).astype(np.uint8)
        return img[:, :, :3] if img.shape[2] == 4 else img

    # sort by colour variation
    def variation(img):
        hist, _ = np.histogram(img.reshape(-1, 3) / 255.0, bins=256, range=(0, 1))
        return np.std(hist / hist.sum())

    raw   = [load(p, THUMB_H, THUMB_W) for p in selected_paths]
    # order[0] = least varied (centre), order[4] = most varied (bottom)
    order = sorted(range(5), key=lambda i: variation(raw[i]))

    # dimensions - Left, Centre, Right all same width with overlaps
    FH, FV   = 80, 50  # fade px: horizontal, vertical
    CANVAS_W = THUMB_W * 3  # 1080
    SHARED_W = (CANVAS_W + 2 * FH) // 3  # Equal width for all 3 middle images
    SIDE_H   = int(THUMB_H * 1.5)  # 405

    # build binary mask for one image
    # inside is 1 (opaque), border strip is 0 (transparent)
    # Gaussian blur creates a soft gradual fade instead of a hard edge
    def alpha_mask(h, w, top, bottom, left, right):
        a = np.zeros((h, w), dtype=float)
        a[top:h-bottom, left:w-right] = 1.0      # inside = opaque
        # decrease sigma for faster blur
        a = gaussian(a, sigma=max(top, left)/1.3) # soft blur outward from edges
        return a[:, :, None]

    # adding img onto canvas at position (x, y)
    def paste(canvas, img, x, y, mask):
        h, w = img.shape[:2]
        bg = canvas[y:y+h, x:x+w].astype(float)
        canvas[y:y+h, x:x+w] = (bg*(1-mask) + img*mask).astype(np.uint8)

    # design layout
    mid_y    = THUMB_H  - FV
    bot_y    = mid_y + SIDE_H - FV
    centre_x = SHARED_W   - FH
    right_x  = centre_x + SHARED_W - FH

    # blank white canvas
    canvas = np.full((bot_y + THUMB_H, CANVAS_W, 3), 255, dtype=np.uint8)

    paste(canvas, load(selected_paths[order[1]], THUMB_H, CANVAS_W), 0,        0,     alpha_mask(THUMB_H, CANVAS_W, FV, FV, FH, FH))
    paste(canvas, load(selected_paths[order[2]], SIDE_H,  SHARED_W),   0,        mid_y, alpha_mask(SIDE_H,  SHARED_W,   FV, FV, FH, FH))
    paste(canvas, load(selected_paths[order[0]], SIDE_H,  SHARED_W), centre_x, mid_y, alpha_mask(SIDE_H,  SHARED_W, FV, FV, FH, FH))
    paste(canvas, load(selected_paths[order[3]], SIDE_H,  SHARED_W),   right_x,  mid_y, alpha_mask(SIDE_H,  SHARED_W,   FV, FV, FH, FH))
    paste(canvas, load(selected_paths[order[4]], THUMB_H, CANVAS_W), 0,        bot_y, alpha_mask(THUMB_H, CANVAS_W, FV, FV, FH, FH))

    plt.figure(figsize=(12, 8))
    plt.imshow(canvas)
    plt.axis('off')
    plt.title('Frame Collage')
    plt.tight_layout()
    plt.show()

    # save result
    io.imsave('task1_collage.png', canvas)

    return canvas