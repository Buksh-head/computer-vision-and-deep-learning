import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage import io, color, feature, draw
from skimage.util import img_as_float
from scipy.ndimage import distance_transform_edt, correlate

# Constants
CIRCLE_RADIUS_OFFSET = 2
RECTANGLE_HEIGHT_SCALE = 0.25
RECTANGLE_ASPECT_RATIO = 1.6
POLYGON_SIZE_OFFSET = 2
OUTPUT_DPI = 120

# Template rotation angles
TEMPLATE_ROTATIONS = {
    "square": 45,
    "octagon": 22.5,
    "triangle": 90,
    "diamond": 0,
}

# Colors for visualization
DETECTION_COLORS = ["red", "lime", "cyan", "yellow", "magenta"]

def load_grey(path):
    img = io.imread(path)
    img = img_as_float(img)

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = color.rgb2gray(img)

    return img

# Template generation functions
def make_circle(size):
    mask = np.zeros((size, size), dtype=bool)
    cx, cy = size // 2, size // 2
    r = size // 2 - CIRCLE_RADIUS_OFFSET
    rr, cc = draw.circle_perimeter(cx, cy, r, shape=mask.shape)
    mask[rr, cc] = True
    return mask

def make_polygon(size, n_sides, rotation_deg=0):
    mask = np.zeros((size, size), dtype=bool)
    cx, cy = size / 2, size / 2
    r = size / 2 - POLYGON_SIZE_OFFSET

    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    angles += np.deg2rad(rotation_deg)

    pts = np.array([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles])

    for i in range(n_sides):
        row0, col0 = int(pts[i][1]), int(pts[i][0])
        row1, col1 = int(pts[(i + 1) % n_sides][1]), int(pts[(i + 1) % n_sides][0])
        rr, cc = draw.line(row0, col0, row1, col1)

        valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        mask[rr[valid], cc[valid]] = True

    return mask

def make_rectangle(size, aspect=RECTANGLE_ASPECT_RATIO):
    """Generate rectangle outline template."""
    mask = np.zeros((size, size), dtype=bool)

    h = int(size * RECTANGLE_HEIGHT_SCALE)
    w = int(size * RECTANGLE_HEIGHT_SCALE * aspect)

    cy, cx = size // 2, size // 2
    r0, r1 = cy - h, cy + h
    c0, c1 = cx - w, cx + w

    r0, r1 = max(0, r0), min(size - 1, r1)
    c0, c1 = max(0, c0), min(size - 1, c1)

    mask[r0:r1 + 1, c0] = True
    mask[r0:r1 + 1, c1] = True
    mask[r0, c0:c1 + 1] = True
    mask[r1, c0:c1 + 1] = True

    return mask

def _create_templates():
    return {
        "circle": make_circle,
        "square": lambda s: make_polygon(s, 4, rotation_deg=TEMPLATE_ROTATIONS["square"]),
        "rectangle": make_rectangle,
        "octagon": lambda s: make_polygon(s, 8, rotation_deg=TEMPLATE_ROTATIONS["octagon"]),
        "triangle": lambda s: make_polygon(s, 3, rotation_deg=TEMPLATE_ROTATIONS["triangle"]),
        "diamond": lambda s: make_polygon(s, 4, rotation_deg=TEMPLATE_ROTATIONS["diamond"]),
    }

TEMPLATES = _create_templates()

# chamfer matching 
def match_template(dt, template):
    # use correlate to compute the sum of DT values under the template at every position
    # dividing by n gives the mean DT — lower mean = template edges closer to real edges
    kernel = template.astype(float)
    n = kernel.sum()

    if n == 0:
        return np.full(dt.shape, np.inf)

    score = correlate(dt, kernel, mode="constant", cval=np.max(dt))
    return score / n


def chamfer_match(dt, scales):
    best_per_shape = {}

    for name, template_fn in TEMPLATES.items():
        best_match = (np.inf, name, 0, 0, 0)

        # test each scale — largest first since signs tend to dominate the image
        for size in scales:
            template = template_fn(size)
            score_map = match_template(dt, template)

            # find the position with the lowest mean DT score for this size
            min_idx = np.unravel_index(np.argmin(score_map), score_map.shape)
            min_score = score_map[min_idx]

            if min_score < best_match[0]:
                best_match = (min_score, name, min_idx[0], min_idx[1], size)

        best_per_shape[name] = best_match
        print(
            f"{name:10s}: score={best_match[0]:7.2f} "
            f"at ({best_match[2]:4d},{best_match[3]:4d}) size={best_match[4]:3d}"
        )

    return best_per_shape

# display 
def show_all(original, edges, dt, output_file="sign_edges_dt.png"):
    """Display original, edges, and distance transform."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original (grayscale)")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny edges")
    axes[1].axis("off")

    axes[2].imshow(dt / dt.max(), cmap="hot")
    axes[2].set_title("Distance transform")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.show()

# show detections
def show_detections(img_grey, best_per_shape, top_n=3, output_file="sign_detections.png"):
    """Visualize top N detections on the image."""
    all_matches = sorted(best_per_shape.values(), key=lambda x: x[0])[:top_n]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_grey, cmap="gray")

    for i, (score, name, row, col, size) in enumerate(all_matches):
        half_size = size // 2
        color = DETECTION_COLORS[i % len(DETECTION_COLORS)]

        rect = patches.Rectangle(
            (col - half_size, row - half_size),
            size,
            size,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            col - half_size,
            row - half_size - 5,
            f"{name} ({score:.1f})",
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.7, pad=2)
        )

    ax.set_title(f"Top {top_n} detections", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.show()

    print(f"Saved → {output_file}")

def SignRecognition(image_path):
    img = load_grey(image_path)
    h, w = img.shape

    edges = feature.canny(
        img,
        sigma=0.5,
        low_threshold=0.85,
        high_threshold=0.95,
        use_quantiles=True
    )

    # distance transform — low values near edges, high values away from edges
    dt = distance_transform_edt(~edges)

    show_all(img, edges, dt)

    max_scale = min(h, w) // 2
    scales = list(range(max_scale, 70, -5)) 

    best = chamfer_match(dt, scales)
    show_detections(img, best, top_n=3)
