import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, feature, morphology
from skimage.util import img_as_float
from skimage.restoration import denoise_bilateral
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans


# Load image
def load_rgb(path):
    img = io.imread(path)
    # grayscale to RGB
    if img.ndim == 2:
        img = color.gray2rgb(img)
    # RGBA to RGB
    if img.shape[2] == 4:
        img = color.rgba2rgb(img)
    return img_as_float(img)


# Display and save image to file
def save_and_show(img, title, out_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(img, 0, 1))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")


# implementation 1: K-means
def cartoonize_kmeans(img, k=18):
    h, w, _ = img.shape
    # Reshape image into list of RGB pixels
    pixels = img.reshape(-1, 3)
    # Cluster pixels into k groups based on color similarity
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(pixels)
    # Replace each pixel with its cluster center color
    quantized = kmeans.cluster_centers_[labels].reshape(h, w, 3)
    return np.clip(quantized, 0, 1)


# implementation 2: Bilateral filtering
def cartoonize_bilateral(img, sigma_color=0.05, sigma_spatial=1, n_iter=3):
    # Apply bilateral filter multiple times for cumulative effect
    for _ in range(n_iter):
        # Process each RGB channel separately
        for ch in range(3):
            # Bilateral filter: smooths similar colors while preserving edges
            img[:, :, ch] = denoise_bilateral(
                img[:, :, ch],
                sigma_color=sigma_color, # Higher sigma_color = more aggressive smoothing across color differences
                sigma_spatial=sigma_spatial, # Higher sigma_spatial = more distances
                channel_axis=None
            )
    return np.clip(img, 0, 1)


# implementation 3: skin segmentation + K-means foreground + blurred background + saturation boost
def cartoonize_improved(img, k=24, saturation=1.4):
    h, w, _ = img.shape

    # convert to HSV — easier to isolate skin tones than in RGB
    hsv = color.rgb2hsv(img)
    hch, sch, vch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # skin pixels have a warm reddish hue (0–10° or 350–360°),
    # moderate saturation and brightness
    skin_mask = (
        (((hch >= 0.0) & (hch <= 0.10)) | ((hch >= 0.90) & (hch <= 1.0))) &
        (sch >= 0.15) & (sch <= 0.8) &
        (vch >= 0.2)  & (vch <= 1.0)
    )

    # blur the mask itself so the skin/background boundary fades gradually
    # instead of having a hard cut between the two regions
    soft_mask = gaussian_filter(skin_mask.astype(float), sigma=8)
    soft_mask = soft_mask[:, :, None]   # add channel dim for broadcasting over RGB

    # K-means quantises the image into k flat colours for the cartoon foreground
    # higher k keeps more colour detail, lower k gives a stronger poster effect
    pixels    = img.reshape(-1, 3)
    kmeans    = KMeans(n_clusters=k, random_state=1, n_init=10)
    labels    = kmeans.fit_predict(pixels)
    quantized = kmeans.cluster_centers_[labels].reshape(h, w, 3)
    quantized = denoise_bilateral(quantized, sigma_color=0.05, sigma_spatial=1, channel_axis=-1)

    # background is just the original image blurred — keeps scene context
    # but draws focus toward the subject
    background = gaussian_filter(img, sigma=(6.0, 6.0, 0))

    # soft_mask = 1 over skin → quantized shows through
    # soft_mask = 0 elsewhere → blurred background shows through
    result = quantized * soft_mask + background * (1.0 - soft_mask)

    # blending washes out colours slightly so boost saturation to compensate
    hsv_result = color.rgb2hsv(result)
    hsv_result[:, :, 1] = np.clip(hsv_result[:, :, 1] * saturation, 0, 1)
    result = color.hsv2rgb(hsv_result)

    return np.clip(result, 0, 1)


def CartoonNizer(image_path):
    img = load_rgb(image_path)

    # METHOD 1: K-means
    result_km = cartoonize_kmeans(img)
    save_and_show(result_km, "K-means Cartoon", "task2_kmeans.png")

    # METHOD 2: Bilateral
    result_bi = cartoonize_bilateral(img.copy())
    save_and_show(result_bi, "Bilateral Cartoon", "task2_bilateral.png")

    # METHOD 3: Improved
    result_im = cartoonize_improved(img.copy())
    save_and_show(result_im, "Improved Cartoon", "task2_improved.png")

    # Create comparison figure showing all methods side by side
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, im, t in zip(axes,
                         [img, result_km, result_bi, result_im],
                         ["Original", "K-means", "Bilateral", "Improved"]):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    # Save comparison image
    plt.savefig("task2_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()