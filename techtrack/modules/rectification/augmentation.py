import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

def horizontal_flip(image):
    return cv2.flip(image, 1)

def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def resize(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def color_jitter(image, brightness=0, contrast=0, saturation=0):
    # Convert to PIL Image for easier enhancement
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if brightness > 0:
        enhancer = ImageEnhance.Brightness(img)
        factor = 1 + random.uniform(-brightness, brightness)
        img = enhancer.enhance(factor)

    if contrast > 0:
        enhancer = ImageEnhance.Contrast(img)
        factor = 1 + random.uniform(-contrast, contrast)
        img = enhancer.enhance(factor)

    if saturation > 0:
        enhancer = ImageEnhance.Color(img)
        factor = 1 + random.uniform(-saturation, saturation)
        img = enhancer.enhance(factor)

    # Convert back to OpenCV image
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def random_crop(image, crop_size):
    h, w = image.shape[:2]
    ch, cw = crop_size

    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than the image size.")

    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)

    return image[y:y+ch, x:x+cw]