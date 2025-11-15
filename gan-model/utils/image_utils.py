"""
Image preprocessing and utility functions for GAN model.
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Union


def load_image(image_path: Union[str, Path], target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (height, width)
    
    Returns:
        Preprocessed image as numpy array normalized to [-1, 1]
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img)
    
    # Normalize to [-1, 1]
    img_array = (img_array - 127.5) / 127.5
    
    return img_array


def load_images_from_directory(
    directory: Union[str, Path],
    target_size: Tuple[int, int] = (64, 64),
    max_images: int = None
) -> np.ndarray:
    """
    Load multiple images from a directory.
    
    Args:
        directory: Path to directory containing images
        target_size: Target size for all images
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        Array of preprocessed images
    """
    directory = Path(directory)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    image_files = [
        f for f in directory.iterdir()
        if f.suffix.lower() in valid_extensions
    ]
    
    if max_images:
        image_files = image_files[:max_images]
    
    images = []
    for img_file in image_files:
        try:
            img = load_image(img_file, target_size)
            images.append(img)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    return np.array(images)


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [-1, 1] to [0, 255].
    
    Args:
        image: Image array in range [-1, 1]
    
    Returns:
        Image array in range [0, 255]
    """
    image = (image + 1) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def save_image(image: np.ndarray, save_path: Union[str, Path]) -> None:
    """
    Save a generated image to disk.
    
    Args:
        image: Image array (normalized or denormalized)
        save_path: Path to save the image
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if image needs denormalization
    if image.min() < 0 or image.max() <= 1:
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = denormalize_image(image)
    
    img = Image.fromarray(image)
    img.save(save_path)


def augment_image(image: np.ndarray, augmentation_type: str = 'flip') -> np.ndarray:
    """
    Apply data augmentation to an image.
    
    Args:
        image: Input image
        augmentation_type: Type of augmentation ('flip', 'rotate', 'brightness')
    
    Returns:
        Augmented image
    """
    if augmentation_type == 'flip':
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
    
    elif augmentation_type == 'rotate':
        angle = np.random.randint(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    elif augmentation_type == 'brightness':
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, -1, 1)
    
    return image


def create_image_grid(
    images: np.ndarray,
    grid_size: Tuple[int, int] = (4, 4),
    padding: int = 2
) -> np.ndarray:
    """
    Create a grid of images for visualization.
    
    Args:
        images: Array of images
        grid_size: Grid dimensions (rows, cols)
        padding: Padding between images in pixels
    
    Returns:
        Single image containing the grid
    """
    n_rows, n_cols = grid_size
    n_images = min(len(images), n_rows * n_cols)
    
    # Get image dimensions
    img_h, img_w = images[0].shape[:2]
    channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
    
    # Create canvas
    grid_h = n_rows * img_h + (n_rows - 1) * padding
    grid_w = n_cols * img_w + (n_cols - 1) * padding
    
    if channels == 1:
        grid = np.ones((grid_h, grid_w)) * 255
    else:
        grid = np.ones((grid_h, grid_w, channels)) * 255
    
    # Place images in grid
    for idx in range(n_images):
        row = idx // n_cols
        col = idx % n_cols
        
        y_start = row * (img_h + padding)
        x_start = col * (img_w + padding)
        
        img = images[idx]
        # Denormalize if needed
        if img.min() < 0:
            img = denormalize_image(img)
        
        grid[y_start:y_start+img_h, x_start:x_start+img_w] = img
    
    return grid.astype(np.uint8)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
    
    Returns:
        Resized image
    """
    if len(image.shape) == 2:
        # Grayscale
        resized = cv2.resize(image, (target_size[1], target_size[0]))
    else:
        # Color
        resized = cv2.resize(image, (target_size[1], target_size[0]))
    
    return resized


def get_image_statistics(images: np.ndarray) -> dict:
    """
    Calculate statistics for a batch of images.
    
    Args:
        images: Array of images
    
    Returns:
        Dictionary of statistics
    """
    return {
        'count': len(images),
        'shape': images[0].shape,
        'mean': np.mean(images),
        'std': np.std(images),
        'min': np.min(images),
        'max': np.max(images)
    }
