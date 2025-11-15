"""
Visualization utilities for GAN training and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Union


def plot_training_history(
    generator_losses: List[float],
    discriminator_losses: List[float],
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot training loss curves for generator and discriminator.
    
    Args:
        generator_losses: List of generator losses per epoch
        discriminator_losses: List of discriminator losses per epoch
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Combined losses
    axes[0].plot(generator_losses, label='Generator', color='blue', linewidth=2)
    axes[0].plot(discriminator_losses, label='Discriminator', color='red', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Losses', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Generator loss only
    axes[1].plot(generator_losses, color='blue', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Generator Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_image_grid(
    images: np.ndarray,
    n_rows: int = 2,
    n_cols: int = 5,
    title: str = "Generated Images",
    save_path: Optional[Union[str, Path]] = None,
    denormalize: bool = True
) -> None:
    """
    Plot a grid of images.
    
    Args:
        images: Array of images to plot
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        title: Title for the plot
        save_path: Optional path to save the plot
        denormalize: Whether to denormalize images from [-1, 1] to [0, 1]
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i]
            
            # Denormalize if needed
            if denormalize:
                img = (img + 1) / 2.0
                img = np.clip(img, 0, 1)
            
            ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_image_comparison(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    n_samples: int = 5,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot side-by-side comparison of real and generated images.
    
    Args:
        real_images: Array of real images
        generated_images: Array of generated images
        n_samples: Number of image pairs to display
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    
    for i in range(n_samples):
        # Real images
        real_img = (real_images[i] + 1) / 2.0
        real_img = np.clip(real_img, 0, 1)
        axes[0, i].imshow(real_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Real Images', fontweight='bold', fontsize=12)
        
        # Generated images
        gen_img = (generated_images[i] + 1) / 2.0
        gen_img = np.clip(gen_img, 0, 1)
        axes[1, i].imshow(gen_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Generated Images', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_latent_space_interpolation(
    generator,
    latent_dim: int,
    n_steps: int = 10,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Visualize interpolation in latent space.
    
    Args:
        generator: Trained generator model
        latent_dim: Dimension of latent space
        n_steps: Number of interpolation steps
        save_path: Optional path to save the plot
    """
    # Generate two random points in latent space
    z1 = np.random.normal(0, 1, (1, latent_dim))
    z2 = np.random.normal(0, 1, (1, latent_dim))
    
    # Linear interpolation between z1 and z2
    interpolated_z = []
    for alpha in np.linspace(0, 1, n_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        interpolated_z.append(z)
    
    interpolated_z = np.concatenate(interpolated_z, axis=0)
    
    # Generate images
    generated_images = generator.predict(interpolated_z)
    
    # Plot
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2))
    fig.suptitle('Latent Space Interpolation', fontsize=14, fontweight='bold')
    
    for i in range(n_steps):
        img = (generated_images[i] + 1) / 2.0
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_loss_distribution(
    losses: List[float],
    title: str = "Loss Distribution",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot histogram of loss values.
    
    Args:
        losses: List of loss values
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Loss Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    plt.axvline(mean_loss, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_loss:.4f}')
    plt.axvline(mean_loss - std_loss, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(mean_loss + std_loss, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'Std: {std_loss:.4f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_training_animation(
    image_files: List[Path],
    output_path: Union[str, Path],
    duration: int = 200
) -> None:
    """
    Create an animated GIF from training progress images.
    
    Args:
        image_files: List of image file paths
        output_path: Path to save the animation
        duration: Duration of each frame in milliseconds
    """
    from PIL import Image
    
    images = [Image.open(img_file) for img_file in image_files]
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    print(f"Animation saved to: {output_path}")


def plot_model_architecture(model, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Visualize model architecture.
    
    Args:
        model: Keras model
        save_path: Optional path to save the plot
    """
    from tensorflow.keras.utils import plot_model
    
    if save_path is None:
        save_path = f"{model.name}_architecture.png"
    
    plot_model(
        model,
        to_file=save_path,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=150
    )
    
    print(f"Model architecture saved to: {save_path}")
