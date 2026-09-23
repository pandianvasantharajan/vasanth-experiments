# AMIS Experiments - Image Segmentation

This project contains experiments for image segmentation using various techniques, starting with K-Means clustering.

## Overview

Image segmentation is the process of partitioning an image into multiple segments or regions, making it easier to analyze and understand the image. This project explores:

- **K-Means Clustering**: Unsupervised learning algorithm for color-based segmentation
- Color space analysis (RGB, HSV, LAB)
- Image preprocessing and enhancement
- Segmentation evaluation metrics

## Project Structure

```
amis-experiments/
├── notebooks/           # Jupyter notebooks with experiments
│   └── kmeans_image_segmentation.ipynb
├── data/               # Input images (add your images here)
├── results/            # Segmented images and analysis results
├── utils/              # Helper functions and utilities
└── README.md           # This file
```

## K-Means Image Segmentation

K-Means clustering groups pixels with similar colors together, effectively segmenting the image into K distinct regions.

### Algorithm Steps:
1. **Image Preprocessing**: Load and optionally preprocess the image
2. **Feature Extraction**: Convert pixels to feature vectors (RGB, HSV, or LAB)
3. **K-Means Clustering**: Group pixels into K clusters
4. **Reconstruction**: Create segmented image using cluster centers
5. **Visualization**: Display original and segmented images

### Key Parameters:
- `n_clusters`: Number of segments/regions (K)
- `color_space`: RGB, HSV, or LAB color space
- `random_state`: For reproducible results

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. Add your images to the `data/` directory
2. Open the notebook: `jupyter notebook notebooks/kmeans_image_segmentation.ipynb`
3. Run the cells to perform segmentation
4. Results will be saved to `results/` directory

### Example

```python
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Load image
image = Image.open('data/sample.jpg')
pixels = np.array(image).reshape(-1, 3)

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels)

# Create segmented image
segmented = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented.reshape(image.size[1], image.size[0], 3)
```

## Features

- Multiple color space support (RGB, HSV, LAB)
- Adjustable number of clusters
- Elbow method for optimal K selection
- Comparison visualizations
- Performance metrics
- Batch processing support

## Requirements

- Python 3.8+
- NumPy
- scikit-learn
- Pillow (PIL)
- Matplotlib
- OpenCV (cv2)
- SciPy

## Applications

- **Object Detection**: Separate objects from background
- **Medical Imaging**: Segment anatomical structures
- **Satellite Imagery**: Land use classification
- **Computer Vision**: Preprocessing for object recognition
- **Image Compression**: Reduce color palette

## References

- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"

## License

MIT License
