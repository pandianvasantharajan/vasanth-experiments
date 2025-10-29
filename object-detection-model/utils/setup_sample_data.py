#!/usr/bin/env python3
"""
Sample Test Images Setup Script
Downloads sample images for object detection testing
"""

import os
import requests
from pathlib import Path
import sys

def download_sample_images():
    """Download sample images for testing"""
    
    # Create sample images directory
    sample_dir = Path("../data/sample_images")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample image URLs (public domain/free to use)
    sample_images = [
        {
            "url": "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=800",
            "filename": "dog.jpg",
            "description": "Dog on couch"
        },
        {
            "url": "https://images.unsplash.com/photo-1549280328-6c04698b8653?w=800",
            "filename": "street_cars.jpg", 
            "description": "Street scene with cars"
        },
        {
            "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",
            "filename": "cat.jpg",
            "description": "Cat sitting"
        },
        {
            "url": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=800",
            "filename": "person_kitchen.jpg",
            "description": "Person in kitchen"
        },
        {
            "url": "https://images.unsplash.com/photo-1558618666-fccd25c85cd3?w=800",
            "filename": "busy_street.jpg",
            "description": "Busy street scene"
        }
    ]
    
    print("📥 Downloading sample test images...")
    
    successful_downloads = 0
    for img_info in sample_images:
        file_path = sample_dir / img_info["filename"]
        
        if file_path.exists():
            print(f"  ✓ {img_info['filename']} already exists")
            successful_downloads += 1
            continue
            
        try:
            print(f"  📦 Downloading {img_info['filename']}...")
            response = requests.get(img_info["url"], timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  ✓ {img_info['filename']} downloaded ({img_info['description']})")
            successful_downloads += 1
            
        except Exception as e:
            print(f"  ✗ Failed to download {img_info['filename']}: {e}")
    
    print(f"\n✅ Downloaded {successful_downloads}/{len(sample_images)} images")
    print(f"📁 Images saved to: {sample_dir.absolute()}")
    
    return successful_downloads > 0

def create_dataset_info():
    """Create dataset information file"""
    
    dataset_info = {
        "name": "Sample Object Detection Test Images",
        "description": "Collection of test images for object detection validation",
        "source": "Unsplash (free to use)",
        "total_images": 5,
        "expected_objects": [
            "person", "dog", "cat", "car", "bicycle", "chair", "couch",
            "bottle", "cup", "laptop", "cell phone", "book"
        ],
        "image_details": [
            {
                "filename": "dog.jpg",
                "expected_classes": ["dog", "couch", "person"],
                "description": "Indoor scene with dog on furniture"
            },
            {
                "filename": "street_cars.jpg", 
                "expected_classes": ["car", "person", "traffic light"],
                "description": "Urban street with vehicles"
            },
            {
                "filename": "cat.jpg",
                "expected_classes": ["cat"],
                "description": "Portrait of a cat"
            },
            {
                "filename": "person_kitchen.jpg",
                "expected_classes": ["person", "bottle", "cup", "bowl"],
                "description": "Person in kitchen environment"
            },
            {
                "filename": "busy_street.jpg",
                "expected_classes": ["person", "car", "bicycle", "backpack"],
                "description": "Busy urban street scene"
            }
        ]
    }
    
    info_file = Path("../data/sample_images_info.json")
    with open(info_file, 'w') as f:
        import json
        json.dump(dataset_info, f, indent=2)
    
    print(f"📋 Dataset info saved to: {info_file.absolute()}")

def main():
    """Main function"""
    print("🎯 Object Detection Sample Images Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("../data").exists():
        print("❌ Error: Run this script from the utils/ directory")
        print("   Expected project structure: object-detection-model/utils/")
        sys.exit(1)
    
    # Download images
    success = download_sample_images()
    
    if success:
        # Create dataset info
        create_dataset_info()
        
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the main notebook: notebooks/object_detection_analysis.ipynb")
        print("2. Or use make demo for a quick test")
        
    else:
        print("\n❌ Setup failed. Please check your internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()