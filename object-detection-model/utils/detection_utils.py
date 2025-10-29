# Object Detection Utilities

import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# Import cv2 with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available. Some functionality may be limited.")

class DetectionUtils:
    """Utility functions for object detection tasks"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image from file path"""
        return cv2.imread(image_path)
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, target_size)
    
    @staticmethod
    def draw_bounding_box(image: np.ndarray, bbox: List[int], 
                         label: str, confidence: float, 
                         color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw bounding box with label on image"""
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label_text = f"{label}: {confidence:.2f}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        return image
    
    @staticmethod
    def calculate_iou(box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

class COCODatasetHandler:
    """Handler for COCO dataset operations"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.classes = self._load_coco_classes()
    
    def _load_coco_classes(self) -> List[str]:
        """Load COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def get_image_paths(self, split: str = 'val2017') -> List[Path]:
        """Get list of image paths for a given split"""
        split_dir = self.dataset_path / split
        if split_dir.exists():
            return list(split_dir.glob('*.jpg'))
        return []
    
    def load_annotations(self, split: str = 'val2017') -> Optional[Dict]:
        """Load COCO annotations for a given split"""
        annotation_file = self.dataset_path / 'annotations' / f'instances_{split}.json'
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                return json.load(f)
        return None

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        self.metrics = {}
    
    def calculate_precision_recall(self, predictions: List[Dict], 
                                 ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate precision and recall metrics"""
        # Implementation for precision/recall calculation
        # This is a simplified version - full implementation would require
        # proper matching of predictions to ground truth based on IoU
        
        total_predictions = len(predictions)
        total_ground_truth = len(ground_truth)
        
        # Simplified calculation
        true_positives = min(total_predictions, total_ground_truth)
        false_positives = max(0, total_predictions - total_ground_truth)
        false_negatives = max(0, total_ground_truth - total_predictions)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate_model_performance(self, results: List[Dict]) -> Dict[str, float]:
        """Evaluate overall model performance"""
        if not results:
            return {}
        
        total_inference_time = sum(r.get('inference_time', 0) for r in results)
        total_detections = sum(r.get('analysis', {}).get('total_detections', 0) for r in results)
        
        return {
            'avg_inference_time': total_inference_time / len(results),
            'avg_detections_per_image': total_detections / len(results),
            'throughput_fps': len(results) / total_inference_time if total_inference_time > 0 else 0,
            'total_images': len(results),
            'total_detections': total_detections
        }

class ReportGenerator:
    """Generate comprehensive reports"""
    
    def __init__(self, output_dir: str = "../results/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(self, results: List[Dict], metrics: Dict) -> str:
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .section {{ margin: 20px 0; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .image-card {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Object Detection Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metric">Total Images: {metrics.get('total_images', 'N/A')}</div>
                <div class="metric">Total Detections: {metrics.get('total_detections', 'N/A')}</div>
                <div class="metric">Average Inference Time: {metrics.get('avg_inference_time', 0):.3f}s</div>
                <div class="metric">Throughput: {metrics.get('throughput_fps', 0):.2f} FPS</div>
            </div>
            
            <div class="section">
                <h2>Detection Results</h2>
                <div class="image-grid">
        """
        
        for i, result in enumerate(results[:6]):  # Show first 6 results
            image_name = Path(result['image_path']).name
            detections = result['analysis']['total_detections']
            classes = ', '.join(result['analysis']['class_counts'].keys())
            
            html_content += f"""
                <div class="image-card">
                    <h3>Image {i+1}: {image_name}</h3>
                    <p>Detections: {detections}</p>
                    <p>Classes: {classes}</p>
                </div>
            """
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        html_path = self.output_dir / "detection_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def save_metrics_csv(self, results: List[Dict]) -> str:
        """Save metrics to CSV file"""
        import pandas as pd
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            analysis = result['analysis']
            csv_data.append({
                'image_name': Path(result['image_path']).name,
                'model_name': result['model_name'],
                'total_detections': analysis['total_detections'],
                'inference_time': result['inference_time'],
                'unique_classes': len(analysis['class_counts']),
                'mean_confidence': analysis['confidence_stats'].get('mean', 0),
                'detected_classes': ', '.join(analysis['class_counts'].keys())
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "detection_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)