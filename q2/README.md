# BlackFly Detection with YOLOv8

This project is a YOLOv8-based application for blackfly detection in pest trap images.

## Features

- YOLOv8-based 2-class model training (black-fly, insect)
- Multi-scale tiled inference for large images
- Dockerized deployment
- Production-ready code structure

## Installation and Usage

### Docker (Recommended)

```bash
# Build Docker image
docker build -t blackfly-detector .

# Run container
docker run -v $(pwd)/images:/app/images:ro \
           -v $(pwd)/labels:/app/labels:ro \
           -v $(pwd)/runs:/app/runs \
           blackfly-detector
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python blackfly_pipeline.py
```

## Project Structure

```
yolov8/
├── blackfly_pipeline.py    # Main application
├── blackfly.yaml          # YOLOv8 data configuration
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── runs/detect            # Model outputs and results
```

## Usage

The application works in two stages:

1. **Model Training**: Fine-tuning using YOLOv8n base model on custom dataset
2. **Prediction**: Inference on test images using trained model

### Test Sets

- **test1**: Direct inference for small/cropped images (416px)
- **test2**: Multi-scale tiled inference for large images (4096, 10016, 12032px)

## Configuration

Main parameters are defined at the top of `blackfly_pipeline.py`:

- `IMG_SIZE_ZOOM`: Image size for test1
- `TILE_SIZES`: Multi-scale tile sizes for test2
- `OVERLAP`: Overlap ratio between tiles
- `CONF_THRES`: Confidence threshold

## Outputs

Results are saved in `runs/detect/` directory:

- Predicted images with bounding boxes
- Bounding box coordinates (txt format)
- Confidence scores
