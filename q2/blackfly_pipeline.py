import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

# SETTINGS 

DATA_YAML    = "blackfly.yaml"
BASE_MODEL   = "yolov8n.pt"

IMG_SIZE_ZOOM = 416                        # for test1 (cropped images)
TILE_SIZES    = [4096, 10016, 12032]       # multi-scale tiles for test2 (pest trap images)
OVERLAP       = 0.2                        # 20% overlap between tiles
CONF_THRES    = 0.5

PROJECT_OUT   = "runs/detect"
WEIGHTS_PATH  = "runs/detect/train/weights/best.pt"

# Use relative paths for Docker compatibility
TEST_FOLDERS = {
    "test1_zoomed": Path("images/test1"),
    "test2_wide":   Path("images/test2"),
}


# TRAIN & VALIDATE

def train_and_validate():
    """Train a 2-class model and return the path to best.pt."""
    print("\n=== TRAINING & VALIDATION ===")
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=10,
        patience=3,
        imgsz=IMG_SIZE_ZOOM,
        batch=16,
        verbose=True
    )
    best_weight = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest model saved to: {best_weight}")
    return str(best_weight)


# AUXILIARY FUNCTIONS 

def tiled_predict_multi_scale(img: np.ndarray, model: YOLO, tile_size: int, overlap: float):
    """
    Break 'img' into overlapping tiles of size tile_size, run inference on each,
    return all raw [x1, y1, x2, y2] boxes in full-image coords.
    """
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    boxes = []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            tile = img[y0:y1, x0:x1]
            results = model(tile, imgsz=tile_size, conf=CONF_THRES, classes=[0])
            for box in results[0].boxes.xyxy.cpu().numpy():
                x_min, y_min, x_max, y_max = box
                # shift tile coords -> full-image coords
                boxes.append([x_min + x0, y_min + y0, x_max + x0, y_max + y0])

    return boxes


def nms(boxes: list, iou_threshold: float = 0.5):
    """
    Perform Non-Max Suppression on list of [x1,y1,x2,y2] boxes.
    Returns the filtered list of boxes.
    """
    if not boxes:
        return []
    boxes_arr = np.array(boxes)
    x1, y1, x2, y2 = boxes_arr[:,0], boxes_arr[:,1], boxes_arr[:,2], boxes_arr[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []

    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_int = np.maximum(0, xx2 - xx1)
        h_int = np.maximum(0, yy2 - yy1)
        inter = w_int * h_int
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes_arr[keep].tolist()


# PREDICTION ON BOTH TEST SETS

def predict_on_multiple_tests(weights_path: str):
    """
    Run predictions on both test sets using the trained model.
    """
    if not os.path.exists(weights_path):
        print(f"Warning: Model weights not found at {weights_path}")
        print("Training model first...")
        weights_path = train_and_validate()
    
    model = YOLO(weights_path)

    # 1) test1: simple direct inference at IMG_SIZE_ZOOM
    zoomed_name = "blackfly_test1_zoomed"
    if TEST_FOLDERS["test1_zoomed"].exists() and any(TEST_FOLDERS["test1_zoomed"].iterdir()):
        print(f"\n=== PREDICTING test1_zoomed @ {IMG_SIZE_ZOOM}px ===")
        model.predict(
            source=str(TEST_FOLDERS["test1_zoomed"]),
            imgsz=IMG_SIZE_ZOOM,
            conf=CONF_THRES,
            classes=[0],
            save=True,
            save_txt=True,
            save_conf=True,
            project=PROJECT_OUT,
            name=zoomed_name
        )
        print(f"Saved zoomed results to {(Path(PROJECT_OUT)/zoomed_name).resolve()}")
    else:
        print(f"No images found in {TEST_FOLDERS['test1_zoomed']}")

    # 2) test2: multi-scale tiled inference
    wide_name = "blackfly_test2_wide_multiscale"
    if TEST_FOLDERS["test2_wide"].exists() and any(TEST_FOLDERS["test2_wide"].iterdir()):
        save_dir = Path(PROJECT_OUT) / wide_name
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== MULTI-SCALE TILED PREDICTION ON test2_wide ===")

        for img_path in TEST_FOLDERS["test2_wide"].iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                all_boxes = []
                # run tiled inference at each tile size
                for ts in TILE_SIZES:
                    all_boxes.extend(tiled_predict_multi_scale(img, model, ts, OVERLAP))
                # merge detections
                final_boxes = nms(all_boxes, iou_threshold=0.5)

                # draw and save
                for (x1, y1, x2, y2) in final_boxes:
                    cv2.rectangle(img,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0),
                                  2)
                out_path = save_dir / img_path.name
                cv2.imwrite(str(out_path), img)

        print(f"Saved multi-scale wide results to {save_dir.resolve()}")
    else:
        print(f"No images found in {TEST_FOLDERS['test2_wide']}")


# MAIN ENTRYPOINT 

if __name__ == "__main__":
    # To retrain instead of predicting with existing weights, uncomment:
    # WEIGHTS_PATH = train_and_validate()
    predict_on_multiple_tests(WEIGHTS_PATH)
