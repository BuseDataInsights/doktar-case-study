# Doktar Case Study – VMC Calibration & Black‑Fly Detection

This repository contains my solutions for the Doktar Machine Learning Engineer case study.  
Each question lives in its own folder so reviewers can run or inspect the parts independently.

---

## Directory Overview

```
q1/             VMC calibration task 
q2/             Black‑fly detection task 
q3/             System‑architecture answer (PDF)
README.md       This file
```

---

## Question 1 – VMC Calibration

* **Where to look**  
  * `q1/NOTEBOOK.ipynb` – full exploratory analysis, model comparison, validation and diagnostics with all visuals and code cells.  
  * `q1/Dockerfile` and Python modules reproduce the training and inference pipeline in a container.

* **Content highlights**  
  * **a. EDA & transformations** – shows the monotonic but slightly non‑linear relationship between *Measured VMC* and *Normalized_Values*.  
  * **b. Model comparison** – selects isotonic for lower RMSE while keeping interpretability (metric = RMSE). 
  * **c–d. Robustness & diagnostics** – out‑of‑range guard, drift test proposal and residual checks are demonstrated directly in the notebook.

---

## Question 2 – Black‑Fly Detection

All code and artefacts live in `Q2/`.

```
q2/
├─ blackfly_pipeline.py      detection + inference script
├─ blackfly.yaml             YOLO data config
├─ yolov8n.pt                base weights (YOLO‑v8 nano)
└─ runs/detect               auto‑generated experiment outputs

```

> **Note**  
> The raw `images/` and `labels/` folders (train, val, test1, test2) are **not** committed because they are several GB and were annotated in CVAT. I only included one of the images with its annotation (label) for the training set as an example, in the image-label example folder.
> The model is trained by using **all** the provided train set for blackfly. All images are utilized for model training. 
> The trained model's **annotated prediction images** can be inspected in:
>
> * `q2/runs/detect/blackfly_test1_zoomed`  
> * `q2/runs/detect/blackfly_test2_wide_multiscale`

### How the model was built

* **Training data** – only class 0 `"black‑fly"` annotated so the task focuses on one species.  
* **Hyper‑parameters** – YOLO‑v8 *nano*, 10 epochs, batch 16, patience 3.  
* **Inference flow** – single‑scale for `test1`, multi‑scale tiled for `test2`.

---

## Question 3 – Production Architecture

`q3/Q3 answer.pdf` contains the detailed system design that explains:

* Component diagram covering Docker images, FastAPI microservices, NGINX gateway and ECS blue/green deployment.  
* Observability, authentication, drift monitoring and CI‑CD flow.

---

Feel free to open an issue if anything is unclear.
