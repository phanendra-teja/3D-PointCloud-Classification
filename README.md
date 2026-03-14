# 🧊 3D Point Cloud Classification — PointNet++ (TensorFlow & PyTorch)

A custom implementation of the **PointNet++** architecture for classifying 3D point cloud objects across **16 shape categories** from the ShapeNetPart dataset — built independently in both **TensorFlow/Keras** and **PyTorch** for cross-framework comparison.

This project was developed as part of research/internship work at NIT Puducherry (NITPY).

---

## 📌 Project Overview

| Detail | Value |
|---|---|
| Task | Multi-class 3D point cloud classification |
| Dataset | ShapeNetPart (HDF5 format) |
| Input | 1024 points × 3 coordinates (XYZ) |
| Output | Shape category (0–15), 16 classes |
| Architecture | PointNet++ (FPS + Set Abstraction + Global Pooling) |

### Notebooks

| Notebook | Framework | Save Format |
|---|---|---|
| `3d.ipynb` | TensorFlow / Keras | `.keras` |
| `3d_pytorch.ipynb` | PyTorch | `.pth` (state dict) |

> Both notebooks implement the **same architecture and task** — the project was built twice to compare framework behaviour, pipeline design, and training ergonomics.

---

## 📂 Dataset — ShapeNetPart

- **Source**: ShapeNetPart dataset (HDF5 format, pre-split into `train*.h5` / `test*.h5`)
- **Points per object**: Up to 2048 (sampled down to 1024 during preprocessing)
- **Classes**: 16 object categories

| Label | Category | Label | Category |
|---|---|---|---|
| 0 | Airplane ✈️ | 8 | Lamp 💡 |
| 1 | Bag 👜 | 9 | Laptop 💻 |
| 2 | Cap 🧢 | 10 | Motorbike 🏍️ |
| 3 | Car 🚗 | 11 | Mug ☕ |
| 4 | Chair 🪑 | 12 | Pistol 🔫 |
| 5 | Earphone 🎧 | 13 | Rocket 🚀 |
| 6 | Guitar 🎸 | 14 | Skateboard 🛹 |
| 7 | Knife 🔪 | 15 | Table 🪵 |

### Dataset Split

| Split | Shape | Description |
|---|---|---|
| train_points | (N_train, 2048, 3) | Raw training point clouds |
| train_labels | (N_train,) | Integer category labels |
| test_points | (N_test, 2048, 3) | Raw test point clouds |
| test_labels | (N_test,) | Integer category labels |

> ⚠️ Dataset files are **NOT included** in this repo. Download ShapeNetPart HDF5 separately and set `DATA_DIR` in each notebook.

---

## 🧠 Model Architecture — PointNet++

Both notebooks implement the same PointNet++ architecture. It hierarchically extracts local and global 3D geometric features using **Farthest Point Sampling (FPS)** and **Set Abstraction (SA) layers**.

```
Input: (batch, N, 3) — N = 1024 XYZ point cloud

── Set Abstraction 1 (SA1) ─────────────────
  FPS → 512 centroids
  MLP: [64, 64, 128] + BatchNorm + ReLU
  Output: (batch, 512, 128)

── Set Abstraction 2 (SA2) ─────────────────
  FPS → 128 centroids
  MLP: [128, 128, 256] + BatchNorm + ReLU
  Output: (batch, 128, 256)

── Set Abstraction Global (SA3) ────────────
  No sampling — global max pooling
  MLP: [256, 512, 1024] + BatchNorm + ReLU
  Output: (batch, 1024)

── Classification Head ─────────────────────
  Dense/Linear(512) + BatchNorm + ReLU + Dropout(0.5)
  Dense/Linear(256) + BatchNorm + ReLU + Dropout(0.5)
  Dense/Linear(16)  + Softmax/Logits      ← 16-class output
```

---

## ⚙️ Framework Comparison

| Aspect | TensorFlow (`3d.ipynb`) | PyTorch (`3d_pytorch.ipynb`) |
|---|---|---|
| Input shape | `(batch, 1024, 3)` | `(batch, 3, 1024)` — transposed |
| Data pipeline | `tf.data` with `AUTOTUNE` prefetch | Custom `Dataset` + `DataLoader` |
| Preprocessing | `@tf.function` GPU-accelerated ops | NumPy inside `__getitem__` |
| Conv layers | `Conv1D` (Keras) | `Conv1d` (PyTorch) |
| Loss function | `SparseCategoricalCrossentropy` | `CrossEntropyLoss` (raw logits) |
| Model save | `model.save()` → `.keras` | `torch.save(state_dict())` → `.pth` |
| Model load | `keras.models.load_model()` | `model.load_state_dict()` |
| GPU memory logging | ✅ Before + after training | ❌ Not implemented |
| Windows fix | `KMP_DUPLICATE_LIB_OK=TRUE` | `KMP_DUPLICATE_LIB_OK=TRUE` + `num_workers=0` |
| Batch size guard | ❌ Not needed | ✅ Skips batches with size < 2 (BatchNorm1d requirement) |

---

## ⚙️ Preprocessing & Augmentation

Both notebooks apply identical preprocessing logic, just in different frameworks.

### Training
1. **Random Sampling** — select 1024 points from 2048
2. **Centroid Normalization** — subtract mean XYZ
3. **Unit Sphere Scaling** — divide by max point distance
4. **Jitter Augmentation** — Gaussian noise clipped to ±0.05
5. **Random Z-Rotation** — random angle rotation matrix

### Test
Steps 1–3 only (no augmentation).

---

## 📈 Training Configuration

| Parameter | Value |
|---|---|
| NUM_POINTS | 1024 |
| NUM_CLASSES | 16 |
| BATCH_SIZE | 8 |
| EPOCHS | 1 *(safety default — increase for real training)* |
| Optimizer | Adam (lr=0.001) |

---

## 🛠️ Tech Stack

| Library | TensorFlow Notebook | PyTorch Notebook |
|---|---|---|
| Python 3.8+ | ✅ | ✅ |
| TensorFlow 2.x | ✅ | ❌ |
| PyTorch 2.x | ❌ | ✅ |
| NumPy | ✅ | ✅ |
| h5py | ✅ | ✅ |
| Matplotlib | ✅ | ✅ |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/phanendra-teja/3D-PointCloud-Classification.git
cd 3D-PointCloud-Classification
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

**For TensorFlow notebook:**
```bash
pip install tensorflow h5py numpy matplotlib
```

**For PyTorch notebook:**
```bash
pip install torch torchvision h5py numpy matplotlib
```

> For CUDA GPU support with PyTorch, install the correct build from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Download Dataset

Download the ShapeNetPart HDF5 dataset. Place the folder containing `train*.h5` and `test*.h5` files locally, then update `DATA_DIR` in whichever notebook you're running:

```python
DATA_DIR = r"path\to\shapenetpart_hdf5_2048"
```

### 5. Run

```bash
# TensorFlow
jupyter notebook 3d.ipynb

# PyTorch
jupyter notebook 3d_pytorch.ipynb
```

---

## 📁 Project Structure

```
3D-PointCloud-Classification/
│
├── 3d.ipynb                              # TensorFlow/Keras implementation
├── 3d_pytorch.ipynb                      # PyTorch implementation
├── shapenet_classifier_model.keras       # TF saved model (after training)
├── shapenet_classifier_pytorch.pth       # PyTorch state dict (after training)
├── .gitignore
└── README.md                             # This file

# (Not included — download separately)
# shapenetpart_hdf5_2048/
#   ├── train0.h5 ... trainN.h5
#   └── test0.h5  ... testN.h5
```

---

## 🔮 Sample Single-Object Prediction

**TensorFlow:**
```python
single_object_processed = preprocess_test_tf(tf.constant(test_points[5]))
single_object_batch = tf.expand_dims(single_object_processed, axis=0)

predictions = model.predict(single_object_batch)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"Predicted: {category_names[predicted_class]} ({confidence*100:.2f}%)")
print(f"True:      {category_names[test_labels[5]]}")
```

**PyTorch:**
```python
single_object_tensor = torch.from_numpy(normalized_points).float().transpose(0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    predictions = model(single_object_tensor)

probabilities = torch.softmax(predictions, dim=1)[0]
confidence, pred_idx = torch.max(probabilities, 0)

print(f"Predicted: {category_names[pred_idx.item()]} ({confidence.item()*100:.2f}%)")
print(f"True:      {category_names[true_label]}")
```

---

## 📌 .gitignore (Recommended)

```
__pycache__/
*.pyc
.venv/
*.keras
*.pth
shapenetpart_hdf5_2048/
```

---

## 🎯 Future Improvements

- Train for **more epochs (50–200)** with a learning rate scheduler — 1 epoch is a safety default only
- Implement **ball query / radius search** for true PointNet++ local neighborhood grouping
- Add **part segmentation head** to extend from classification to per-point label prediction (ShapeNetPart fully supports this)
- Use **CUDA-accelerated FPS** (e.g., `torch-points-kernels`) for faster sampling
- Add **confusion matrix and per-class accuracy** after evaluation
- Experiment with **PointNet++ MSG (Multi-Scale Grouping)** for improved accuracy
- Save the **best model checkpoint** based on validation accuracy during training
- Build a **3D viewer demo** using Open3D for interactive visualization of predictions

---

## 👤 Author

**Phanendra Teja V**  
B.Tech CSE — NIT Puducherry (NITPY), Batch 2024–2028

---

## 📄 License

This project is for educational and research purposes.  
Dataset: ShapeNetPart — originally from the ShapeNet project.
