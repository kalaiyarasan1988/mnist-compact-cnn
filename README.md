# MNIST - CNN under 20K Parameters with max validation accuracy


**Goal:** Build a compact CNN for MNIST that satisfies the following constraints and achieves high accuracy:
- **Validation/Test accuracy target:** ≥ 99.4% (measured on the 10k split taken from the 60k train set)
- **Parameter budget:** < 20,000 parameters
- **Training budget:** < 20 epochs
- **Required techniques:** Batch Normalization, Dropout, Global Average Pooling (or small FC)


---


## Repo contents
- `mnist_train.py` — training script (CompactMNISTNet, tuned hyperparams)
- `mnist_notebook.ipynb` — notebook (same model + training cells)
- `README.md` — this file (you are reading it)
- `best_mnist.pth` — (saved model after training, generated during your run)


---


## Quick instructions
```bash
# create virtualenv, install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt # or: pip install torch torchvision tqdm


# run training (50k train / 10k val split)
python3 mnist_train.py --epochs 20 --batch-size 128 --device cuda
```


**Outputs printed to console** include: parameter count at start, per-epoch train/val accuracy, and the saved `best_mnist.pth` when the best val acc improves.


---


## Final Model Summary
- **Model name:** `CompactMNISTNet` (final tuned)
- **Total trainable parameters:** **18,472**
- **Techniques used:** Batch Normalization (after every conv), Dropout (Dropout2d with p=0.05), Global Average Pooling (GAP) + small FC head
- **Optimizer & LR:** Adam, lr=0.001 (weight_decay=1e-4)
- **Augmentation:** RandomRotation(±5°), small translate (±4%)
- **Best validation accuracy (50k/10k split):** **99.25%**
- **Epochs used:** 20 (stopped after best observed earlier)


---


## Detailed change-log & step-by-step improvements 
This section documents each architectural/training decision and the observed result. It demonstrates concepts considered (layers, receptive fields, BN, dropout, pooling, kernels, learning rate, batch size effects, etc.).


### Run 1 — **Very small net** (initial attempt)
- **Arch:** simple 3-block network with channels `[8,16,32]` but only _one_ conv per block (very lightweight)
- **Params:** ~**6,584**
- **Result:** saturated early; **Best val acc: 95.7%** after 15 epochs
- **Why it failed:** under-parameterized — not enough representational capacity; receptive field/layout limited.


**Lesson:** need slightly more capacity (more filters or extra convs per block) while keeping params <20k.


---


### Run 2 — **CompactMNISTNet (first pass)**
- **Arch:** 3 blocks with **two** 3×3 convs per block and channel progression: `[8 → 16 → 32]` + GAP + FC(32→10)
- **Params:** **18,472** (within limit)
- **Optimizer:** SGD (momentum=0.9) in earlier experiments
- **Result:** reached **99.00%** val acc in 15 epochs (very promising)


**Why this helped:** doubling convs per block increased depth/expressivity and receptive field while keeping channels small — good MD tradeoff for MNIST.


---


### Run 3 — **Tuned hyperparams (final)**
- **What changed vs Run C:**
- Optimizer: **Adam** (lr=0.001)
- Dropout2d: reduced to **p=0.05** (less aggressive regularization)
- Augmentation: smaller rotation (±5°) & smaller translate (±4%)
- Epochs: increased to **20** (cosine LR scheduler)
- **Params:** **18,472** (unchanged)
- **Result (official final logs):** **Best val acc: 99.25%**


**Why these tweaks helped:** Adam often gives faster/sharper convergence on MNIST; lighter augmentation & lower dropout allow the model to fully fit subtle digit variations; small extra training time with annealing LR gives final polish to accuracy.

---


## Training logs (selected runs)


### Run 1 — tiny net (sample)
```
Total parameters: 6584
Epoch 01: Train acc 41.94% | Val acc 69.95%
Epoch 02: Train acc 69.28% | Val acc 87.75%
Epoch 03: Train acc 77.89% | Val acc 92.11%
...
Best validation acc: 95.70%
```

### Run 2 — CompactMNISTNet (first pass)
```
Total parameters: 18472
Epoch 01: Train acc 69.52% | Val acc 94.31%
Epoch 02: Train acc 94.34% | Val acc 96.46%
...
Epoch 15: Train acc 98.55% | Val acc 98.95%
Best validation acc: 99.00%
```

### Run 3 — Tuned hyperparameters (final submission logs)
```
Total parameters: 18472
Epoch 01: Train acc 82.63% | Val acc 97.19%
Epoch 02: Train acc 97.06% | Val acc 97.51%
Epoch 03: Train acc 97.90% | Val acc 98.46%
Epoch 04: Train acc 98.12% | Val acc 98.61%
Epoch 05: Train acc 98.33% | Val acc 98.87%
Epoch 06: Train acc 98.55% | Val acc 98.38%
Epoch 07: Train acc 98.62% | Val acc 98.76%
Epoch 08: Train acc 98.74% | Val acc 98.85%
Epoch 09: Train acc 98.77% | Val acc 98.94%
Epoch 10: Train acc 98.84% | Val acc 98.83%
Epoch 11: Train acc 98.98% | Val acc 98.97%
Epoch 12: Train acc 98.99% | Val acc 99.12%
Epoch 13: Train acc 99.09% | Val acc 99.14%
Epoch 14: Train acc 99.11% | Val acc 99.09%
Epoch 15: Train acc 99.16% | Val acc 99.18%
Epoch 16: Train acc 99.24% | Val acc 99.22%
Epoch 17: Train acc 99.19% | Val acc 99.22%
Epoch 18: Train acc 99.21% | Val acc 99.23%
Epoch 19: Train acc 99.26% | Val acc 99.25%
Epoch 20: Train acc 99.29% | Val acc 99.23%
Best validation acc: 99.25%
```

> **Note:** Validation set in logs is the 10k split taken from the 60k training set (50k train / 10k val) as requested in the assignment.


---


## Design notes
- **How many layers / where to stop convolutions:** For MNIST, a shallow but slightly deepened design (3 conv-blocks with two convs each) provides enough hierarchy. I have stopped adding more blocks when params approached the budget and diminishing returns were observed.


- **MaxPooling & position:** I have placed MaxPool after the first two blocks (reducing 28→14→7) so the last block operates on 7×7 features — this gives sufficient receptive field while keeping spatial info.


- **1×1 convs / Kernels:** I have used only 3×3 kernels for spatial mixing. I can use 1×1 for channel reduction if needed, but keeping channels small (8→16→32) met the parameter budget.


- **Receptive field:** Two stacked 3×3 convolutions increase the effective receptive field similar to one 5×5 with fewer params and more non-linearity.


- **Softmax:** Final classification is via CrossEntropyLoss (PyTorch’s API combines LogSoftmax + NLLLoss).


- **Learning rate & optimizer:** SGD with momentum initially tried, but Adam (lr=0.001) gave slightly faster & higher peak accuracy in the compact setup.


- **BatchNorm & Image Normalization:** BatchNorm after each conv stabilizes training. I have normalized images using MNIST mean/std ((0.1307,), (0.3081,)).


- **Dropout & Overfitting:** Dropout2d p=0.05 reduced overfitting without underfitting. Monitor val-loss/val-acc to detect overfitting.


- **Distance of pooling / BN / dropout from prediction:** BN is applied immediately after convs; dropout applied after pooling layers in blocks (not right before prediction) to preserve signal to final GAP.


- **Batch size effects:** I have used batch_size=128. Larger batch sizes can let you use higher LR; smaller batches give noisier updates.


---