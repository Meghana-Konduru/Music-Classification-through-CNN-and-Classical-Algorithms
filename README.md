# Music Classification through CNN and Classical Algorithms

## Overview
This project investigates whether deep features learned by a **Dilated CNN** can improve traditional machine-learning
algorithms for music genre classification. It compares, on one identical train/test split:

1. Classical models (Logistic Regression, SVM, Random Forest, Gaussian discriminant) on handcrafted audio features.
2. A standalone 2-layer Dilated CNN on MFCC sequences.
3. **Hybrid** models: the same classical models trained on features extracted from the CNN's first conv block,
   second conv block, and final embedding layer.

The research question: *do intermediate representations learned by a Dilated CNN provide better features for classical
algorithms than traditional handcrafted audio features?*

## Reference
Inspired by "Combining CNN and Classical Algorithms for Music Genre Classification", Stanford CS229 Project, 2018
(<https://cs229.stanford.edu/proj2018/report/19.pdf>). **This code is an independent implementation and was not
written by the Stanford authors.**

## Architecture
```
Audio (.wav)
      |
      v
    MFCC
      |
      +--------------------------+
      |                          |
      v                          v
Classical Features          Dilated CNN
      |                          |
      v                  Layer 1 Features
LR/SVM/RF/GDA                    |
                                 v
                         Layer 2 Features
                                 |
                                 v
                           CNN Embedding
                                 |
               +-----------------+------------------+
               |                 |                  |
               v                 v                  v
              LR                SVM            Random Forest (+ Gaussian)
               |
               v
          Genre Prediction
```
CNN: `Input(1292,20) -> Conv1D(32,k=8,d=2) -> BN -> ReLU -> AvgPool(4) -> Dropout -> Conv1D(64,k=8,d=4) -> BN -> ReLU
-> AvgPool(4) -> Dropout -> GlobalAvgPool -> Dense(64) -> Dropout(0.5) -> Softmax`.
Layer-1 / Layer-2 features are the post-ReLU conv outputs with global average pooling (32-d / 64-d); the embedding is the
64-d dense layer.

## Dataset
[GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html): 1000 audio tracks, 30 s each, 10 genres.
The default experiment uses 5 genres (blues, classical, hiphop, metal, pop); set `genres:` in `config.yaml` to all 10 for
the full dataset. The dataset is **not** downloaded automatically.

## Models
- **Logistic Regression** - multinomial (softmax) regression, `max_iter >= 2000`.
- **SVM** - RBF kernel, `probability=True`.
- **Random Forest** - `random_state=42`, configurable trees, `max_depth` grid includes 7 (reference paper controls depth).
- **Gaussian discriminant** - `LinearDiscriminantAnalysis` (default) or `QuadraticDiscriminantAnalysis`
  (`gaussian.kind` in config): the practical scikit-learn equivalent of the Gaussian Discriminant Analysis in the paper
  (class-conditional Gaussians + Bayes rule).
- **Dilated CNN** - 2 dilated conv blocks (dilation 2 then 4); the paper reports deeper nets overfit.
- **Hybrid** - the 4 classical models above on CNN Layer 1 / Layer 2 / Embedding features (12 experiments).

All classical models use `Pipeline(StandardScaler -> optional PCA -> classifier)` tuned by stratified CV on the
**training set only**.

## Setup
```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```
Python 3.10-3.12 (TensorFlow >= 2.16). Native Windows TensorFlow runs on CPU, which is sufficient here.

## Dataset setup
Place the GTZAN folder so it looks like:
```
data/genres_original/{blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock}/*.wav
```
(Known issue: `jazz.00054.wav` is corrupted in some GTZAN copies; unreadable files are skipped and logged to
`artifacts/features/dropped_files.csv`.)

## Feature extraction
```
python -m src.feature_extraction          # add --force to recompute
python -m src.eda                         # exploratory plots -> results/plots/
```
Creates `X.npy`, `y.npy`, `metadata.csv`, `X_mfcc_seq.npy`, `split.json`, `mfcc_norm_stats.npz` in `artifacts/features/`.

## Training (run in this order)
```
python train_baselines.py     # classical models on handcrafted + MFCC-summary features
python train_cnn.py           # dilated CNN -> artifacts/cnn/best_cnn.keras
python train_hybrid.py        # CNN features + classical models; writes hybrid_results.csv
```

## Evaluation
```
python evaluate_all.py
```
Produces accuracy/F1 bar charts, PCA plots (handcrafted, Layer 1, Layer 2, Embedding), silhouette scores, McNemar
tests, `results/metrics/final_comparison.csv`. Per-model confusion matrices are in `results/confusion_matrices/`.
`hybrid_results.csv` is the full table: `Model | Feature Source | Accuracy | Precision | Recall | F1 (+ CV F1, params)`.

## Prediction
```
python predict_song.py --file sample.wav
```
Prints the CNN prediction and the prediction of the best hybrid model (selected by **training-CV** score, not test score).

## Methodology notes (leakage control)
- One stratified 80/20 **song-level** split (`random_state=42`) saved in `split.json` and reused by every script.
- Scalers / PCA live inside sklearn Pipelines (fit on train only); CNN input normalisation uses train-only statistics.
- CNN validation data is carved out of the training set; test songs are used only for final scoring.
- No audio segmentation is used. If you add it, split by `song_id` first and aggregate predictions per song.
- Hyper-parameters and the "best hybrid" are chosen by CV on the training set.
- Caveat: the CNN has seen the training songs, so its features on training data are cleaner than on test data. Hybrid
  CV scores are therefore optimistic (consistently across all hybrid candidates). Out-of-fold CNN features would be
  the rigorous fix (a possible extension).
- Small test sets (e.g. 100 songs for 5 genres) mean 1 song = 1%. Use `mcnemar_vs_handcrafted.csv` before claiming
  differences. Results are close to, but not bit-for-bit, reproducible across hardware (TensorFlow).

## Project structure
See the repository tree: `src/` (modules), root scripts (`train_*.py`, `evaluate_all.py`, `predict_song.py`),
`config.yaml`, `artifacts/` (models, features), `results/` (plots, metrics), `notebooks/experiment_analysis.ipynb`.
