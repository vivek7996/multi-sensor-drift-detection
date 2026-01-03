# multi-sensor-drift-detection
A project based on finding precursor by analyzing multi-sensor's reading drift and predicting future impacts.


--Description

This project proposes an unsupervised machine learning framework for detecting early drift and degradation patterns in multi-sensor time-series data.
Latent embeddings are learned using autoencoders and analyzed through manifold alignment and distance-based drift metrics.

--Dataset

NASA CMAPSS Turbofan Engine Dataset
Multiple engine sensors (21 channels)
Run-to-failure time-series data

--Methodology

Sliding window generation
Autoencoder-based latent embedding
Manifold alignment across time windows
Drift detection using Euclidean, Cosine & Mahalanobis distances
Stability, sensitivity & lead-time analysis

--Technologies Used

Python
NumPy, Pandas
PyTorch / TensorFlow
Scikit-learn
Matplotlib / Seaborn

--Results

Early drift detection capability
Stable manifold representations
Reduced false alarm rate



--workflow
multi-sensor-drift-detection/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── scalers/
│   ├── ae_scalers.pkl
│   └── scaler_ssensors.pkl
│
├── models/
│   ├── Z_concat.npy
    |── config.json
    |── trans_model.pth
│   └── ae_sensors.pth
    
│
├── src/
│   ├── untitled24(1).py
│   ├── data_load(1).py
│   ├── inference(1).py
    ├── utils(1).py
│   └── runner(1).py
│
├── README.md
├── requirements.txt

