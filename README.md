# 🎓 ML Projects Portfolio

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org)

> A curated collection of production-quality machine learning and AI projects spanning computer vision, NLP, generative AI, fraud detection, and MLOps.

---

## 🏆 Featured Projects

### 1. 😊 Face Expression Recognition & Generation (Jul–Aug 2024)

**Recognition (VGG Transfer Learning)** — 78% accuracy on 15,000 images
- Transfer learning with VGG16/VGG19 on FER2013 dataset
- Preprocessing: Gaussian blur, histogram equalization, augmentation
- Regularization + hyperparameter tuning to control overfitting
- Real-time Streamlit app with live webcam predictions
- Evaluated with AUC, confusion matrix, and per-class F1

**WGAN for Expression Generation** — IS/FID-evaluated realistic diversity
- Wasserstein GAN with gradient penalty for stable training
- Five distinct emotional expressions generated (happy, sad, angry, fear, surprise)
- MTCNN face detection preprocessing
- Evaluated with Inception Score (IS) and Fréchet Inception Distance (FID)

**DCGAN for Expression Generation** — 64×64 realistic faces after 100 epochs
- Generator/discriminator with transposed convolution layers
- Binary Cross-Entropy + Adam optimizer
- Training visualized with Matplotlib loss curves

**cGAN for Conditional Generation** — class-conditioned expression synthesis
- Custom generator + discriminator with label conditioning
- Batch normalization + LeakyReLU throughout
- Live training updates with generator/discriminator loss tracking

```
Skills: TensorFlow, Keras, VGG, WGAN, DCGAN, cGAN, MTCNN, OpenCV, Streamlit
Metrics: 78% accuracy (recognition), FID/IS scores (generation)
```

---

### 2. 🔍 Fraud Detection System (Jul 2024)

- **95% accuracy** in real-time transaction classification
- Feature engineering on transaction amount, time, merchant category, behavioral patterns
- Scikit-Learn ensemble: XGBoost + Random Forest + Logistic Regression stacking
- SQL-based data pipeline for high-throughput transaction processing
- Docker deployment for scalable operations
- Power BI dashboard visualizing fraud patterns and model performance

```
Skills: Python, Scikit-Learn, XGBoost, SQL, Docker, Power BI
Metrics: 95% accuracy, <50ms inference latency
```

---

### 3. 🧬 Customer Segmentation & Risk Analysis (May–Jun 2024)

- Analyzed **millions of financial transactions** with Python + SQL + PySpark
- 95% accuracy fraud detection with predictive models
- 20% improvement in customer segmentation strategies
- 15% reduction in credit risk through ML-based risk modeling
- 25% CTR improvement via SEM campaign optimization
- Tableau dashboards for real-time model performance monitoring

```
Skills: PySpark, Python, SQL, Tableau, Machine Learning, Collaborative Filtering
```

---

### 4. 🫁 COVID-19 Diagnosis from Chest X-Rays (May 2024)

- Deep learning model for automated COVID-19 detection from chest X-ray images
- Preprocessing: OpenCV histogram equalization, Gaussian blur, morphological ops
- Model evaluation: confusion matrix, ROC-AUC, precision, recall, F1
- Deployment via **Flask + Gunicorn + Streamlit** web interface
- Activation map visualization (Grad-CAM) for model interpretability

```
Skills: TensorFlow, OpenCV, NumPy, Flask, Streamlit, Grad-CAM
Metrics: ROC-AUC evaluated, production Streamlit deployment
```

---

### 5. 🐦 Multi-Model Sentiment Analysis of Tweets (Apr 2024)

- 5 ML models: SVM, Naive Bayes, Random Forest, Logistic Regression, XGBoost
- Text preprocessing: emojis, slang, abbreviations, negation handling
- Vectorization: Count Vectors, TF-IDF, Word2Vec, GloVe, FastText, ELMo
- Hyperparameter tuning + cross-validation + ensemble methods
- Evaluation: confusion matrix, F1, AUC

```
Skills: Python, NLP, Scikit-Learn, Word2Vec, GloVe, NLTK
```

---

### 6. 🎬 Content Recommendation System (Mar 2024)

- Collaborative filtering + matrix factorization for personalized recommendations
- TF-IDF and Word2Vec for content-based similarity
- EDA on large datasets to ensure data quality
- SQL + Python integration for client analytics platforms
- Evaluation: Precision@K, Recall@K, F1, ROC AUC

```
Skills: Python, SQL, Collaborative Filtering, TF-IDF, Word2Vec
```

---

### 7. 🌿 Climate Pollution Meter (Sep–Oct 2023)

**Associated with Loyalist College**

- Air quality analysis for 6 Ontario cities (Toronto, Mississauga, Brampton)
- Pollutants tracked: SO₂, NOx, CO, Ground-Level Ozone, PM2.5
- Time-series analysis of daily pollutant concentrations
- Interactive visualizations and actionable insight generation

```
Skills: Python, Pandas, ML, Data Visualization
```

---

### 8. 🫀 Heart Failure & Breast Cancer Prediction (Jun–Aug 2023)

**Associated with Loyalist College**

Comparative analysis of two predictive models using Logistic Regression:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Heart Failure | 85.4% | 81.2% | 91.7% | 86.2% |
| **Breast Cancer** | **97.4%** | **98.0%** | **95.3%** | **96.6%** |

- ROC curves, precision-recall analysis
- Breast Cancer model outperformed across all metrics

```
Skills: Python, Scikit-Learn, Logistic Regression, Data Preprocessing
```

---

### 9. 🚗 Automatic Number Plate Recognition (Jan–May 2021)

**Associated with Gujarat Technological University**

- GUI-based ANPR system with 98.6% prediction accuracy
- Pipeline: Image preprocessing → License plate segmentation → OCR
- Returns plate number + owner name, pin code, city
- OpenCV-based image processing with Tesseract OCR

```
Skills: Python, OpenCV, Tesseract OCR, Image Segmentation
Metrics: 98.6% ANPR accuracy
```

---

### 10. 📉 Customer Churn Prediction (Jul–Dec 2020)

- Supervised ML model for customer churn classification
- Feature selection, engineering, and preprocessing pipeline
- Integrated into client-facing analytics platform
- SQL for efficient data management and extraction

```
Skills: Python, SQL, Scikit-Learn, Feature Engineering, Data Science
```

---

### 11. ₿ Bitcoin Price Prediction

**Associated with Gujarat Technological University**

- 5 ML models with 95% prediction accuracy
- User-friendly web interface for buy/sell signals
- Time-series feature engineering

```
Skills: NumPy, Scikit-Learn, Pandas, Flask
Metrics: 95% prediction accuracy
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow, Keras, PyTorch |
| ML | Scikit-Learn, XGBoost, LightGBM |
| Computer Vision | OpenCV, PIL, MTCNN, Grad-CAM |
| NLP | NLTK, spaCy, Word2Vec, GloVe, Transformers |
| Data | Pandas, NumPy, PySpark, SQL |
| Deployment | Flask, Streamlit, Docker, Gunicorn |
| Visualization | Matplotlib, Seaborn, Plotly, Tableau, Power BI |

---

## 📫 Contact

- **LinkedIn**: [Rutvik Trivedi](https://linkedin.com/in/rutviktrivedi29)
- **GitHub**: [@rutvik29](https://github.com/rutvik29)
