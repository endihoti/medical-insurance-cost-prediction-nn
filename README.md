# Medical Insurance Cost Prediction (Neural Networks & Deep Learning)

A deep learning regression pipeline built with Keras and TensorFlow to predict individual medical insurance charges based on demographic and lifestyle factors.

## Project Overview
- **Dataset:** 1,338 records with 6 features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) and target variable `charges`.
- **Preprocessing:** Train/test split (80/20) performed **prior** to scaling to prevent data leakage, followed by one-hot encoding for categorical features and `StandardScaler` normalization for numerical features and target values.
- **Modeling:** Iterative experimentation across baseline and regularized Multi-Layer Perceptron (MLP) architectures using Adam optimizer, Huber/MSE loss, and EarlyStopping.
- **Evaluation:** Quantitative test metrics (MAE, RMSE, $R^2$), training/validation loss curves, and actual vs. predicted scatter analysis.

## Repository Structure
- `insurance_cost_prediction.ipynb`: Complete Jupyter Notebook covering EDA, preprocessing, model experiments, evaluation, visualizations, and summary findings.
- `requirements.txt`: Python dependencies (`tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`).

## Quick Start
```bash
pip install -r requirements.txt
jupyter notebook insurance_cost_prediction.ipynb
