# Time Series Analysis for Bicycle Usage Prediction in Eskişehir

## ⚠️ IMPORTANT
- This is a private repository containing proprietary analysis and models
- The dataset used in this project is private and not shared publicly due to data protection requirements
- The code and methodology are specifically designed for the Eskişehir bicycle usage prediction case
- For any inquiries about the project or collaboration requests, please contact the repository owner directly

## Overview
This project implements various time series analysis techniques to predict daily bicycle usage patterns across different regions of Eskişehir, Turkey. The project utilizes multiple advanced modeling approaches to achieve accurate predictions and compare their performances.

## Models Implemented
- **LSTM (Long Short-Term Memory)**: Deep learning approach for sequence prediction
- **ARIMA (Autoregressive Integrated Moving Average)**: Statistical method for time series forecasting
- **SARIMA (Seasonal ARIMA)**: Extension of ARIMA that handles seasonal components
- **XGBoost**: Gradient boosting framework adapted for time series
- **SSA (Singular Spectrum Analysis)**: Non-parametric technique for time series decomposition and forecasting

## Project Structure
```
Time-Series/
│
├── data/                  # Data files and datasets (private)
├── models/               # Implemented models
├── notebooks/           # Jupyter notebooks for analysis
├── src/                 # Source code
├── results/             # Output files and visualizations
└── requirements.txt     # Project dependencies
```

## Requirements
- Python 3.7+
- Required packages:
  ```
  numpy
  pandas
  scikit-learn
  tensorflow
  xgboost
  statsmodels
  matplotlib
  seaborn
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/neslieda/Time-Series.git
   cd Time-Series
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Data Preparation:
   - Place your bicycle usage data in the `data/` directory
   - Run data preprocessing scripts:
     ```bash
     python src/preprocess.py
     ```

2. Model Training:
   ```bash
   python src/train_models.py --model [lstm|arima|sarima|xgboost|ssa]
   ```

3. Prediction:
   ```bash
   python src/predict.py --model [model_name] --input [input_data]
   ```

## Model Details

### LSTM
- Implemented using TensorFlow/Keras
- Suitable for capturing long-term dependencies in the data
- Handles multivariate time series data

### ARIMA/SARIMA
- Statistical approaches for time series forecasting
- SARIMA specifically handles seasonal patterns in bicycle usage
- Implemented using statsmodels library

### XGBoost
- Gradient boosting framework adapted for time series
- Features engineering for temporal aspects
- Handles both linear and non-linear relationships

### SSA (Singular Spectrum Analysis)
- Non-parametric spectral estimation method
- Useful for trend extraction and noise reduction
- Helps in understanding underlying patterns

## Results
The project compares the performance of different models based on:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²) values
- Prediction accuracy on test data

## Visualization
The project includes various visualizations:
- Time series decomposition plots
- Prediction vs. actual usage comparisons
- Error distribution analysis
- Seasonal pattern visualization

## Contributing
Due to the private nature of this project, contributions are currently limited to authorized team members. For any suggestions or issues, please contact the repository owner.

## License
This project is private and proprietary. All rights reserved.

## Acknowledgments
- Eskişehir Municipality for providing the bicycle usage data
- Contributors and researchers in the field of time series analysis
- Open source community for various tools and libraries used in this project
