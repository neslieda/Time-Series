# Time Series Analysis for Bicycle Usage Prediction in Eskişehir

## ⚠️ IMPORTANT
- This is a private repository containing proprietary analysis and models
- The dataset used in this project is private and not shared publicly due to data protection requirements
- The code and methodology are specifically designed for the Eskişehir bicycle usage prediction case
- For any inquiries about the project or collaboration requests, please contact the repository owner directly
# Time-Series Analysis for Bicycle Usage Prediction

This project involves time series analysis techniques to predict daily bicycle usage in different regions of Eskişehir. Various models such as LSTM, ARIMA, SARIMA, XGBoost, and SSA were tested.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Models](#models)
  - [LSTM](#lstm)
  - [ARIMA](#arima)
  - [SARIMA](#sarima)
  - [XGBoost](#xgboost)
  - [SSA](#ssa)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Bicycle usage prediction is a crucial task for urban planning and transportation management. This project aims to predict daily bicycle usage in different regions of Eskişehir using various time series analysis techniques. The models tested in this project include LSTM, ARIMA, SARIMA, XGBoost, and SSA.

## Project Structure
The repository is structured as follows:
```
Time-Series/
│
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   └── external/           # External data sources
│
├── notebooks/
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── model_training.ipynb# Model training and evaluation
│   └── ...                 # Other notebooks
│
├── src/
│   ├── data_preparation.py # Data preprocessing scripts
│   ├── models/
│   │   ├── lstm.py         # LSTM model definition and training
│   │   ├── arima.py        # ARIMA model definition and training
│   │   ├── sarima.py       # SARIMA model definition and training
│   │   ├── xgboost.py      # XGBoost model definition and training
│   │   └── ssa.py          # SSA model definition and training
│   ├── evaluation.py       # Model evaluation scripts
│   └── utils.py            # Utility functions
│
├── tests/
│   └── test_models.py      # Unit tests for models
│
├── README.md               # Project description and usage
├── requirements.txt        # Python dependencies
└── setup.py                # Project setup script
```

## Installation
To run this project, you need to have Python installed on your system. Follow the steps below to set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/neslieda/Time-Series.git
    cd Time-Series
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Data Preparation
1. Place your raw data files in the `data/raw/` directory.

2. Run the data preparation script to process the raw data:
    ```bash
    python src/data_preparation.py
    ```

### Model Training
1. Open the `notebooks/model_training.ipynb` notebook.

2. Follow the instructions in the notebook to train and evaluate the models.

### Model Evaluation
To evaluate a trained model, you can use the evaluation script:
```bash
python src/evaluation.py --model LSTM --data data/processed/bicycle_usage.csv
```

## Models
### LSTM
**LSTM (Long Short-Term Memory)** is a type of recurrent neural network (RNN) suitable for sequential data. It is capable of learning long-term dependencies and is widely used in time series forecasting.

Model definition and training scripts can be found in `src/models/lstm.py`.

### ARIMA
**ARIMA (AutoRegressive Integrated Moving Average)** is a class of models that explains a given time series based on its past values. It is a widely used statistical model for time series forecasting.

Model definition and training scripts can be found in `src/models/arima.py`.

### SARIMA
**SARIMA (Seasonal ARIMA)** is an extension of ARIMA that supports seasonality. It is useful for modeling and forecasting data with seasonal patterns.

Model definition and training scripts can be found in `src/models/sarima.py`.

### XGBoost
**XGBoost** is an optimized gradient boosting library designed to be highly efficient and flexible. It is widely used for regression and classification tasks, including time series forecasting.

Model definition and training scripts can be found in `src/models/xgboost.py`.

### SSA
**SSA (Singular Spectrum Analysis)** is a technique for analyzing time series data. It decomposes the series into a sum of interpretable components such as trend, oscillatory components, and noise.

Model definition and training scripts can be found in `src/models/ssa.py`.

## Results
The results of the model evaluations are summarized as follows:
- **LSTM**: Achieved the best performance with a mean absolute error (MAE) of X.
- **ARIMA**: Showed good performance with an MAE of Y.
- **SARIMA**: Improved performance with seasonality, achieving an MAE of Z.
- **XGBoost**: Performed well with an MAE of W.
- **SSA**: Showed moderate performance with an MAE of V.

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
