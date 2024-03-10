# Stock Price Prediction Model

## Overview
This project aims to predict stock prices using Long Short-Term Memory (LSTM) neural networks. The model is trained on historical stock price data retrieved from Yahoo Finance. After preprocessing and scaling the data, the LSTM model is constructed and trained to forecast future stock prices based on past price movements.


## Table of Contents
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Files](#files)
- [Libraries Used](#libraries-used)
- [License](#license)

## Setup
1. **Install Dependencies**: Begin by installing the necessary Python packages:


## Data Preprocessing
- **Data Retrieval**: Historical stock price data is obtained from Yahoo Finance using the `yfinance` library.
- **Data Cleaning**: The data is cleaned and preprocessed by resetting the index and handling missing values.
- **Feature Engineering**: Additional features such as moving averages are computed to capture relevant information from the data.
- **Data Splitting**: The dataset is split into training and testing sets for model evaluation.

## Model Building
- **LSTM Architecture**: The LSTM model is constructed with multiple layers of LSTM units to capture temporal dependencies in the data.
- **Regularization**: Dropout regularization is applied to prevent overfitting during training.
- **Model Compilation**: The model is compiled using the Adam optimizer and Mean Squared Error loss function.
- **Training**: The model is trained on the training data for a specified number of epochs and batch size.

## Evaluation
- **Prediction**: The trained model is used to make predictions on the testing data.
- **Evaluation Metrics**: Performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are calculated to assess the model's accuracy.
- **Visualization**: Predicted and actual stock prices are visualized using line plots to compare the model's predictions with ground truth values.

## Files
- `Stock_Price_Prediction.ipynb`: Jupyter Notebook containing the code for building and training the stock price prediction model.
- `requirements.txt`: List of Python dependencies required for running the code.
- `LICENSE`: MIT License file.

## Libraries Used
- numpy
- pandas
- matplotlib
- yfinance
- scikit-learn
- keras

