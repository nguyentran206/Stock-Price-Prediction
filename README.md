# Stock Price Prediction

## Project Overview
This project focuses on predicting the stock prices of Apple Inc. (AAPL) by leveraging historical market data and macroeconomic indicators. The goal is to build, evaluate, and compare various Machine Learning and Deep Learning models to determine the most accurate approach for time-series stock forecasting.

Key objectives of this project include:
- Integrating external macroeconomic factors (GDP, CPI) with historical stock data to enhance predictive power.
- Performing time-series feature engineering and data normalization.
- Implementing and comparing traditional ML models versus Neural Networks.
- Evaluating model performance using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Dataset
**Sources:** - Apple Inc. (AAPL) Historical Stock Prices
- Global GDP Data (1960 - 2023)
- Consumer Price Index (CPI)

The dataset integrates daily stock price movements with broader economic indicators to capture market trends more effectively.

## Data Overview
The raw data requires significant preprocessing before being fed into the models. Key steps include:
- **Data Merging:** Combining AAPL stock prices with global GDP and CPI datasets based on corresponding timelines.
- **Handling Missing Values:** Cleaning the data to ensure continuous time-series integrity.
- **Feature Scaling:** Applying `MinMaxScaler` to normalize the data, ensuring that large economic indicators do not dominate the stock price features during model training.

## Methodology & Models
To find the best predictive model, several algorithms were implemented and compared:
1. **Linear Regression:** Used as a baseline model to capture linear trends.
2. **Support Vector Regression (SVR):** For capturing non-linear patterns in the stock data.
3. **Decision Tree & Random Forest Regressor:** Ensemble methods to handle complex feature interactions.
4. **Feedforward Neural Network (FNN):** A deep learning approach using Dense layers to capture complex mapping.
5. **Recurrent Neural Network (RNN):** Specifically designed for sequential and time-series data.

## Project Structure

| File/Folder | Description |
|-------------|-------------|
| `data/` | Contains the raw datasets (AAPL stock prices, GDP, CPI). |
| `StockPricePrediction.ipynb` | The main Jupyter Notebook containing EDA, data preprocessing, model building, hyperparameter tuning, and evaluation. |
| `README.md` | Project documentation and overview. |

## Results & Evaluation
The models were evaluated primarily using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**. 

Key findings from the validation and testing phases:
- The **Feedforward Neural Network (FNN)** emerged as the best-performing model on the test set, achieving an outstanding **MSE of 0.00037** and a **MAE of 0.0166**.
- The **Linear Regression** model also showed surprisingly strong performance in capturing basic trends, achieving a validation MSE of **3.47e-06**.
- Time-series specific models like RNN required careful tuning to prevent overfitting, highlighting the importance of hyperparameter optimization.

## Tech Stack
- **Languages:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Linear Regression, SVR, Decision Trees, Random Forest, MinMaxScaler)
- **Deep Learning:** TensorFlow / Keras (Dense, SimpleRNN, LSTM, Dropout)
