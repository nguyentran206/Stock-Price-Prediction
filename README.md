# Apple Stock Price Prediction using Deep Learning (LSTM)

## Project Overview
This project focuses on predicting the closing price of Apple Inc. (AAPL) stock by leveraging historical market data. Accurately forecasting stock price trends helps in making informed investment decisions and understanding market dynamics. The project approaches this time-series forecasting problem from a Deep Learning perspective, implementing a Long Short-Term Memory (LSTM) neural network with PyTorch to capture long-term temporal dependencies in stock prices.

## Dataset
* **Stock Data (`AAPL_data.csv`):** Historical Apple stock data, extracted from Yahoo Finance, containing daily trading records.

## Data Overview
The table below summarizes the dimensions and key features of the dataset used before sequencing:

| Dataset | Rows | Features | Key Information / Attributes |
| :--- | :--- | :--- | :--- |
| `AAPL_data.csv` | 5,035 | 7 | Date, Open, High, Low, Close, Adj Close, Volume |

## Data Preprocessing & Feature Engineering
To prepare the continuous time-series data for the deep learning model, the raw dataset underwent the following preprocessing pipeline:
* **Feature Selection:** Isolated the `Close` price column as the primary target and input feature for the time-series model.
* **Data Scaling:** Utilized `MinMaxScaler` to scale the closing prices to a standardized range of [0, 1] to ensure stable gradient updates and faster convergence during neural network training.
* **Time-Series Sequencing:** Converted the 1D dataset into overlapping sequences using a lookback window (`seq_length`) of 100 days. The model utilizes the past 100 days of closing prices to forecast the price for the 101st day.
* **Data Partitioning:** Chronologically split the sequential data into an 80% training set to train the model and a 20% test set to evaluate its predictive performance.
* **Tensor Transformation:** Converted the split NumPy sequences into PyTorch Tensors (`torch.from_numpy`) to fit the PyTorch training pipeline.

## Methodology & Model
The project utilizes a custom Recurrent Neural Network built with PyTorch, specifically leveraging the Long Short-Term Memory (LSTM) architecture:
* **Model Architecture (`LSTMModel`):** 
  - Input size: 1 (Closing price)
  - 2 hidden LSTM layers with a hidden state size of 64
  - A fully connected linear layer (`nn.Linear`) to map the hidden state to a single continuous output value.
* **Training Configuration:**
  - **Loss Function:** Mean Squared Error Loss (`nn.MSELoss()`)
  - **Optimizer:** Adam Optimizer with a learning rate of 0.001
  - **Epochs:** 100

## Results & Evaluation
The model demonstrated steady optimization and convergence over the 100 training epochs, with the MSE loss consistently decreasing:

### Training Progress Summary
* **Epoch [10/100]:** Loss: `0.0461`
* **Epoch [50/100]:** Loss: `0.0033`
* **Epoch [100/100]:** Loss: `0.0015`

The final model achieved a minimal training MSE loss of **0.0015**, indicating highly stable optimization.

### Visual Evaluation
To evaluate performance on unseen data, the predictions on the 20% test set were inverse-scaled back to their original USD values. The plotted comparison between the **Actual Price** and **Predicted Price** demonstrates that the LSTM model successfully replicates the overall market trajectory and closely follows the actual daily fluctuations of the AAPL stock price.

![Visualization](visualization/ouput.png)
## Project Structure

```text
├── data/
│   ├── AAPL_data.csv
|
├── sources/
│   ├── StockPricePrediction.ipynb
|
├── visualization/
│   ├── output.png
|
└── README.md