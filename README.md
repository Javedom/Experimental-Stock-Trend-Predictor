# Stock Price Trend Prediction Model

A machine learning application that predicts quarterly stock price trends (up or down) for a given set of ticker symbols. This model analyzes historical stock data and financial metrics to make predictions about future price movements.

## Features

- **Data Collection**: Automatically fetches historical stock price data and financial ratios from Yahoo Finance
- **Feature Engineering**: Calculates quarterly financial metrics, technical indicators, and combines them with fundamental ratios
- **Machine Learning**: Uses Random Forest classifier to predict if a stock price will go up or down in the next quarter
- **Hyperparameter Optimization**: Implements both Grid Search and Bayesian Optimization (via Optuna) to find the best model parameters
- **Evaluation**: Provides detailed model performance metrics including accuracy, precision, recall, F1 score, ROC curve, and confusion matrix
- **Feature Analysis**: Analyzes and visualizes feature importance and correlations
- **Prediction**: Makes predictions for the upcoming quarter with probability scores
- **Visualization**: Generates plots for model evaluation and feature analysis

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The script can be run with various command-line arguments to customize its behavior:

```bash
python StockPricePredictor.py [OPTIONS]
```

### Options

- `--tickers`: List of ticker symbols to analyze (default: AAPL, MSFT, GOOG, AMZN, META, TSLA, NVDA)
- `--start-date`: Start date for historical data in YYYY-MM-DD format (default: 2016-01-01)
- `--end-date`: End date for historical data in YYYY-MM-DD format (default: current date)
- `--model-file`: Path to save/load the trained model (default: stock_price_model.pkl)
- `--predict-only`: Only make predictions using an existing model (no training)
- `--feature-importance`: Only analyze feature importance of an existing model
- `--historical-only`: Only use historical and technical data (ignore current snapshot metrics)

### Examples

#### Train a new model and make predictions

```bash
python stock_trend_prediction.py --tickers AAPL MSFT GOOG --start-date 2018-01-01

Important Considerations for Nasdaq Helsinki Stocks:

Ticker Format: You must use the correct Yahoo Finance format for Helsinki stocks (with the .HE suffix). For example:

Nokia = NOKIA.HE
Stora Enso = STERV.HE
Kone = KNEBV.HE
```

#### Use an existing model to make predictions only

```bash
python stock_trend_prediction.py --tickers AAPL NVDA AMD --predict-only
```

#### Analyze feature importance of an existing model

```bash
python stock_trend_prediction.py --feature-importance
```

#### Use only historical data (no current snapshot data)

```bash
python stock_trend_prediction.py --historical-only
```

## Output Files

The script generates several output files:

- `stock_price_model.pkl`: Saved trained model (can be customized with `--model-file`)
- `stock_trend_predictions.xlsx`: Excel file with predicted trends and probabilities
- `feature_importance.csv`: CSV file with feature importance scores
- `raw_quarterly_data.xlsx`: Raw financial data used for training
- `feature_importance.png`: Bar chart of top feature importance
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `feature_distributions.png`: Box plots of feature distributions by trend
- `top_correlation_matrix.png`: Heatmap of correlations between top features

## How It Works

1. **Data Collection**: The script fetches historical stock price data and financial information for the specified tickers using Yahoo Finance's API.

2. **Feature Engineering**: 
   - Calculates quarterly prices and returns
   - Extracts financial ratios from balance sheets, income statements, and cash flow statements
   - Combines price data with financial fundamentals
   - Calculates technical indicators (moving averages, momentum, volatility)

3. **Model Training**:
   - Preprocesses data (handling missing values, scaling features)
   - Performs cross-validation to assess model stability
   - Optimizes hyperparameters using Grid Search and Optuna
   - Trains a Random Forest classifier on historical data

4. **Evaluation**: Evaluates model performance using various metrics and visualizations

5. **Prediction**: Uses the trained model to predict whether each stock will go up or down in the next quarter

## Notes

- The model uses both historical financial data and current snapshot data from Yahoo Finance
- For best results, use a diverse set of stocks with at least 3-5 years of historical data
- The `--historical-only` flag may be useful if you believe current snapshot data may introduce bias
- Predictions are probabilistic - consider the confidence level when making decisions

## Requirements

See requirements.txt for the complete list of dependencies.
