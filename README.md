# Forecast Skill

A multi-model forecasting skill for AnswerRocket that automatically selects the best model based on historical data patterns.

## Features

- Tests multiple forecasting models (Linear, Moving Average, Exponential Smoothing, ARIMA, Prophet)
- Automatically selects the best model based on validation performance
- Provides confidence intervals
- Returns structured data for LLM interpretation
- Supports custom start dates for training data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
ANSWERROCKET_API_URL=your-instance-url
ANSWERROCKET_TOKEN=your-api-token
ANSWERROCKET_COPILOT_ID=your-copilot-id
```

3. Run tests:
```bash
python test_forecast.py
```

## Usage

The skill accepts the following parameters:
- `metric`: The metric to forecast
- `periods`: Number of periods to forecast (default: 12)
- `start_date`: Optional start date for training data
- `confidence_level`: Confidence level for intervals (default: 0.95)

## Project Structure

```
Forecast/
├── forecast_skill.py      # Main skill implementation
├── test_forecast.py        # Local testing script
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (not in git)
├── .gitignore            # Git ignore file
└── README.md             # This file
```