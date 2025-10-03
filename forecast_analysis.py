from __future__ import annotations
"""
Forecast Skill - Multi-model forecasting with automatic model selection
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from skill_framework import skill, SkillParameter, SkillInput, SkillOutput, SkillVisualization
from skill_framework.layouts import wire_layout
from answer_rocket import AnswerRocketClient
from ar_analytics.helpers.utils import get_dataset_id
import json

# Database ID for pasta dataset - required for SQL queries
# This is different from DATASET_ID and must be set correctly for each environment
DATABASE_ID = "83c2268f-af77-4d00-8a6b-7181dc06643e"


@skill(
    name="Forecast Analysis",
    llm_name="forecast_analysis",
    description="Generate intelligent forecasts using best-fit model selection with automatic model optimization",
    capabilities="Provides multi-model forecasting with automatic selection of best-performing algorithm. Supports linear regression, moving average, and other forecasting models. Generates confidence intervals, trend analysis, and seasonality detection. Creates professional visualizations with KPIs, charts, and insights.",
    limitations="Requires minimum 12 historical data points. Limited to 36 months forecast horizon. Assumes monthly granularity (month_new). Performance depends on data quality and historical patterns.",
    example_questions="What will sales be over the next 6 months? Can you forecast volume for Q1 2024? Show me a 12-month revenue projection with confidence intervals. What's the expected growth trend for the next quarter?",
    parameter_guidance="Select metric to forecast (sales, volume, etc.). Choose forecast steps (1-36 months, default 6). Optionally filter by date range or apply dimensional filters. The skill automatically selects the best forecasting model based on historical performance.",
    parameters=[
        SkillParameter(
            name="metric",
            constrained_to="metrics",
            description="The metric to forecast"
        ),
        SkillParameter(
            name="forecast_steps",
            description="Number of periods to forecast (months)",
            default_value=6
        ),
        SkillParameter(
            name="start_date",
            description="Start date for training data (YYYY-MM-DD)",
            default_value="2022-01-01"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            description="Additional filters to apply to the data"
        ),
        SkillParameter(
            name="confidence_level",
            description="Confidence level for prediction intervals",
            default_value=0.95
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt being used for max response.",
            default_value="Answer user question in 30 words or less using following facts:\n{{facts}}"
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed insights.",
            default_value="Analyze the forecast data and provide insights about trends, seasonality, and recommendations for planning."
        ),
        SkillParameter(
            name="forecast_viz_layout",
            parameter_type="visualization",
            description="Forecast Visualization Layout",
            default_value=None
        )
    ]
)
def forecast_analysis(parameters: SkillInput) -> SkillOutput:
    """
    Forecast Analysis skill

    Generates intelligent forecasts using multi-model selection and automatic optimization.
    """
    return run_forecast_analysis(parameters)

def run_forecast_analysis(parameters: SkillInput) -> SkillOutput:
    """
    Main forecast analysis logic
    """
    try:
        print(f"DEBUG: Starting forecast analysis")
        print(f"DEBUG: Raw parameters.arguments: {parameters.arguments}")

        # Extract parameters
        metric = parameters.arguments.metric
        forecast_steps = getattr(parameters.arguments, 'forecast_steps', 6)
        start_date = getattr(parameters.arguments, 'start_date', None)
        other_filters = getattr(parameters.arguments, 'other_filters', [])
        confidence_level = getattr(parameters.arguments, 'confidence_level', 0.95)

        print(f"DEBUG: Extracted parameters:")
        print(f"  - metric: {metric} (type: {type(metric)})")
        print(f"  - forecast_steps: {forecast_steps} (type: {type(forecast_steps)})")
        print(f"  - start_date: {start_date} (type: {type(start_date)})")
        print(f"  - other_filters: {other_filters} (type: {type(other_filters)})")
        print(f"  - confidence_level: {confidence_level} (type: {type(confidence_level)})")

        # Convert string parameters to proper types if needed
        if isinstance(forecast_steps, str):
            forecast_steps = int(forecast_steps)
            print(f"DEBUG: Converted forecast_steps to int: {forecast_steps}")

        if isinstance(confidence_level, str):
            confidence_level = float(confidence_level)
            print(f"DEBUG: Converted confidence_level to float: {confidence_level}")

        # Validate inputs
        print(f"DEBUG: Validating forecast_steps: {forecast_steps}")
        if forecast_steps < 1 or forecast_steps > 36:
            print(f"DEBUG: Validation failed - forecast_steps out of range")
            return SkillOutput(
                final_prompt="Invalid forecast steps. Please specify between 1 and 36 months.",
                warnings=["Forecast steps must be between 1 and 36 months"]
            )

        print(f"DEBUG: Calling fetch_data with metric={metric}, start_date={start_date}")

        # Get data from AnswerRocket
        data_df = fetch_data(metric, start_date, other_filters)

        print(f"DEBUG: fetch_data returned, data_df type: {type(data_df)}")
        print(f"DEBUG: data_df is None: {data_df is None}")
        if data_df is not None:
            print(f"DEBUG: data_df.empty: {data_df.empty if hasattr(data_df, 'empty') else 'No empty attribute'}")
            print(f"DEBUG: data_df shape: {data_df.shape if hasattr(data_df, 'shape') else 'No shape attribute'}")

        if data_df is None or data_df.empty:
            return SkillOutput(
                final_prompt="No data available for the specified metric and time range.",
                warnings=["Unable to retrieve data"]
            )

        # Check minimum data requirements
        if len(data_df) < 12:
            return SkillOutput(
                final_prompt=f"Insufficient historical data. Need at least 12 data points but only have {len(data_df)}.",
                warnings=["Minimum 12 historical data points required for forecasting"]
            )

        # Analyze historical patterns
        patterns = analyze_patterns(data_df)

        # Run multiple models
        model_results = run_models(data_df, forecast_steps, confidence_level)

        if not model_results:
            return SkillOutput(
                final_prompt="Unable to generate forecast. All models failed to converge.",
                warnings=["Model fitting failed"]
            )

        # Select best model
        best_model, best_results = select_best_model(model_results)

        # Prepare output data
        output_df = prepare_output(data_df, best_results, best_model, patterns, model_results)

        # Generate prompt for LLM
        prompt = generate_prompt(
            metric=metric,
            forecast_steps=forecast_steps,
            best_model=best_model,
            patterns=patterns,
            model_results=model_results,
            forecast_stats=calculate_forecast_stats(best_results)
        )

        # Create visualizations
        visualizations = create_visualizations(
            output_df=output_df,
            metric=metric,
            best_model=best_model,
            patterns=patterns,
            model_results=model_results,
            forecast_stats=calculate_forecast_stats(best_results)
        )

        return SkillOutput(
            final_prompt=prompt,
            visualizations=visualizations,
            narrative=None
        )

    except Exception as e:
        # Don't throw exceptions - return user-friendly error
        print(f"DEBUG: EXCEPTION OCCURRED: {str(e)}")
        print(f"DEBUG: Exception type: {type(e)}")
        import traceback
        print(f"DEBUG: Full traceback:\n{traceback.format_exc()}")
        return SkillOutput(
            final_prompt=f"An error occurred while generating the forecast. Please check your data and try again. Error: {str(e)}",
            warnings=[f"Error: {str(e)}"]
        )

def fetch_data(metric, start_date, other_filters):
    """
    Fetch data using SQL query
    """
    import os

    print(f"DEBUG: fetch_data called with metric={metric}, start_date={start_date}, filters={other_filters}")

    # Create AnswerRocket client
    try:
        arc = AnswerRocketClient()
        print(f"DEBUG: Created AnswerRocketClient successfully")
    except Exception as e:
        print(f"DEBUG: Failed to create AnswerRocketClient: {e}")
        return None

    # Use DATABASE_ID - try environment variable first, then fall back to constant
    database_id = os.getenv('DATABASE_ID', DATABASE_ID)
    print(f"DEBUG: Using DATABASE_ID: {database_id}")

    # Build SQL query to get time series data for forecasting
    sql_query = f"""
    SELECT
        month_new,
        SUM({metric}) as {metric}
    FROM read_csv('pasta_2025.csv')
    WHERE 1=1
    """

    # Add date filter if provided
    if start_date:
        sql_query += f" AND month_new >= '{start_date}'"
        print(f"DEBUG: Added date filter: {start_date}")

    # Add other filters
    if other_filters:
        for filter_item in other_filters:
            if isinstance(filter_item, dict):
                for key, value in filter_item.items():
                    if isinstance(value, list):
                        values_str = "', '".join(str(v) for v in value)
                        sql_query += f" AND {key} IN ('{values_str}')"
                    else:
                        sql_query += f" AND {key} = '{value}'"
                    print(f"DEBUG: Added filter: {key} = {value}")

    sql_query += f"""
    GROUP BY month_new
    ORDER BY month_new
    """

    print(f"DEBUG: Executing SQL query:\n{sql_query}")

    try:
        # Execute SQL query using AnswerRocketClient
        print(f"DEBUG: About to execute SQL using database_id: {database_id}")
        result = arc.data.execute_sql_query(database_id, sql_query, 1000)
        print(f"DEBUG: SQL execution result - success: {result.success if hasattr(result, 'success') else 'No success attr'}")

        if hasattr(result, 'success') and result.success and hasattr(result, 'df'):
            raw_df = result.df
            print(f"DEBUG: SQL executed successfully, got shape: {raw_df.shape if raw_df is not None else 'None'}")
        else:
            print(f"DEBUG: SQL execution failed or no data")
            if hasattr(result, 'error'):
                print(f"DEBUG: Error: {result.error}")
            return None

        if raw_df is not None and not raw_df.empty:
            print(f"DEBUG: Raw data columns: {list(raw_df.columns)}")
            print(f"DEBUG: Raw data sample:\n{raw_df.head()}")

            # Standardize column names
            if 'month_new' in raw_df.columns:
                raw_df = raw_df.rename(columns={'month_new': 'period'})
                print(f"DEBUG: Renamed month_new to period")

            if metric in raw_df.columns:
                raw_df = raw_df.rename(columns={metric: 'value'})
                print(f"DEBUG: Renamed {metric} to value")

            print(f"DEBUG: Final columns: {list(raw_df.columns)}")
            print(f"DEBUG: Final shape: {raw_df.shape}")
        else:
            print(f"DEBUG: No data returned from SQL query")

        return raw_df

    except Exception as e:
        print(f"DEBUG: SQL execution failed: {str(e)}")
        return None

def analyze_patterns(df):
    """
    Analyze historical data patterns
    """
    from scipy import stats

    values = df['value'].values
    x = np.arange(len(values))

    # Trend analysis
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

    # Volatility
    returns = pd.Series(values).pct_change().dropna()
    volatility = np.std(returns)

    # Seasonality (simple check)
    seasonal_strength = 0
    if len(values) >= 24:
        monthly_avgs = [np.mean(values[i::12]) for i in range(min(12, len(values)))]
        seasonal_strength = np.std(monthly_avgs) / np.mean(monthly_avgs) if np.mean(monthly_avgs) > 0 else 0

    return {
        'trend_slope': slope,
        'trend_r2': r_value ** 2,
        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
        'volatility': volatility,
        'volatility_level': 'high' if volatility > 0.2 else 'medium' if volatility > 0.1 else 'low',
        'has_seasonality': seasonal_strength > 0.1,
        'data_points': len(values)
    }

def run_models(df, periods, confidence_level):
    """
    Run multiple forecasting models using statsmodels and prophet
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from prophet import Prophet
    import warnings
    warnings.filterwarnings('ignore')

    # Prepare data
    df = df.copy()

    # Ensure period is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['period']):
        df['period'] = pd.to_datetime(df['period'])

    df['period_num'] = range(len(df))

    # Split for validation
    train_size = int(len(df) * 0.8)
    train = df[:train_size]
    validate = df[train_size:]

    models = {}

    # 1. Prophet Model
    try:
        # Prepare data for Prophet
        prophet_df = train[['period', 'value']].rename(columns={'period': 'ds', 'value': 'y'})

        # Create and fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=confidence_level
        )
        model.fit(prophet_df)

        # Validate
        future_val = pd.DataFrame({'ds': validate['period']})
        val_forecast = model.predict(future_val)
        val_pred = val_forecast['yhat'].values

        # Calculate metrics
        mae = np.mean(np.abs(validate['value'].values - val_pred))
        mape = np.mean(np.abs((validate['value'].values - val_pred) / validate['value'].values)) * 100

        # Full forecast
        future_dates = pd.date_range(
            start=df['period'].iloc[-1] + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)

        models['prophet'] = {
            'forecast': forecast['yhat'].values,
            'lower_bound': forecast['yhat_lower'].values,
            'upper_bound': forecast['yhat_upper'].values,
            'mae': mae,
            'mape': mape,
            'score': mae
        }
    except Exception as e:
        print(f"DEBUG: Prophet model failed: {e}")

    # 2. Holt-Winters Exponential Smoothing
    try:
        # Determine seasonality
        seasonal_periods = 12 if len(train) >= 24 else None

        if seasonal_periods:
            model = ExponentialSmoothing(
                train['value'],
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add'
            )
        else:
            model = ExponentialSmoothing(
                train['value'],
                trend='add'
            )

        fitted_model = model.fit()

        # Validate
        val_pred = fitted_model.forecast(len(validate))
        mae = np.mean(np.abs(validate['value'].values - val_pred))
        mape = np.mean(np.abs((validate['value'].values - val_pred) / validate['value'].values)) * 100

        # Full forecast
        forecast = fitted_model.forecast(periods)

        # Confidence intervals (simple approach)
        residuals = train['value'] - fitted_model.fittedvalues
        std_error = np.std(residuals)
        z_score = 1.96 if confidence_level == 0.95 else 2.58

        models['holt_winters'] = {
            'forecast': forecast.values if hasattr(forecast, 'values') else forecast,
            'lower_bound': forecast - z_score * std_error,
            'upper_bound': forecast + z_score * std_error,
            'mae': mae,
            'mape': mape,
            'score': mae
        }
    except Exception as e:
        print(f"DEBUG: Holt-Winters model failed: {e}")

    # 3. Simple Moving Average (fallback)
    try:
        window = min(3, len(train) // 4)
        ma_forecast = []
        recent_values = list(train['value'].values[-window:])

        for _ in range(len(validate)):
            pred = np.mean(recent_values)
            ma_forecast.append(pred)
            recent_values.pop(0)
            recent_values.append(pred)

        mae = np.mean(np.abs(validate['value'].values - np.array(ma_forecast)))
        mape = np.mean(np.abs((validate['value'].values - np.array(ma_forecast)) / validate['value'].values)) * 100

        # Full forecast
        recent_values = list(df['value'].values[-window:])
        full_forecast = []
        for _ in range(periods):
            pred = np.mean(recent_values)
            full_forecast.append(pred)
            recent_values.pop(0)
            recent_values.append(pred)

        models['moving_average'] = {
            'forecast': np.array(full_forecast),
            'lower_bound': np.array(full_forecast) * 0.9,
            'upper_bound': np.array(full_forecast) * 1.1,
            'mae': mae,
            'mape': mape,
            'score': mae
        }
    except Exception as e:
        print(f"DEBUG: Moving Average model failed: {e}")

    return models

def select_best_model(model_results):
    """
    Select the best performing model
    """
    best_model = None
    best_score = float('inf')

    for model_name, results in model_results.items():
        if results['score'] < best_score:
            best_score = results['score']
            best_model = model_name

    return best_model, model_results[best_model]

def prepare_output(historical_df, forecast_results, model_name, patterns, all_models):
    """
    Prepare output DataFrame
    """
    # Historical data
    hist_df = historical_df.copy()

    # Convert period to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(hist_df['period']):
        hist_df['period'] = pd.to_datetime(hist_df['period'])

    hist_df['type'] = 'historical'
    hist_df['forecast'] = np.nan
    hist_df['lower_bound'] = np.nan
    hist_df['upper_bound'] = np.nan
    hist_df.rename(columns={'value': 'actual'}, inplace=True)

    # Forecast data
    forecast_periods = pd.date_range(
        start=hist_df['period'].iloc[-1] + pd.DateOffset(months=1),
        periods=len(forecast_results['forecast']),
        freq='M'
    )

    forecast_df = pd.DataFrame({
        'period': forecast_periods,
        'type': 'forecast',
        'actual': np.nan,
        'forecast': forecast_results['forecast'],
        'lower_bound': forecast_results['lower_bound'],
        'upper_bound': forecast_results['upper_bound']
    })

    # Combine
    output_df = pd.concat([hist_df, forecast_df], ignore_index=True)

    # Add metadata
    output_df['model_used'] = model_name
    output_df['trend_direction'] = patterns['trend_direction']

    return output_df

def calculate_forecast_stats(results):
    """
    Calculate summary statistics for the forecast
    """
    forecast = results['forecast']
    return {
        'total': np.sum(forecast),
        'average': np.mean(forecast),
        'min': np.min(forecast),
        'max': np.max(forecast),
        'growth': ((forecast[-1] - forecast[0]) / forecast[0] * 100) if forecast[0] != 0 else 0
    }

def generate_prompt(metric, forecast_steps, best_model, patterns, model_results, forecast_stats):
    """
    Generate prompt for LLM interpretation
    """
    prompt = f"""
    Analyze this forecast for {metric}:

    FORECAST SUMMARY:
    - Forecast steps: {forecast_steps} months
    - Model selected: {best_model}
    - Total forecasted: {forecast_stats['total']:,.0f}
    - Average per period: {forecast_stats['average']:,.0f}
    - Growth: {forecast_stats['growth']:.1f}%

    HISTORICAL PATTERNS:
    - Trend: {patterns['trend_direction']} (R²={patterns['trend_r2']:.3f})
    - Volatility: {patterns['volatility_level']}
    - Seasonality: {'Yes' if patterns['has_seasonality'] else 'No'}
    - Data points used: {patterns['data_points']}

    MODEL COMPARISON:
    """

    for model_name, results in model_results.items():
        selected = " (SELECTED)" if model_name == best_model else ""
        prompt += f"\n    - {model_name}: MAE={results['mae']:.0f}, MAPE={results['mape']:.1f}%{selected}"

    prompt += """

    Please provide insights about:
    1. Why this model was most appropriate
    2. Key trends and what to expect
    3. Confidence level and risks
    4. Recommendations for planning
    """

    return prompt

def create_visualizations(output_df, metric, best_model, patterns, model_results, forecast_stats):
    """
    Create visualizations for forecast results
    """
    visualizations = []

    # Prepare data for chart
    historical = output_df[output_df['type'] == 'historical'].copy()
    forecast = output_df[output_df['type'] == 'forecast'].copy()

    # Format dates for chart
    historical['period_str'] = pd.to_datetime(historical['period']).dt.strftime('%Y-%m')
    forecast['period_str'] = pd.to_datetime(forecast['period']).dt.strftime('%Y-%m')

    # Create series data for Highcharts
    historical_series = {
        "name": "Historical",
        "data": [{"x": i, "y": float(val), "name": date}
                 for i, (val, date) in enumerate(zip(historical['actual'], historical['period_str']))],
        "color": "#2E86C1",
        "marker": {"enabled": True, "radius": 4}
    }

    forecast_series = {
        "name": "Forecast",
        "data": [{"x": len(historical) + i, "y": float(val), "name": date}
                 for i, (val, date) in enumerate(zip(forecast['forecast'], forecast['period_str']))],
        "color": "#E74C3C",
        "dashStyle": "Dash",
        "marker": {"enabled": True, "radius": 4}
    }

    # Confidence interval series
    lower_bound_series = {
        "name": "Lower Bound",
        "data": [{"x": len(historical) + i, "y": float(val)}
                 for i, val in enumerate(forecast['lower_bound'])],
        "color": "rgba(231, 76, 60, 0.2)",
        "lineWidth": 0,
        "marker": {"enabled": False},
        "enableMouseTracking": False
    }

    upper_bound_series = {
        "name": "Upper Bound",
        "data": [{"x": len(historical) + i, "y": float(val)}
                 for i, val in enumerate(forecast['upper_bound'])],
        "color": "rgba(231, 76, 60, 0.2)",
        "fillOpacity": 0.3,
        "lineWidth": 0,
        "marker": {"enabled": False},
        "type": "arearange",
        "linkedTo": ":previous"
    }

    # All categories for x-axis
    all_categories = list(historical['period_str']) + list(forecast['period_str'])

    # Create Highcharts configuration
    chart_config = {
        "type": "highcharts",
        "config": {
            "chart": {"type": "line", "height": 400},
            "title": {"text": f"{metric.title()} Forecast - {best_model.replace('_', ' ').title()} Model"},
            "xAxis": {
                "categories": all_categories,
                "title": {"text": "Period"}
            },
            "yAxis": {
                "title": {"text": metric.title()},
                "labels": {"format": "{value:,.0f}"}
            },
            "tooltip": {
                "shared": True,
                "valueDecimals": 0,
                "valuePrefix": "$"
            },
            "series": [historical_series, forecast_series],
            "plotOptions": {
                "line": {
                    "marker": {"enabled": True}
                }
            }
        }
    }

    # Create layout with chart
    layout = {
        "layoutJson": {
            "type": "Document",
            "style": {
                "backgroundColor": "#ffffff",
                "padding": "20px"
            },
            "children": [
                {
                    "type": "Header",
                    "text": f"Forecast Analysis: {metric.title()}",
                    "style": {
                        "fontSize": "24px",
                        "fontWeight": "bold",
                        "marginBottom": "20px",
                        "color": "#2C3E50"
                    }
                },
                {
                    "type": "HighchartsChart",
                    "options": chart_config["config"]
                },
                {
                    "type": "Paragraph",
                    "text": f"**Model Used:** {best_model.replace('_', ' ').title()}",
                    "style": {"marginTop": "20px", "fontSize": "14px"}
                },
                {
                    "type": "Paragraph",
                    "text": f"**Model Accuracy:** {model_results[best_model]['mape']:.1f}% MAPE",
                    "style": {"fontSize": "14px"}
                },
                {
                    "type": "Paragraph",
                    "text": f"**Trend:** {patterns['trend_direction'].title()} (R² = {patterns['trend_r2']:.3f})",
                    "style": {"fontSize": "14px"}
                },
                {
                    "type": "Paragraph",
                    "text": f"**Total Forecasted:** ${forecast_stats['total']:,.0f}",
                    "style": {"fontSize": "14px", "fontWeight": "bold"}
                }
            ]
        },
        "inputVariables": []
    }

    rendered = wire_layout(layout, {})
    visualizations.append(SkillVisualization(title="Forecast", layout=rendered))

    return visualizations


if __name__ == '__main__':
    # Load environment variables for local testing
    from dotenv import load_dotenv
    load_dotenv('/Users/mitchelltravis/cursor/.env')

    skill_input: SkillInput = forecast_analysis.create_input(arguments={
        'metric': "sales",
        'forecast_steps': 6,
        'start_date': "2022-01-01",
        'other_filters': None
    })

    print("=" * 80)
    print("TESTING FORECAST ANALYSIS SKILL")
    print("=" * 80)

    result = forecast_analysis(skill_input)

    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(f"Final Prompt: {result.final_prompt}")
    if hasattr(result, 'warnings') and result.warnings:
        print(f"Warnings: {result.warnings}")
    print("=" * 80)