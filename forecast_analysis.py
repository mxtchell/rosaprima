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

from skill_framework import skill, SkillParameter, SkillInput, SkillOutput, SkillVisualization, ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from answer_rocket import AnswerRocketClient
from ar_analytics.helpers.utils import get_dataset_id
from ar_analytics import ArUtils
import json
import jinja2

# Database ID for rosaprima dataset - required for SQL queries
# This is different from DATASET_ID and must be set correctly for each environment
DATABASE_ID = "56aa551e-2913-4f3a-9e4c-2ab5896ac908"
DATASET_ID = "e5e7c8a3-a9ff-47d8-9b57-cda29401e625"


@skill(
    name="Forecast Analysis",
    llm_name="forecast_analysis",
    description="Generate intelligent forecasts using best-fit model selection with automatic model optimization",
    capabilities="Provides multi-model forecasting with automatic selection of best-performing algorithm. Supports linear regression, moving average, and other forecasting models. Generates confidence intervals, trend analysis, and seasonality detection. Creates professional visualizations with KPIs, charts, and insights.",
    limitations="Requires minimum 12 historical data points. Limited to 36 months forecast horizon. Assumes monthly granularity (InvoiceDate). Performance depends on data quality and historical patterns.",
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
            description="Prompt being used for max response (brief summary).",
            default_value="Provide a brief 2-3 sentence summary of the forecast results using the following facts:\n{{facts}}"
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed skill insights.",
            default_value="""Analyze the forecast results and provide comprehensive insights. The facts contain 'is_price_metric' which tells you if this is a price/rate metric (averaging) vs volume metric (summing).

If is_price_metric is true, focus on:
- Price trends, changes, and volatility
- Starting vs ending prices
- Price range expectations
- Do NOT mention "total forecasted value" or sum totals

If is_price_metric is false, focus on:
- Total volume/sales forecasted
- Average per period
- Growth in quantities/revenue

Format your response as:

## Forecast Summary
[Brief overview of forecast period and model selected. For prices: mention forecasted price range and average. For volumes: mention total forecasted value and average per period]

## Why the {{model_name}} Model Was Selected
[Explain why this model outperformed others based on accuracy metrics]

## Accuracy
[Discuss model accuracy metrics and reliability]

## Suitability
[Explain why this model is appropriate for the data patterns observed]

## Key Trends and Expectations
**Trend:** [Describe the overall trend direction and strength. For prices: describe price movement. For volumes: describe quantity/revenue growth]

**Seasonality:** [Discuss any seasonal patterns detected]

**Volatility:** [Comment on data stability and consistency]

## Confidence Level and Risks
**Confidence Level:** [High/Medium/Low based on MAPE]

**Risks to Watch For:**
- [List key risks - for prices: pricing pressure, competition. For volumes: demand shifts, market changes]

## Strategic Recommendations
1. **[Recommendation category]:** [Detailed recommendation appropriate for the metric type]
2. **[Recommendation category]:** [Detailed recommendation appropriate for the metric type]

Facts:
{{facts}}"""
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

        # Prepare facts for prompts
        forecast_stats_dict = calculate_forecast_stats(best_results)

        # Determine if this is a price metric (uses AVG aggregation)
        price_metrics = ['unitprice', 'price', 'rate', 'cost', 'margin', 'discount']
        is_price_metric = any(price_term in metric.lower() for price_term in price_metrics)

        # Format facts differently for price vs volume metrics
        if is_price_metric:
            facts = {
                'metric': metric,
                'forecast_steps': forecast_steps,
                'model_name': best_model.replace('_', ' ').title(),
                'is_price_metric': True,
                'forecasted_range': f"${forecast_stats_dict['min']:,.2f} to ${forecast_stats_dict['max']:,.2f}",
                'average_forecasted_price': f"${forecast_stats_dict['average']:,.2f}",
                'starting_price': f"${best_results['forecast'][0]:,.2f}",
                'ending_price': f"${best_results['forecast'][-1]:,.2f}",
                'price_change': f"{forecast_stats_dict['growth']:.1f}%",
                'trend_direction': patterns['trend_direction'],
                'trend_r2': f"{patterns['trend_r2']:.3f}",
                'volatility': patterns['volatility_level'],
                'has_seasonality': 'Yes' if patterns['has_seasonality'] else 'No',
                'data_points': patterns['data_points'],
                'model_accuracy_mape': f"{model_results[best_model]['mape']:.1f}%",
                'model_accuracy_mae': f"${model_results[best_model]['mae']:,.2f}",
                'model_comparison': [
                    {
                        'name': name.replace('_', ' ').title(),
                        'mae': f"${results['mae']:,.2f}",
                        'mape': f"{results['mape']:.1f}%",
                        'selected': name == best_model
                    }
                    for name, results in model_results.items()
                ]
            }
        else:
            facts = {
                'metric': metric,
                'forecast_steps': forecast_steps,
                'model_name': best_model.replace('_', ' ').title(),
                'is_price_metric': False,
                'total_forecasted': f"${forecast_stats_dict['total']:,.0f}",
                'average_per_period': f"${forecast_stats_dict['average']:,.0f}",
                'growth_pct': f"{forecast_stats_dict['growth']:.1f}%",
                'trend_direction': patterns['trend_direction'],
                'trend_r2': f"{patterns['trend_r2']:.3f}",
                'volatility': patterns['volatility_level'],
                'has_seasonality': 'Yes' if patterns['has_seasonality'] else 'No',
                'data_points': patterns['data_points'],
                'model_accuracy_mape': f"{model_results[best_model]['mape']:.1f}%",
                'model_accuracy_mae': f"{model_results[best_model]['mae']:,.0f}",
                'model_comparison': [
                    {
                        'name': name.replace('_', ' ').title(),
                        'mae': f"{results['mae']:,.0f}",
                        'mape': f"{results['mape']:.1f}%",
                        'selected': name == best_model
                    }
                    for name, results in model_results.items()
                ]
            }

        # Generate brief Max response
        max_template = jinja2.Template(parameters.arguments.max_prompt)
        max_response = max_template.render(facts=json.dumps(facts, indent=2))

        # Generate detailed insights using LLM
        insight_template = jinja2.Template(parameters.arguments.insight_prompt)
        insight_prompt = insight_template.render(
            facts=json.dumps(facts, indent=2),
            model_name=facts['model_name']
        )

        try:
            ar_utils = ArUtils()
            detailed_insights = ar_utils.get_llm_response(insight_prompt)
        except Exception as e:
            print(f"DEBUG: Failed to generate insights: {e}")
            detailed_insights = "Forecast analysis completed successfully."

        # Create parameter display pills
        param_info = [
            ParameterDisplayDescription(key="metric", value=f"Metric: {metric}"),
            ParameterDisplayDescription(key="forecast_steps", value=f"Forecast Steps: {forecast_steps} months"),
            ParameterDisplayDescription(key="start_date", value=f"Start Date: {start_date}"),
            ParameterDisplayDescription(key="model", value=f"Model: {best_model.replace('_', ' ').title()}"),
            ParameterDisplayDescription(key="accuracy", value=f"Accuracy: {model_results[best_model]['mape']:.1f}% MAPE")
        ]

        # Add filter pills if filters exist
        if other_filters:
            for filter_item in other_filters:
                if isinstance(filter_item, dict) and 'dim' in filter_item and 'val' in filter_item:
                    dimension = filter_item['dim'].title()
                    values = filter_item['val']
                    if isinstance(values, list):
                        values_str = ', '.join(str(v).title() for v in values)
                    else:
                        values_str = str(values).title()
                    param_info.append(
                        ParameterDisplayDescription(key=f"filter_{dimension}", value=f"{dimension}: {values_str}")
                    )

        # Create visualizations (without insights - they go in narrative)
        visualizations = create_visualizations(
            output_df=output_df,
            metric=metric,
            best_model=best_model,
            patterns=patterns,
            model_results=model_results,
            forecast_stats=forecast_stats_dict,
            other_filters=other_filters,
            confidence_level=confidence_level
        )

        return SkillOutput(
            final_prompt=max_response,
            narrative=detailed_insights,
            visualizations=visualizations,
            parameter_display_descriptions=param_info
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

    # Determine aggregation method based on metric type
    # Price/rate metrics use AVG, all others use SUM
    price_metrics = ['unitprice', 'price', 'rate', 'cost', 'margin', 'discount']
    agg_method = 'AVG' if any(price_term in metric.lower() for price_term in price_metrics) else 'SUM'

    print(f"DEBUG: Using {agg_method} aggregation for metric: {metric}")

    # Build SQL query to get time series data for forecasting
    sql_query = f"""
    SELECT
        DATE_TRUNC('month', InvoiceDate) as invoice_month,
        {agg_method}({metric}) as {metric}
    FROM read_parquet('Top20_SKUs_2021_Present_V2.parquet')
    WHERE 1=1
    """

    # Add date filter if provided
    if start_date:
        sql_query += f" AND InvoiceDate >= '{start_date}'"
        print(f"DEBUG: Added date filter: {start_date}")

    # Hardcode end date to exclude October 2025 onwards (forecast always starts October 2025)
    sql_query += f" AND InvoiceDate < '2025-10-01'"
    print(f"DEBUG: Added end date filter: < 2025-10-01")

    # Add other filters
    if other_filters:
        for filter_item in other_filters:
            if isinstance(filter_item, dict):
                # Handle AnswerRocket filter format: {'dim': 'brand', 'op': '=', 'val': ['barilla']}
                if 'dim' in filter_item and 'val' in filter_item:
                    dimension = filter_item['dim']
                    values = filter_item['val']
                    operator = filter_item.get('op', '=')

                    if isinstance(values, list):
                        if operator == '=' or operator == 'IN':
                            # Make filter case-insensitive for better matching
                            values_str = "', '".join(str(v).upper() for v in values)
                            sql_query += f" AND UPPER({dimension}) IN ('{values_str}')"
                            print(f"DEBUG: Added filter: UPPER({dimension}) IN ({[v.upper() for v in values]})")
                        else:
                            # Handle other operators if needed
                            sql_query += f" AND UPPER({dimension}) {operator} '{str(values[0]).upper()}'"
                            print(f"DEBUG: Added filter: UPPER({dimension}) {operator} {values[0].upper()}")
                    else:
                        sql_query += f" AND UPPER({dimension}) {operator} '{str(values).upper()}'"
                        print(f"DEBUG: Added filter: UPPER({dimension}) {operator} {values.upper()}")
                else:
                    # Fallback for other dict formats
                    for key, value in filter_item.items():
                        if isinstance(value, list):
                            values_str = "', '".join(str(v) for v in value)
                            sql_query += f" AND {key} IN ('{values_str}')"
                        else:
                            sql_query += f" AND {key} = '{value}'"
                        print(f"DEBUG: Added filter: {key} = {value}")

    sql_query += f"""
    GROUP BY invoice_month
    ORDER BY invoice_month
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
            if 'invoice_month' in raw_df.columns:
                raw_df = raw_df.rename(columns={'invoice_month': 'period'})
                print(f"DEBUG: Renamed invoice_month to period")

            if metric in raw_df.columns:
                raw_df = raw_df.rename(columns={metric: 'value'})
                print(f"DEBUG: Renamed {metric} to value")

            # Ensure value column is float for numpy operations
            if 'value' in raw_df.columns:
                raw_df['value'] = raw_df['value'].astype(float)
                print(f"DEBUG: Converted value column to float")

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

    print(f"DEBUG: Training models with {len(train)} training points, {len(validate)} validation points")

    # 1. Prophet Model
    try:
        print(f"DEBUG: Attempting Prophet model...")
        # Prepare data for Prophet
        prophet_df = train[['period', 'value']].rename(columns={'period': 'ds', 'value': 'y'})

        # Create and fit model
        model = Prophet(
            yearly_seasonality=True if len(train) >= 24 else False,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=confidence_level,
            changepoint_prior_scale=0.05  # More conservative
        )
        print(f"DEBUG: Prophet data shape: {prophet_df.shape}, date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
        model.fit(prophet_df)
        print(f"DEBUG: Prophet model fitted successfully")

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
        print(f"DEBUG: Prophet model FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"DEBUG: Prophet traceback: {traceback.format_exc()}")

    # 2. Holt-Winters Exponential Smoothing
    try:
        print(f"DEBUG: Attempting Holt-Winters model...")
        # Determine seasonality
        seasonal_periods = 12 if len(train) >= 24 else None
        print(f"DEBUG: Holt-Winters seasonal_periods: {seasonal_periods}")

        if seasonal_periods:
            model = ExponentialSmoothing(
                train['value'],
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add',
                initialization_method='estimated'
            )
        else:
            model = ExponentialSmoothing(
                train['value'],
                trend='add',
                initialization_method='estimated'
            )

        print(f"DEBUG: Holt-Winters model created, fitting...")
        fitted_model = model.fit()
        print(f"DEBUG: Holt-Winters fitted successfully")

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
        print(f"DEBUG: Holt-Winters model FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"DEBUG: Holt-Winters traceback: {traceback.format_exc()}")

    # 3. Simple Moving Average (fallback)
    try:
        print(f"DEBUG: Attempting Moving Average model...")
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
        print(f"DEBUG: Moving Average model fitted successfully")
    except Exception as e:
        print(f"DEBUG: Moving Average model FAILED: {type(e).__name__}: {str(e)}")

    print(f"DEBUG: Models completed. Available models: {list(models.keys())}")
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

def create_visualizations(output_df, metric, best_model, patterns, model_results, forecast_stats, other_filters=None, confidence_level=0.95):
    """
    Create visualizations for forecast results (chart only - insights go in narrative)
    """
    visualizations = []

    # Convert confidence level to percentage for display
    confidence_pct = int(confidence_level * 100)

    # Prepare data for chart
    historical = output_df[output_df['type'] == 'historical'].copy()
    forecast = output_df[output_df['type'] == 'forecast'].copy()

    # Format dates for chart
    historical['period_str'] = pd.to_datetime(historical['period']).dt.strftime('%b %Y')
    forecast['period_str'] = pd.to_datetime(forecast['period']).dt.strftime('%b %Y')

    # Determine decimal places based on data magnitude (calculate early for use in series)
    all_values = list(historical['actual']) + list(forecast['forecast'])
    max_value = max(all_values)

    # Use decimals for small numbers (< 100), no decimals for large numbers
    if max_value < 10:
        y_format = "${value:,.2f}"  # 2 decimals for very small numbers (e.g., $1.23)
        tooltip_format = "${point.y:,.2f}"
        ci_format = "${point.low:,.2f} - ${point.high:,.2f}"
    elif max_value < 100:
        y_format = "${value:,.2f}"  # 2 decimals for small numbers (e.g., $12.34)
        tooltip_format = "${point.y:,.2f}"
        ci_format = "${point.low:,.2f} - ${point.high:,.2f}"
    elif max_value < 1000:
        y_format = "${value:,.1f}"  # 1 decimal for medium numbers (e.g., $123.4)
        tooltip_format = "${point.y:,.1f}"
        ci_format = "${point.low:,.1f} - ${point.high:,.1f}"
    else:
        y_format = "${value:,.0f}"  # No decimals for large numbers (e.g., $1,234)
        tooltip_format = "${point.y:,.0f}"
        ci_format = "${point.low:,.0f} - ${point.high:,.0f}"

    # Build filter display for chart subtitle
    filter_text = ""
    if other_filters:
        filter_parts = []
        for filter_item in other_filters:
            if isinstance(filter_item, dict) and 'dim' in filter_item and 'val' in filter_item:
                dimension = filter_item['dim'].title()
                values = filter_item['val']
                if isinstance(values, list):
                    values_str = ', '.join(str(v).title() for v in values)
                else:
                    values_str = str(values).title()
                filter_parts.append(f"{dimension}: {values_str}")
        if filter_parts:
            filter_text = " | " + " | ".join(filter_parts)

    # Create series data for Highcharts
    historical_series = {
        "name": "Actual",
        "data": [float(val) for val in historical['actual']],
        "color": "#3498DB",
        "lineWidth": 3,
        "marker": {"enabled": True, "radius": 5, "symbol": "circle"}
    }

    forecast_series = {
        "name": "Forecast",
        "data": [None] * len(historical) + [float(val) for val in forecast['forecast']],
        "color": "#E74C3C",
        "dashStyle": "ShortDash",
        "lineWidth": 3,
        "marker": {"enabled": True, "radius": 5, "symbol": "diamond"}
    }

    # Confidence interval as area range (shows uncertainty range)
    confidence_series = {
        "name": f"{confidence_pct}% Confidence Range",
        "data": [None] * len(historical) + [[float(lower), float(upper)]
                 for lower, upper in zip(forecast['lower_bound'], forecast['upper_bound'])],
        "type": "arearange",
        "lineWidth": 0,
        "color": "rgba(231, 76, 60, 0.2)",
        "fillOpacity": 0.3,
        "marker": {"enabled": False},
        "enableMouseTracking": True,
        "zIndex": 0,
        "tooltip": {
            "pointFormat": f"<span style=\"color:{{series.color}}\">\u25CF</span> {{series.name}}: <b>{ci_format}</b><br/><span style=\"font-size:11px;color:#7F8C8D\">{confidence_pct}% probability actual values fall within this range</span><br/>"
        }
    }

    # All categories for x-axis
    all_categories = list(historical['period_str']) + list(forecast['period_str'])

    # Create Highcharts configuration
    chart_config = {
        "type": "highcharts",
        "config": {
            "chart": {
                "type": "line",
                "height": 500,
                "backgroundColor": "#FAFAFA",
                "style": {"fontFamily": "Arial, sans-serif"}
            },
            "title": {
                "text": f"{metric.title()} Forecast",
                "style": {"fontSize": "20px", "fontWeight": "bold", "color": "#2C3E50"}
            },
            "subtitle": {
                "text": f"Model: {best_model.replace('_', ' ').title()} | Accuracy: {model_results[best_model]['mape']:.1f}% MAPE{filter_text}",
                "style": {"fontSize": "14px", "color": "#7F8C8D"}
            },
            "xAxis": {
                "categories": all_categories,
                "title": {"text": "Time Period", "style": {"fontWeight": "bold"}},
                "gridLineWidth": 1,
                "gridLineColor": "#E0E0E0",
                "labels": {"rotation": -45, "style": {"fontSize": "11px"}}
            },
            "yAxis": {
                "title": {"text": f"{metric.title()} ($)", "style": {"fontWeight": "bold"}},
                "labels": {"format": y_format},
                "gridLineColor": "#E0E0E0"
            },
            "tooltip": {
                "shared": True,
                "crosshairs": True,
                "backgroundColor": "#FFFFFF",
                "borderColor": "#CCCCCC",
                "borderRadius": 8,
                "shadow": True,
                "useHTML": True,
                "headerFormat": "<b>{point.key}</b><br/>",
                "pointFormat": f"<span style=\"color:{{series.color}}\">\u25CF</span> {{series.name}}: <b>{tooltip_format}</b><br/>"
            },
            "legend": {
                "enabled": True,
                "align": "center",
                "verticalAlign": "bottom",
                "borderWidth": 0
            },
            "plotOptions": {
                "line": {
                    "marker": {"enabled": True}
                },
                "series": {
                    "animation": True
                }
            },
            "series": [confidence_series, historical_series, forecast_series],
            "credits": {"enabled": False}
        }
    }

    # Create layout with chart only (insights go in narrative section)
    layout = {
        "layoutJson": {
            "type": "Document",
            "style": {
                "backgroundColor": "#ffffff",
                "padding": "20px",
                "fontFamily": "Arial, sans-serif"
            },
            "children": [
                {
                    "type": "HighchartsChart",
                    "options": chart_config["config"]
                }
            ]
        },
        "inputVariables": []
    }

    rendered = wire_layout(layout, {})
    visualizations.append(SkillVisualization(title="Forecast Chart", layout=rendered))

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