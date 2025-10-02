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

from skill_builder import skill, SkillOutput, ExportData, InputParam, SkillInput


@skill(
    name="forecast_analysis",
    llm_name="forecast_analysis",
    description="Generate intelligent forecasts using best-fit model selection with automatic model optimization",
    capabilities="Provides multi-model forecasting with automatic selection of best-performing algorithm. Supports linear regression, moving average, and other forecasting models. Generates confidence intervals, trend analysis, and seasonality detection. Creates professional visualizations with KPIs, charts, and insights.",
    limitations="Requires minimum 12 historical data points. Limited to 36 months forecast horizon. Assumes monthly granularity (max_time_month). Performance depends on data quality and historical patterns.",
    example_questions="What will sales be over the next 6 months? Can you forecast volume for Q1 2024? Show me a 12-month revenue projection with confidence intervals. What's the expected growth trend for the next quarter?",
    parameter_guidance="Select metric to forecast (sales, volume, etc.). Choose forecast steps (1-36 months, default 6). Optionally filter by date range or apply dimensional filters. The skill automatically selects the best forecasting model based on historical performance.",
    parameters=[
        InputParam(
            name="metric",
            description="The metric to forecast",
            type="string",
            required=True
        ),
        InputParam(
            name="forecast_steps",
            description="Number of periods to forecast (months)",
            type="integer",
            required=False,
            default=6
        ),
        InputParam(
            name="start_date",
            description="Start date for training data (YYYY-MM-DD)",
            type="string",
            required=False,
            default=None
        ),
        InputParam(
            name="other_filters",
            description="Additional filters to apply to the data",
            type="list",
            required=False,
            default=[]
        ),
        InputParam(
            name="confidence_level",
            description="Confidence level for prediction intervals",
            type="float",
            required=False,
            default=0.95
        )
    ]
)
def forecast_analysis(context, metric, forecast_steps=6, start_date=None, other_filters=[], confidence_level=0.95):
    """
    Generate forecasts using multiple models and select the best performer
    """
    try:
        # Validate inputs
        if forecast_steps < 1 or forecast_steps > 36:
            return SkillOutput(
                final_prompt="Invalid forecast steps. Please specify between 1 and 36 months.",
                warnings=["Forecast steps must be between 1 and 36 months"]
            )

        # Get data from context
        data_df = fetch_data(context, metric, start_date, other_filters)

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

        # Create robust visualization config
        viz_config = create_forecast_layout(
            metric=metric,
            forecast_steps=forecast_steps,
            start_date=start_date,
            other_filters=other_filters,
            model_name=best_model,
            patterns=patterns,
            forecast_stats=calculate_forecast_stats(best_results)
        )

        # Generate prompt for LLM
        prompt = generate_prompt(
            metric=metric,
            forecast_steps=forecast_steps,
            best_model=best_model,
            patterns=patterns,
            model_results=model_results,
            forecast_stats=calculate_forecast_stats(best_results)
        )

        return SkillOutput(
            data=[ExportData(data=output_df, name="forecast_results")],
            final_prompt=prompt,
            visualization_config=viz_config
        )

    except Exception as e:
        # Don't throw exceptions - return user-friendly error
        return SkillOutput(
            final_prompt=f"An error occurred while generating the forecast. Please check your data and try again.",
            warnings=[f"Error: {str(e)}"]
        )

def fetch_data(context, metric, start_date, other_filters):
    """
    Fetch data from the context
    """
    # Use AnswerRocket context to get data
    query = context.query_builder()
    query.select_metric(metric)

    # Apply other filters (standard filter bucket)
    for filter_item in other_filters:
        query.apply_filter(filter_item)

    if start_date:
        query.filter_by_date(start_date)

    raw_df = query.execute()

    # Standardize column names - AR returns metric name and max_time_month
    if raw_df is not None and not raw_df.empty:
        # Rename columns to standard format
        if 'max_time_month' in raw_df.columns:
            raw_df = raw_df.rename(columns={'max_time_month': 'period'})
        if metric in raw_df.columns:
            raw_df = raw_df.rename(columns={metric: 'value'})
        elif 'sales' in raw_df.columns:  # fallback
            raw_df = raw_df.rename(columns={'sales': 'value'})
        elif 'volume' in raw_df.columns:  # fallback for volume
            raw_df = raw_df.rename(columns={'volume': 'value'})

    return raw_df

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
    Run multiple forecasting models
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Prepare data
    df = df.copy()
    df['period_num'] = range(len(df))

    # Split for validation
    train_size = int(len(df) * 0.8)
    train = df[:train_size]
    validate = df[train_size:]

    models = {}

    # Linear Model
    try:
        X_train = train[['period_num']]
        y_train = train['value']
        X_val = validate[['period_num']]
        y_val = validate['value']

        model = LinearRegression()
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)

        # Calculate metrics
        mae = mean_absolute_error(y_val, val_pred)
        mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100

        # Generate full forecast
        future_periods = np.arange(len(df), len(df) + periods).reshape(-1, 1)
        forecast = model.predict(future_periods)

        # Simple confidence intervals
        residuals = y_train - model.predict(X_train)
        std_error = np.std(residuals)
        z_score = 1.96 if confidence_level == 0.95 else 2.58

        models['linear'] = {
            'forecast': forecast,
            'lower_bound': forecast - z_score * std_error,
            'upper_bound': forecast + z_score * std_error,
            'mae': mae,
            'mape': mape,
            'score': mae  # Use MAE as primary score
        }
    except:
        pass

    # Moving Average Model
    try:
        window = min(3, len(train) // 4)
        ma_forecast = []
        recent_values = list(train['value'].values[-window:])

        for _ in range(len(validate)):
            pred = np.mean(recent_values)
            ma_forecast.append(pred)
            recent_values.pop(0)
            recent_values.append(pred)

        mae = mean_absolute_error(validate['value'], ma_forecast)
        mape = np.mean(np.abs((validate['value'] - ma_forecast) / validate['value'])) * 100

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
            'lower_bound': np.array(full_forecast) * 0.9,  # Simple bounds
            'upper_bound': np.array(full_forecast) * 1.1,
            'mae': mae,
            'mape': mape,
            'score': mae
        }
    except:
        pass

    # Add more models as needed (exponential smoothing, ARIMA, etc.)

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

def create_forecast_layout(metric, forecast_steps, start_date, other_filters, model_name, patterns, forecast_stats):
    """
    Create robust visualization layout with trend line, narrative, and parameter pills
    """
    return {
        "layoutJson": {
            "type": "Document",
            "gap": "0px",
            "style": {
                "backgroundColor": "#ffffff",
                "width": "100%",
                "height": "max-content",
                "padding": "15px",
                "gap": "15px"
            },
            "children": [
                {
                    "name": "FlexContainer_Header",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "80px",
                    "direction": "column",
                    "style": {
                        "backgroundColor": "#059669",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "marginBottom": "20px"
                    },
                    "label": "FlexContainer-Header"
                },
                {
                    "name": "Header_Title",
                    "type": "Header",
                    "children": "",
                    "text": f"Forecast Analysis: {metric.title()}",
                    "style": {
                        "fontSize": "24px",
                        "fontWeight": "bold",
                        "color": "#ffffff",
                        "textAlign": "left",
                        "margin": "0"
                    },
                    "parentId": "FlexContainer_Header",
                    "label": "Header-Main_Title"
                },
                {
                    "name": "Header_Subtitle",
                    "type": "Header",
                    "children": "",
                    "text": f"{forecast_steps} Month Forecast using {model_name.title()} Model",
                    "style": {
                        "fontSize": "16px",
                        "fontWeight": "normal",
                        "color": "#d1fae5",
                        "textAlign": "left",
                        "marginTop": "5px"
                    },
                    "parentId": "FlexContainer_Header",
                    "label": "Header-Subtitle"
                },
                {
                    "name": "FlexContainer_KPI",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "150px",
                    "direction": "row",
                    "label": "FlexContainer-KPI_panel",
                    "extraStyles": "gap: 15px; margin-bottom: 20px;"
                },
                {
                    "name": "FlexContainer_KPI1",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "120px",
                    "direction": "column",
                    "parentId": "FlexContainer_KPI",
                    "label": "FlexContainer-KPI_Card1",
                    "style": {
                        "borderRadius": "8px",
                        "padding": "15px",
                        "backgroundColor": "#ecfdf5",
                        "border-left": "4px solid #059669"
                    }
                },
                {
                    "name": "KPI1_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Total Forecast",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "normal",
                        "color": "#374151",
                        "marginBottom": "5px"
                    },
                    "parentId": "FlexContainer_KPI1"
                },
                {
                    "name": "KPI1_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{forecast_stats['total']:,.0f}",
                    "style": {
                        "fontSize": "28px",
                        "fontWeight": "600",
                        "color": "#000000"
                    },
                    "parentId": "FlexContainer_KPI1"
                },
                {
                    "name": "FlexContainer_KPI2",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "120px",
                    "direction": "column",
                    "parentId": "FlexContainer_KPI",
                    "label": "FlexContainer-KPI_Card2",
                    "style": {
                        "borderRadius": "8px",
                        "padding": "15px",
                        "backgroundColor": "#fef3c7"
                    }
                },
                {
                    "name": "KPI2_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Avg Per Month",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "normal",
                        "color": "#374151",
                        "marginBottom": "5px"
                    },
                    "parentId": "FlexContainer_KPI2"
                },
                {
                    "name": "KPI2_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{forecast_stats['average']:,.0f}",
                    "style": {
                        "fontSize": "28px",
                        "fontWeight": "600",
                        "color": "#000000"
                    },
                    "parentId": "FlexContainer_KPI2"
                },
                {
                    "name": "FlexContainer_KPI3",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "120px",
                    "direction": "column",
                    "parentId": "FlexContainer_KPI",
                    "label": "FlexContainer-KPI_Card3",
                    "style": {
                        "borderRadius": "8px",
                        "padding": "15px",
                        "backgroundColor": "#dbeafe"
                    }
                },
                {
                    "name": "KPI3_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Growth Rate",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "normal",
                        "color": "#374151",
                        "marginBottom": "5px"
                    },
                    "parentId": "FlexContainer_KPI3"
                },
                {
                    "name": "KPI3_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"{forecast_stats['growth']:.1f}%",
                    "style": {
                        "fontSize": "28px",
                        "fontWeight": "600",
                        "color": "#000000"
                    },
                    "parentId": "FlexContainer_KPI3"
                },
                {
                    "name": "FlexContainer_KPI4",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "120px",
                    "direction": "column",
                    "parentId": "FlexContainer_KPI",
                    "label": "FlexContainer-KPI_Card4",
                    "style": {
                        "borderRadius": "8px",
                        "padding": "15px",
                        "backgroundColor": "#fecaca"
                    }
                },
                {
                    "name": "KPI4_Label",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Trend Direction",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "normal",
                        "color": "#374151",
                        "marginBottom": "5px"
                    },
                    "parentId": "FlexContainer_KPI4"
                },
                {
                    "name": "KPI4_Value",
                    "type": "Paragraph",
                    "children": "",
                    "text": patterns['trend_direction'].title(),
                    "style": {
                        "fontSize": "28px",
                        "fontWeight": "600",
                        "color": "#000000"
                    },
                    "parentId": "FlexContainer_KPI4"
                },
                {
                    "name": "HighchartsChart_Forecast",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "450px",
                    "label": "HighchartsChart-Forecast",
                    "extraStyles": "border-radius: 8px; margin-bottom: 20px;",
                    "options": {
                        "chart": {
                            "type": "line",
                            "backgroundColor": "#ffffff"
                        },
                        "title": {
                            "text": f"{metric.title()} Forecast with Trend Line",
                            "style": {
                                "fontSize": "20px",
                                "fontWeight": "600"
                            }
                        },
                        "xAxis": {
                            "type": "datetime",
                            "title": {
                                "text": "Period"
                            }
                        },
                        "yAxis": {
                            "title": {
                                "text": f"{metric.title()}"
                            }
                        },
                        "plotOptions": {
                            "line": {
                                "marker": {
                                    "enabled": True,
                                    "radius": 4
                                }
                            },
                            "series": {
                                "fillOpacity": 0.1
                            }
                        },
                        "series": [],  # Will be populated with actual data
                        "legend": {
                            "enabled": True,
                            "align": "center"
                        },
                        "credits": {
                            "enabled": False
                        }
                    }
                },
                {
                    "name": "FlexContainer_Narrative",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "200px",
                    "direction": "column",
                    "style": {
                        "borderRadius": "8px",
                        "padding": "20px",
                        "backgroundColor": "#f8fafc",
                        "border-left": "4px solid #059669",
                        "marginBottom": "20px"
                    },
                    "label": "FlexContainer-Narrative"
                },
                {
                    "name": "Narrative_Title",
                    "type": "Header",
                    "children": "",
                    "text": "Forecast Insights",
                    "style": {
                        "fontSize": "18px",
                        "fontWeight": "600",
                        "color": "#1f2937",
                        "marginBottom": "10px"
                    },
                    "parentId": "FlexContainer_Narrative"
                },
                {
                    "name": "Narrative_Content",
                    "type": "Markdown",
                    "children": "",
                    "text": "Analysis insights will appear here...",
                    "style": {
                        "fontSize": "16px",
                        "color": "#374151",
                        "lineHeight": "1.6"
                    },
                    "parentId": "FlexContainer_Narrative",
                    "label": "Markdown-insights"
                },
                {
                    "name": "FlexContainer_Pills",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "60px",
                    "direction": "row",
                    "style": {
                        "gap": "10px",
                        "flexWrap": "wrap",
                        "alignItems": "center"
                    },
                    "label": "FlexContainer-Parameter_Pills"
                },
                {
                    "name": "Pill_Metric",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"Metric: {metric}",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "500",
                        "color": "#1f2937",
                        "backgroundColor": "#e5e7eb",
                        "padding": "6px 12px",
                        "borderRadius": "20px",
                        "margin": "0"
                    },
                    "parentId": "FlexContainer_Pills"
                },
                {
                    "name": "Pill_Steps",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"Forecast Steps: {forecast_steps} months",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "500",
                        "color": "#1f2937",
                        "backgroundColor": "#e5e7eb",
                        "padding": "6px 12px",
                        "borderRadius": "20px",
                        "margin": "0"
                    },
                    "parentId": "FlexContainer_Pills"
                },
                {
                    "name": "Pill_Model",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"Model: {model_name}",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "500",
                        "color": "#1f2937",
                        "backgroundColor": "#e5e7eb",
                        "padding": "6px 12px",
                        "borderRadius": "20px",
                        "margin": "0"
                    },
                    "parentId": "FlexContainer_Pills"
                }
            ] + ([
                {
                    "name": "Pill_StartDate",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"Start Date: {start_date}",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "500",
                        "color": "#1f2937",
                        "backgroundColor": "#e5e7eb",
                        "padding": "6px 12px",
                        "borderRadius": "20px",
                        "margin": "0"
                    },
                    "parentId": "FlexContainer_Pills"
                }
            ] if start_date else []) + ([
                {
                    "name": "Pill_Filters",
                    "type": "Paragraph",
                    "children": "",
                    "text": f"Filters: {len(other_filters)} applied",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "500",
                        "color": "#1f2937",
                        "backgroundColor": "#e5e7eb",
                        "padding": "6px 12px",
                        "borderRadius": "20px",
                        "margin": "0"
                    },
                    "parentId": "FlexContainer_Pills"
                }
            ] if other_filters else [])
        },
        "inputVariables": [
            {
                "name": "chart_series",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "HighchartsChart_Forecast",
                        "fieldName": "options.series"
                    }
                ]
            },
            {
                "name": "narrative_content",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Narrative_Content",
                        "fieldName": "text"
                    }
                ]
            }
        ]
    }

if __name__ == '__main__':
    skill_input: SkillInput = forecast_analysis.create_input(arguments={
        'metric': "volume",
        'forecast_steps': 6,
        'start_date': "2022-01-01",
        'other_filters': [{"dim": "manufacturer", "op": "=", "val": ["barilla"]}]
    })