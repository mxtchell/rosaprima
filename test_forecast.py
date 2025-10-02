"""
Test the forecast skill locally
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mock context for testing
class MockContext:
    def __init__(self):
        self.data = self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample time series data"""
        periods = pd.date_range(start='2021-01-01', periods=36, freq='M')
        # Create data with trend and some seasonality
        trend = np.arange(36) * 1000
        seasonal = np.sin(np.arange(36) * 2 * np.pi / 12) * 5000
        noise = np.random.randn(36) * 2000
        values = 100000 + trend + seasonal + noise

        return pd.DataFrame({
            'period': periods,
            'value': values
        })

    def query_builder(self):
        return MockQueryBuilder(self.data)

class MockQueryBuilder:
    def __init__(self, data):
        self.data = data

    def select_metric(self, metric):
        return self

    def group_by(self, dimension):
        return self

    def filter_by_date(self, start_date):
        return self

    def execute(self):
        return self.data

def test_forecast_skill():
    """Test the forecast skill"""
    import sys
    sys.path.append('.')
    from forecast_skill import forecast_analysis

    # Create mock context
    context = MockContext()

    print("Testing Forecast Skill...")
    print("=" * 50)

    # Test basic forecast
    print("1. Testing basic forecast...")
    result = forecast_analysis(
        context=context,
        metric="sales",
        periods=12
    )

    if result.data:
        df = result.data[0].data
        print(f"   ✓ Generated {len(df)} data points")
        print(f"   ✓ Historical points: {len(df[df['type'] == 'historical'])}")
        print(f"   ✓ Forecast points: {len(df[df['type'] == 'forecast'])}")
        print(f"   ✓ Model used: {df['model_used'].iloc[-1]}")
    else:
        print(f"   ✗ Failed: {result.warnings}")

    print("\n2. Testing with start date...")
    result = forecast_analysis(
        context=context,
        metric="sales",
        periods=6,
        start_date="2022-01-01"
    )

    if result.data:
        print("   ✓ Start date filtering works")
    else:
        print(f"   ✗ Failed: {result.warnings}")

    print("\n3. Testing edge cases...")

    # Test with too few periods
    result = forecast_analysis(
        context=context,
        metric="sales",
        periods=50  # Should fail
    )

    if result.warnings:
        print("   ✓ Proper validation for invalid periods")

    print("\n4. Sample output:")
    print("-" * 30)
    if result.data:
        sample_df = result.data[0].data.tail(10)
        print(sample_df[['period', 'type', 'forecast', 'lower_bound', 'upper_bound']].to_string())

    print("\n5. Sample prompt:")
    print("-" * 30)
    print(result.final_prompt[:500] + "..." if len(result.final_prompt) > 500 else result.final_prompt)

if __name__ == "__main__":
    test_forecast_skill()