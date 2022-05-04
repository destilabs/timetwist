from random import sample
import unittest
import pandas as pd
import numpy as np

from tools import series

from pandas.testing import assert_frame_equal, assert_series_equal

import pandas_ta as pta

class SeriesTest(unittest.TestCase):
    def test_get_correlation_between_columns(self):
        sample = pd.DataFrame({'column1': [1, 2, 3, 4, 5], 'column2': [2, 3, 4, 5, 6], 'column3': [5, 4, 3, 2, 1]})

        self.assertAlmostEqual(series.get_correlation(sample, 'column1', 'column2'), 1)
        self.assertAlmostEqual(series.get_correlation(sample, 'column1', 'column3'), -1)

    def test_get_lagging_correlation(self):
        sample = pd.DataFrame({'column1': [1, 2, 3, 4, 5, 6], 'column2': [5, 4, 2, 3, 4, 5]})

        self.assertAlmostEqual(series.get_lagging_correlation(sample, 'column1', 'column2', -2), 1)

    def test_impute_missing_values(self):
        sample = pd.DataFrame({'column1': [1, 2, None, 4, 5, 6], 'column2': [5, 4, 3, 2, None, 0]})

        assert_series_equal(series.impute_values_with_linear_regression(sample, 'column1'), 
                        pd.Series([1, 2, 3, 4, 5, 6], dtype=float, name='column1'))

        assert_series_equal(series.impute_values_with_linear_regression(sample, 'column2'),
                        pd.Series([5, 4, 3, 2, 1, 0], dtype=float, name='column2'))

    def test_calculate_annual_return_percentage(self):
        self.assertAlmostEqual(series.calculate_annual_return_percentage(1000, 2000, 5), 0.1487, places=4)

    def test_get_relative_strength_index(self):
        sample = pd.DataFrame({'column1': [12,11,12,14,18,12,15,13,16,12,11,13,15,14,16,18,22,19,24,17,19]})

        assert_frame_equal(
            series.get_relative_strength_index(sample, 'column1', 14)[['column1_rsi']].tail().reset_index(drop=True), 
            pd.DataFrame({'column1_rsi': [67.53, 59.56, 66.63, 52.73, 55.58]}), check_exact=False, check_less_precise=2)

    def test_get_bollinger_bands(self):
        sample = pd.DataFrame({'close': [12,11,12,14,18,12,15,13,16,12,11,13,15,14,16,18,22,19,24,17,19]})

        result = series.get_bollinger_bands(sample, 'close', 5, 2)[['close_upper_band', 'close_lower_band']].tail().reset_index(drop=True)
        expected = pd.DataFrame({
            'close_upper_band': [23.32, 23.86, 26.18, 25.83, 25.74], 
            'close_lower_band': [10.67, 11.73, 13.41, 14.16, 14.65]})

        assert_frame_equal(result, expected, check_exact=False, check_less_precise=2)

    def test_macd(self):
        sample = pd.DataFrame({'close': [12,11,12,14,18,12,15,13,16,12,11,13,15,14,16,18,22,19,24,17,19]})
        result = series.get_macd(sample, 'close', 12, 26, 9)[['close_macd', 'close_signal']].tail().reset_index(drop=True)
        expected = pd.DataFrame({
            'close_macd': [0.834, 0.951, 1.349, 1.184, 1.174],
            'close_signal': [0.293, 0.427, 0.614, 0.729, 0.819]})
        assert_frame_equal(result, expected, check_exact=False, check_less_precise=2)

    def test_get_exponential_moving_average(self):
        sample = pd.DataFrame({'close': [12,11,12,14,18,12,15,13,16,12,11,13,15,14,16,18,22,19,24,17,19]})
        result = series.get_exponential_moving_average_over_period(sample, 'close', 12).tail().reset_index(drop=True)
        expected = pd.Series(np.array([15.623, 16.142, 17.351, 17.297, 17.559]), name = 'close', dtype=float)
        assert_series_equal(result, expected, check_exact=False, check_less_precise=2)






