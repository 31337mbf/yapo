import unittest
import yapo
import numpy as np


class PortfolioAssetStatisticsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.asset_name = 'quandl/BND'
        cls.asset = yapo.portfolio_asset(name=cls.asset_name,
                                         start_period='2011-1', end_period='2017-2', currency='USD')
        cls.epsilon = 1e-3

    def test_accumulated_rate_of_return(self):
        arors = self.asset.accumulated_rate_of_return()
        self.assertTrue(np.all((.0014 < arors) & (arors < .2070)))

        arors_real = self.asset.accumulated_rate_of_return(real=True)
        self.assertTrue(np.all((-.0134 < arors_real) & (arors_real < 0.1082)))

    def test_handle_related_inflation(self):
        self.assertRaises(Exception, self.asset.inflation, kind='abracadabra')

        self.assertAlmostEqual(self.asset.inflation(kind='accumulated'), .1062, delta=self.epsilon)
        self.assertAlmostEqual(self.asset.inflation(kind='a_mean'), 0.0014, delta=self.epsilon)
        self.assertAlmostEqual(self.asset.inflation(kind='g_mean'), 0.0165, delta=self.epsilon)

        self.assertEqual(self.asset.inflation(kind='values').size,
                         self.asset.rate_of_return().size)

    def test_compound_annual_growth_rate(self):
        cagr_default = self.asset.compound_annual_growth_rate()
        self.assertAlmostEqual(cagr_default, .027, delta=self.epsilon)

        cagr_long_time = self.asset.compound_annual_growth_rate(years_ago=20)
        self.assertEqual(cagr_default, cagr_long_time)

        cagr_one_year = self.asset.compound_annual_growth_rate(years_ago=1)
        self.assertAlmostEqual(cagr_one_year, .011, delta=self.epsilon)

        cagr_array = self.asset.compound_annual_growth_rate(years_ago=[None, 20, 1])
        cagr_diff = np.abs(cagr_array - [cagr_default, cagr_long_time, cagr_one_year]) < self.epsilon
        self.assertTrue(np.all(cagr_diff))

    def test_compound_annual_growth_rate_real(self):
        cagr_default = self.asset.compound_annual_growth_rate(real=True)
        self.assertAlmostEqual(cagr_default, .0099, delta=self.epsilon)

        cagr_long_time = self.asset.compound_annual_growth_rate(years_ago=20, real=True)
        self.assertEqual(cagr_default, cagr_long_time)

        cagr_one_year = self.asset.compound_annual_growth_rate(years_ago=1, real=True)
        self.assertAlmostEqual(cagr_one_year, -.0163, delta=self.epsilon)

        cagr_array = self.asset.compound_annual_growth_rate(years_ago=[None, 20, 1], real=True)
        cagr_diff = np.abs(cagr_array - [cagr_default, cagr_long_time, cagr_one_year]) < self.epsilon
        self.assertTrue(np.all(cagr_diff))

    def test_risk(self):
        short_asset = yapo.portfolio_asset(name=self.asset_name,
                                           start_period='2016-8', end_period='2016-12', currency='USD')

        self.assertRaises(Exception, short_asset.risk, period='year')
        self.assertAlmostEqual(self.asset.risk(), .0311, delta=self.epsilon)
        self.assertAlmostEqual(self.asset.risk(period='year'), .0311, delta=self.epsilon)
        self.assertAlmostEqual(self.asset.risk(period='month'), .0087, delta=self.epsilon)

    def test_inflation(self):
        self.assertRaises(Exception, self.asset.inflation, kind='abracadabra')

        self.assertAlmostEqual(self.asset.inflation(kind='accumulated'), .1062, delta=self.epsilon)
        self.assertAlmostEqual(self.asset.inflation(kind='a_mean'), .0014, delta=self.epsilon)
        self.assertAlmostEqual(self.asset.inflation(kind='g_mean'), .0173, delta=self.epsilon)

        self.assertEqual(self.asset.inflation(kind='values').size,
                         self.asset.rate_of_return().size)
