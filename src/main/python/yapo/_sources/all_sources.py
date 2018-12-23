from abc import ABCMeta, abstractmethod

from serum import singleton, inject

from .micex_stocks_source import MicexStocksSource
from .mutru_funds_source import MutualFundsRuSource
from .quandl_source import QuandlSource
from .single_financial_symbol_source import CbrCurrenciesSource, CbrTopRatesSource, MicexMcftrSource
from .inflation_sources import InflationUsSource, InflationRuSource, InflationEuSource


@singleton
class SymbolSources(metaclass=ABCMeta):
    @property
    @abstractmethod
    def sources(self):
        raise NotImplementedError()


@inject
class AllSymbolSources(SymbolSources):
    cbr_currencies_source: CbrCurrenciesSource
    cbr_top_rates_source: CbrTopRatesSource
    inflation_ru_source: InflationRuSource
    inflation_eu_source: InflationEuSource
    inflation_us_source: InflationUsSource
    micex_mcftr_source: MicexMcftrSource
    micex_stocks_source: MicexStocksSource
    mut_ru_source: MutualFundsRuSource
    quandl_source: QuandlSource

    @property
    def sources(self):
        return [
            self.cbr_currencies_source,
            self.cbr_top_rates_source,
            self.inflation_ru_source,
            self.inflation_eu_source,
            self.inflation_us_source,
            self.micex_mcftr_source,
            self.micex_stocks_source,
            self.mut_ru_source,
            self.quandl_source,
        ]
