from typing import List

import cvxpy as cvx
import numpy as np
import pandas as pd

from ..._portfolio.portfolio import PortfolioAsset
from ..._settings import _MONTHS_PER_YEAR
from ...common.enums import Currency


class EfficiencyFrontierPoint:
    def __init__(self, weights: np.array, ret: float, ret1: float, risk1: float, risk: float):
        self.weights = weights
        self.ret1 = ret1
        self.ret = ret
        self.risk = risk
        self.risk1 = risk1


class EfficiencyFrontier:
    def __init__(self,
                 start_period: pd.Period, end_period: pd.Period,
                 currency: Currency):
        self.points: List[EfficiencyFrontierPoint] = []

    def add_point(self, point: EfficiencyFrontierPoint):
        self.points.append(point)


def compute(assets: List[PortfolioAsset], samples_count,
            start_period: pd.Period, end_period: pd.Period,
            currency: Currency) -> EfficiencyFrontier:
    asset_rors = [a.rate_of_return().values for a in assets]
    mu = np.mean(asset_rors, axis=1)
    sigma = np.cov(asset_rors)

    efficiency_frontier = EfficiencyFrontier(start_period=start_period, end_period=end_period, currency=currency)

    for idx, return_trg in enumerate(np.linspace(mu.min(), mu.max(), num=samples_count)):
        w = cvx.Variable(len(assets))
        ret = mu.T * w
        risk = cvx.quad_form(w, sigma)
        problem = cvx.Problem(objective=cvx.Minimize(risk),
                              constraints=[cvx.sum(w) == 1,
                                           w >= 0,
                                           cvx.abs(ret - return_trg) <= 1e-8,
                                           ])
        problem.solve(solver=cvx.GUROBI)

        ret_yearly = (ret.value + 1.) ** _MONTHS_PER_YEAR - 1.
        risk_yearly = (risk.value ** 2 + (ret.value + 1.) ** 2) ** _MONTHS_PER_YEAR - (ret.value + 1.) ** (
                    _MONTHS_PER_YEAR * 2)
        risk_yearly = np.sqrt(risk_yearly)
        point = EfficiencyFrontierPoint(weights=w.value,
                                        ret1=ret.value,
                                        ret=ret_yearly,
                                        risk=risk_yearly,
                                        risk1=risk.value)
        efficiency_frontier.add_point(point)

    return efficiency_frontier
