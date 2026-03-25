"""
DAA 전략 구현 - 최종 수정 (scalar 오류 완전 제거)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from utils import (
    calculate_monthly_returns,
    calculate_momentum,
    get_bad_assets,
    count_breadth_bad,
    calculate_cash_fraction,
    select_top_assets,
    calculate_portfolio_weights,
    backtest_returns,
    calculate_performance_metrics
)


@dataclass
class DAAConfig:
    """DAA 전략 설정"""
    risky_tickers: List[str]
    canary_tickers: List[str]
    cash_tickers: List[str]
    momentum_periods: List[int] = None
    momentum_weights: List[int] = None
    breadth_parameter: int = 2
    top_selection: int = 6
    transaction_cost: float = 0.001
    
    def __post_init__(self):
        if self.momentum_periods is None:
            self.momentum_periods = [1, 3, 6, 12]
        if self.momentum_weights is None:
            self.momentum_weights = [12, 4, 2, 1]


class DAAStrategy:
    """DAA 전략"""
    
    def __init__(self, config: DAAConfig):
        self.config = config
        self.price_data = None
        self.monthly_returns = None
        self.momentum = None
        self.bad_assets = None
        self.breadth_bad = None
        self.cash_fraction = None
        self.top_assets = None
        self.weights_risky = None
        self.weights_cash = None
        self.portfolio_returns = None
        
    def fit(self, price_df: pd.DataFrame) -> 'DAAStrategy':
        """전략 학습"""
        try:
            self.price_data = price_df
            self.monthly_returns = calculate_monthly_returns(price_df)
            self.momentum = calculate_momentum(
                self.monthly_returns,
                self.config.momentum_periods,
                self.config.momentum_weights
            )
            self.bad_assets = get_bad_assets(self.momentum, threshold=0.0)
            self.breadth_bad = count_breadth_bad(self.bad_assets, self.config.canary_tickers)
            self.cash_fraction = calculate_cash_fraction(self.breadth_bad, self.config.breadth_parameter)
            self.top_assets = select_top_assets(self.momentum, self.config.top_selection, self.config.risky_tickers)
            self.weights_risky, self.weights_cash = calculate_portfolio_weights(
                self.top_assets, self.cash_fraction, self.config.risky_tickers, self.config.cash_tickers, self.config.top_selection
            )
            
            all_tickers = self.config.risky_tickers + self.config.cash_tickers
            weights_combined = pd.concat([self.weights_risky, self.weights_cash], axis=1)
            monthly_returns_all = self.monthly_returns[all_tickers]
            self.portfolio_returns = backtest_returns(monthly_returns_all, weights_combined, self.config.transaction_cost)
        except Exception as e:
            print(f"Error in fit: {str(e)}")
        
        return self
    
    def get_signals(self) -> pd.DataFrame:
        """DAA 신호 반환 - 완전히 수정"""
        try:
            if self.momentum is None or self.bad_assets is None:
                return pd.DataFrame()
            
            # 방법: 딕셔너리로 만든 후 DataFrame 생성
            data_dict = {}
            
            # 카나리 신호 (numpy array로 변환)
            for ticker in self.config.canary_tickers:
                if ticker in self.momentum.columns:
                    data_dict[f'{ticker}_mom'] = self.momentum[ticker].values
                    data_dict[f'{ticker}_bad'] = self.bad_assets[ticker].astype(int).values
            
            # Breadth 신호
            data_dict['Breadth_Bad_Count'] = self.breadth_bad.values
            data_dict['Cash_Fraction'] = self.cash_fraction.values
            
            # DataFrame 생성 (index 명시)
            signals = pd.DataFrame(data_dict, index=self.momentum.index)
            return signals
        
        except Exception as e:
            print(f"Error in get_signals: {str(e)}")
            return pd.DataFrame()
    
    def get_weights(self) -> pd.DataFrame:
        """포트폴리오 가중치"""
        try:
            if self.weights_risky is None or self.weights_cash is None:
                return pd.DataFrame()
            
            all_tickers = self.config.risky_tickers + self.config.cash_tickers
            weights = pd.concat([self.weights_risky, self.weights_cash], axis=1)
            existing_cols = [t for t in all_tickers if t in weights.columns]
            
            if existing_cols:
                return weights[existing_cols]
            return weights
        except:
            return pd.DataFrame()
    
    def get_returns(self) -> pd.Series:
        """월별 수익률"""
        try:
            if self.portfolio_returns is None:
                return pd.Series(dtype=float)
            return self.portfolio_returns
        except:
            return pd.Series(dtype=float)
    
    def get_cumulative_returns(self) -> pd.Series:
        """누적 수익률"""
        try:
            if self.portfolio_returns is None:
                return pd.Series(dtype=float)
            return (1 + self.portfolio_returns).cumprod()
        except:
            return pd.Series(dtype=float)


class DAABacktest:
    """DAA 백테스트"""
    
    def __init__(self, daa_strategy: DAAStrategy):
        self.daa = daa_strategy
        self.performance = None
    
    def run(self, benchmark_returns: pd.Series = None, risk_free_rate: float = 0.02) -> Dict:
        """백테스트 실행"""
        try:
            strategy_returns = self.daa.get_returns()
            
            if benchmark_returns is not None and len(strategy_returns) > 0:
                common_dates = strategy_returns.index.intersection(benchmark_returns.index)
                if len(common_dates) > 0:
                    strategy_returns = strategy_returns[common_dates]
                    benchmark_returns = benchmark_returns[common_dates]
                else:
                    benchmark_returns = None
            
            self.performance = calculate_performance_metrics(strategy_returns, benchmark_returns, risk_free_rate)
        except Exception as e:
            print(f"Error in run: {str(e)}")
            self.performance = {}
        
        return self.performance if self.performance else {}
    
    def get_performance(self) -> Dict:
        """성과 지표"""
        return self.performance if self.performance else {}


class DAAComparison:
    """여러 DAA 전략 비교"""
    
    def __init__(self):
        self.strategies = {}
        self.results = {}
    
    def add_strategy(self, name: str, daa: DAAStrategy) -> 'DAAComparison':
        """전략 추가"""
        self.strategies[name] = daa
        return self
    
    def add_benchmark(self, name: str, returns: pd.Series) -> 'DAAComparison':
        """벤치마크 추가"""
        self.strategies[name] = returns
        return self
    
    def run(self, risk_free_rate: float = 0.02) -> pd.DataFrame:
        """모든 전략 비교"""
        results = {}
        
        for name, strategy in self.strategies.items():
            if isinstance(strategy, DAAStrategy):
                backtest = DAABacktest(strategy)
                perf = backtest.run(risk_free_rate=risk_free_rate)
                results[name] = perf
            elif isinstance(strategy, pd.Series):
                perf = calculate_performance_metrics(strategy, risk_free_rate=risk_free_rate)
                results[name] = perf
        
        comparison_df = pd.DataFrame(results).T
        self.results = comparison_df
        
        return comparison_df
    
    def get_summary(self) -> pd.DataFrame:
        """요약"""
        if self.results.empty:
            return pd.DataFrame()
        
        key_metrics = ['Total Return (%)', 'CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
        summary = self.results[[col for col in key_metrics if col in self.results.columns]]
        
        return summary
