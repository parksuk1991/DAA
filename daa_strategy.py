"""
DAA (Defensive Asset Allocation) 전략 구현
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
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
    """
    DAA (Defensive Asset Allocation) 전략
    
    카나리 유니버스의 모멘텀을 이용한 동적 자산배분 전략
    """
    
    def __init__(self, config: DAAConfig):
        """
        Parameters:
        -----------
        config : DAAConfig
            DAA 전략 설정
        """
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
        """
        가격 데이터로 전략 학습
        
        Parameters:
        -----------
        price_df : pd.DataFrame
            일일 수정 종가 데이터
            
        Returns:
        --------
        DAAStrategy
            학습된 전략 객체
        """
        self.price_data = price_df
        
        # 1. 월별 수익률 계산
        self.monthly_returns = calculate_monthly_returns(price_df)
        
        # 2. 모멘텀 계산
        self.momentum = calculate_momentum(
            self.monthly_returns,
            self.config.momentum_periods,
            self.config.momentum_weights
        )
        
        # 3. Bad 자산 식별
        self.bad_assets = get_bad_assets(self.momentum, threshold=0.0)
        
        # 4. 카나리 유니버스의 Bad 자산 개수
        self.breadth_bad = count_breadth_bad(
            self.bad_assets,
            self.config.canary_tickers
        )
        
        # 5. 현금 비율 계산
        self.cash_fraction = calculate_cash_fraction(
            self.breadth_bad,
            self.config.breadth_parameter
        )
        
        # 6. 상위 자산 선택
        self.top_assets = select_top_assets(
            self.momentum,
            self.config.top_selection,
            self.config.risky_tickers
        )
        
        # 7. 포트폴리오 가중치 계산
        self.weights_risky, self.weights_cash = calculate_portfolio_weights(
            self.top_assets,
            self.cash_fraction,
            self.config.risky_tickers,
            self.config.cash_tickers,
            self.config.top_selection
        )
        
        # 8. 모든 가중치 합치기
        all_tickers = self.config.risky_tickers + self.config.cash_tickers
        weights_combined = pd.concat([self.weights_risky, self.weights_cash], axis=1)
        
        # 9. 포트폴리오 수익률 계산
        monthly_returns_all = self.monthly_returns[all_tickers]
        self.portfolio_returns = backtest_returns(
            monthly_returns_all,
            weights_combined,
            self.config.transaction_cost
        )
        
        return self
    
    def get_signals(self) -> pd.DataFrame:
        """
        DAA 신호 반환
        
        Returns:
        --------
        pd.DataFrame
            신호 데이터 (모멘텀, Bad 여부, 현금 비율 등)
        """
        signals = pd.DataFrame(index=self.momentum.index)
        
        # 카나리 자산의 모멘텀
        for ticker in self.config.canary_tickers:
            signals[f'{ticker}_mom'] = self.momentum[ticker]
            signals[f'{ticker}_bad'] = self.bad_assets[ticker]
        
        # Breadth 신호
        signals['Breadth_Bad_Count'] = self.breadth_bad
        signals['Cash_Fraction'] = self.cash_fraction
        
        return signals
    
    def get_weights(self) -> pd.DataFrame:
        """
        포트폴리오 가중치 반환
        
        Returns:
        --------
        pd.DataFrame
            월별 포트폴리오 가중치
        """
        all_tickers = self.config.risky_tickers + self.config.cash_tickers
        weights = pd.concat([self.weights_risky, self.weights_cash], axis=1)
        weights = weights[all_tickers]  # 정렬
        
        return weights
    
    def get_returns(self) -> pd.Series:
        """
        포트폴리오 월별 수익률 반환
        
        Returns:
        --------
        pd.Series
            월별 수익률
        """
        return self.portfolio_returns
    
    def get_cumulative_returns(self) -> pd.Series:
        """
        누적 수익률 반환 (인덱스 = 1부터 시작)
        
        Returns:
        --------
        pd.Series
            누적 수익률
        """
        return (1 + self.portfolio_returns).cumprod()


class DAABacktest:
    """DAA 전략 백테스트 엔진"""
    
    def __init__(self, daa_strategy: DAAStrategy):
        """
        Parameters:
        -----------
        daa_strategy : DAAStrategy
            학습된 DAA 전략
        """
        self.daa = daa_strategy
        self.performance = None
    
    def run(
        self,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        백테스트 실행
        
        Parameters:
        -----------
        benchmark_returns : pd.Series
            벤치마크 수익률 (선택사항)
        risk_free_rate : float
            무위험 수익률
            
        Returns:
        --------
        Dict
            성과 지표
        """
        strategy_returns = self.daa.get_returns()
        
        if benchmark_returns is not None:
            # 벤치마크와 동일 기간으로 정렬
            common_dates = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_returns = strategy_returns[common_dates]
            benchmark_returns = benchmark_returns[common_dates]
        
        self.performance = calculate_performance_metrics(
            strategy_returns,
            benchmark_returns,
            risk_free_rate
        )
        
        return self.performance
    
    def get_performance(self) -> Dict:
        """
        성과 지표 반환
        
        Returns:
        --------
        Dict
            성과 지표
        """
        if self.performance is None:
            raise ValueError("먼저 run() 메서드를 호출하세요.")
        
        return self.performance


class DAAComparison:
    """여러 DAA 전략 비교"""
    
    def __init__(self):
        self.strategies = {}
        self.results = {}
    
    def add_strategy(self, name: str, daa: DAAStrategy) -> 'DAAComparison':
        """
        전략 추가
        
        Parameters:
        -----------
        name : str
            전략 이름
        daa : DAAStrategy
            DAA 전략
            
        Returns:
        --------
        DAAComparison
            자신 (메서드 체인용)
        """
        self.strategies[name] = daa
        return self
    
    def add_benchmark(self, name: str, returns: pd.Series) -> 'DAAComparison':
        """
        벤치마크 추가
        
        Parameters:
        -----------
        name : str
            벤치마크 이름
        returns : pd.Series
            벤치마크 수익률
            
        Returns:
        --------
        DAAComparison
            자신 (메서드 체인용)
        """
        self.strategies[name] = returns
        return self
    
    def run(self, risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        모든 전략 비교
        
        Parameters:
        -----------
        risk_free_rate : float
            무위험 수익률
            
        Returns:
        --------
        pd.DataFrame
            비교 결과
        """
        results = {}
        
        for name, strategy in self.strategies.items():
            if isinstance(strategy, DAAStrategy):
                backtest = DAABacktest(strategy)
                perf = backtest.run(risk_free_rate=risk_free_rate)
                results[name] = perf
            elif isinstance(strategy, pd.Series):
                # 벤치마크 수익률
                perf = calculate_performance_metrics(strategy, risk_free_rate=risk_free_rate)
                results[name] = perf
        
        # DataFrame으로 변환
        comparison_df = pd.DataFrame(results).T
        self.results = comparison_df
        
        return comparison_df
    
    def get_summary(self) -> pd.DataFrame:
        """
        요약 통계
        
        Returns:
        --------
        pd.DataFrame
            요약 통계
        """
        if self.results.empty:
            raise ValueError("먼저 run() 메서드를 호출하세요.")
        
        # 주요 지표만 추출
        key_metrics = [
            'Total Return (%)',
            'CAGR (%)',
            'Volatility (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)'
        ]
        
        summary = self.results[[col for col in key_metrics if col in self.results.columns]]
        
        return summary
