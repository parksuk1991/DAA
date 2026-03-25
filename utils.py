"""
DAA 백테스트 유틸리티 함수
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


def download_price_data(
    ticker_list: List[str],
    start_date: str,
    end_date: str,
    progress_bar=None
) -> pd.DataFrame:
    """
    yfinance에서 가격 데이터 다운로드
    """
    all_data = {}

    for i, ticker in enumerate(ticker_list):
        try:
            if progress_bar:
                progress_bar.progress((i + 1) / len(ticker_list))

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )['Close']

            all_data[ticker] = data

        except Exception as e:
            print(f"⚠️ {ticker} 다운로드 실패: {str(e)}")
            continue

    if not all_data:
        raise ValueError("데이터를 받아올 수 없습니다. 티커를 확인하세요.")

    df = pd.DataFrame(all_data)
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df


def calculate_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    일일 가격을 월별 수익률로 변환
    """
    monthly_prices = price_df.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    return monthly_returns


def calculate_momentum(
    returns_df: pd.DataFrame,
    periods: List[int] = None,
    weights: List[int] = None
) -> pd.DataFrame:
    """
    13612W 모멘텀 계산 (가중 이동 평균)
    """
    if periods is None:
        periods = [1, 3, 6, 12]
    if weights is None:
        weights = [12, 4, 2, 1]

    momentum = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)

    for col in returns_df.columns:
        weighted_mom = pd.Series(0.0, index=returns_df.index)
        for period, weight in zip(periods, weights):
            cum_return = (1 + returns_df[col]).rolling(period).apply(
                lambda x: x.prod() - 1,
                raw=False
            )
            weighted_mom = weighted_mom.add(cum_return * weight, fill_value=0.0)
        total_weight = float(sum(weights))
        momentum[col] = weighted_mom / total_weight

    return momentum


def get_bad_assets(
    momentum_df: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    return momentum_df <= threshold


def count_breadth_bad(
    bad_assets_df: pd.DataFrame,
    canary_tickers: List[str]
) -> pd.Series:
    return bad_assets_df[canary_tickers].sum(axis=1)


def calculate_cash_fraction(
    breadth_bad: pd.Series,
    breadth_param: int = 2
) -> pd.Series:
    return (breadth_bad / float(breadth_param)).clip(0, 1)


def select_top_assets(
    momentum_df: pd.DataFrame,
    top_n: int = 6,
    risky_tickers: List[str] = None
) -> pd.DataFrame:
    if risky_tickers:
        momentum_risky = momentum_df[risky_tickers]
    else:
        momentum_risky = momentum_df

    top_assets = pd.DataFrame(False, index=momentum_risky.index, columns=momentum_risky.columns)

    for date in momentum_risky.index:
        row = momentum_risky.loc[date]
        if row.isna().all():
            continue
        top_indices = row.nlargest(top_n).index
        top_assets.loc[date, top_indices] = True

    return top_assets


def calculate_portfolio_weights(
    top_assets: pd.DataFrame,
    cash_fraction: pd.Series,
    risky_tickers: List[str],
    cash_tickers: List[str],
    top_n: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weights_risky = pd.DataFrame(0.0, index=top_assets.index, columns=risky_tickers)
    weights_cash = pd.DataFrame(0.0, index=top_assets.index, columns=cash_tickers)

    for date in top_assets.index:
        cf = float(cash_fraction.loc[date])
        risky_ratio = 1.0 - cf

        if risky_ratio > 0:
            top_mask = top_assets.loc[date, risky_tickers]
            top_count = int(top_mask.sum())
            if top_count > 0:
                weights_risky.loc[date, top_mask.index[top_mask]] = risky_ratio / top_count

        if cf > 0:
            weights_cash.loc[date, :] = cf / len(cash_tickers)

    return weights_risky, weights_cash


def backtest_returns(
    monthly_returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    transaction_cost: float = 0.001
) -> pd.Series:
    aligned_returns, aligned_weights = monthly_returns.align(weights_df, join='inner', axis=0)
    aligned_returns, aligned_weights = aligned_returns.align(aligned_weights, join='inner', axis=1)

    strategy_returns = (aligned_weights * aligned_returns).sum(axis=1)

    weight_changes = aligned_weights.diff().abs().sum(axis=1).fillna(0.0)
    trading_costs = weight_changes * transaction_cost

    final_returns = strategy_returns - trading_costs
    return final_returns


def calculate_performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.02
) -> Dict:
    cum_strategy = (1 + strategy_returns).cumprod()
    total_return = (cum_strategy.iloc[-1] - 1) * 100.0

    num_years = len(strategy_returns) / 12.0
    cagr = (cum_strategy.iloc[-1] ** (1.0 / num_years) - 1.0) * 100.0

    volatility = strategy_returns.std() * np.sqrt(12.0) * 100.0

    excess_return = strategy_returns.mean() * 12.0 - risk_free_rate
    sharpe_ratio = excess_return / (volatility / 100.0) if volatility > 0 else 0.0

    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(12.0)
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0.0

    running_max = cum_strategy.cummax()
    drawdown_series = cum_strategy / running_max - 1.0
    max_drawdown = drawdown_series.min() * 100.0

    win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100.0

    metrics = {
        'Total Return (%)': total_return,
        'CAGR (%)': cagr,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Win Rate (%)': win_rate,
        'Avg Monthly Return (%)': strategy_returns.mean() * 100.0,
        'Monthly Volatility (%)': strategy_returns.std() * 100.0,
    }

    if benchmark_returns is not None:
        aligned_strat, aligned_bench = strategy_returns.align(benchmark_returns, join='inner')
        cum_benchmark = (1 + aligned_bench).cumprod()
        benchmark_return = (cum_benchmark.iloc[-1] - 1) * 100.0

        covariance = np.cov(aligned_strat, aligned_bench)[0, 1]
        benchmark_var = np.var(aligned_bench)
        beta = covariance / benchmark_var if benchmark_var > 0 else 0.0
        alpha = aligned_strat.mean() - beta * aligned_bench.mean()
        excess = (cum_strategy.iloc[-1] / cum_benchmark.iloc[-1] - 1.0) * 100.0

        metrics['Benchmark Return (%)'] = benchmark_return
        metrics['Alpha (%)'] = alpha * 100.0 * 12.0
        metrics['Beta'] = beta
        metrics['Excess Return (%)'] = excess

    return metrics


def format_metrics_display(metrics: Dict) -> Dict:
    formatted = {}

    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Rate' in key or 'Return' in key or 'Volatility' in key or 'Drawdown' in key:
                formatted[key] = f"{value:.2f}%"
            elif 'Ratio' in key or 'Beta' in key or 'Alpha' in key:
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = value

    return formatted
