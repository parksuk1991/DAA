"""
DAA 백테스트 유틸리티 함수 - 완전히 수정된 버전
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
    
    # DataFrame으로 변환
    df = pd.DataFrame(all_data)
    # 최신 pandas 버전 호환
    df = df.ffill().bfill()
    
    return df


def calculate_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    일일 가격을 월별 수익률로 변환
    """
    # 월말 데이터 추출
    monthly_prices = price_df.resample('M').last()
    
    # 월별 수익률 계산
    monthly_returns = monthly_prices.pct_change()
    
    return monthly_returns


def calculate_momentum(
    returns_df: pd.DataFrame,
    periods: List[int] = [1, 3, 6, 12],
    weights: List[int] = [12, 4, 2, 1]
) -> pd.DataFrame:
    """
    13612W 모멘텀 계산 (가중 이동 평균)
    """
    momentum = pd.DataFrame(index=returns_df.index)
    
    for col in returns_df.columns:
        weighted_mom = pd.Series(0.0, index=returns_df.index)
        
        for period, weight in zip(periods, weights):
            # 누적 수익률 계산
            cum_return = (1 + returns_df[col]).rolling(period).apply(
                lambda x: (x.prod() - 1), raw=False
            )
            weighted_mom = weighted_mom + cum_return * weight
        
        # 정규화
        total_weight = sum(weights)
        momentum[col] = weighted_mom / total_weight
    
    return momentum


def get_bad_assets(
    momentum_df: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Bad 자산 식별 (모멘텀 <= threshold)
    """
    bad_assets = momentum_df <= threshold
    return bad_assets


def count_breadth_bad(
    bad_assets_df: pd.DataFrame,
    canary_tickers: List[str]
) -> pd.Series:
    """
    카나리 유니버스의 Bad 자산 개수 계산
    """
    # 컬럼 필터링
    available_tickers = [t for t in canary_tickers if t in bad_assets_df.columns]
    
    if available_tickers:
        canary_bad = bad_assets_df[available_tickers].sum(axis=1).astype(int)
    else:
        canary_bad = pd.Series(0, index=bad_assets_df.index)
    
    return canary_bad


def calculate_cash_fraction(
    breadth_bad: pd.Series,
    breadth_param: int = 2
) -> pd.Series:
    """
    현금 비율 계산 (CF = b / B)
    """
    cash_fraction = (breadth_bad / breadth_param).clip(0, 1)
    return cash_fraction


def select_top_assets(
    momentum_df: pd.DataFrame,
    top_n: int = 6,
    risky_tickers: List[str] = None
) -> pd.DataFrame:
    """
    상위 N개 자산 선택
    """
    # 사용 가능한 컬럼 필터링
    if risky_tickers:
        available_cols = [t for t in risky_tickers if t in momentum_df.columns]
        momentum_risky = momentum_df[available_cols]
    else:
        momentum_risky = momentum_df
    
    # DataFrame 생성 (모든 컬럼 포함하되, 선택된 것만 True)
    top_assets = pd.DataFrame(
        False, 
        index=momentum_df.index, 
        columns=momentum_df.columns
    )
    
    for date_idx in momentum_risky.index:
        top_indices = momentum_risky.loc[date_idx].nlargest(min(top_n, len(available_cols))).index
        top_assets.loc[date_idx, top_indices] = True
    
    return top_assets


def calculate_portfolio_weights(
    top_assets: pd.DataFrame,
    cash_fraction: pd.Series,
    risky_tickers: List[str],
    cash_tickers: List[str],
    top_n: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    포트폴리오 가중치 계산
    """
    # 사용 가능한 티커 필터링
    available_risky = [t for t in risky_tickers if t in top_assets.columns]
    available_cash = [t for t in cash_tickers if t in top_assets.columns]
    
    weights_risky = pd.DataFrame(
        0.0, 
        index=top_assets.index, 
        columns=available_risky
    )
    weights_cash = pd.DataFrame(
        0.0, 
        index=top_assets.index, 
        columns=available_cash
    )
    
    for date_idx in top_assets.index:
        # 현금 비율
        cf = float(cash_fraction.loc[date_idx])
        risky_ratio = 1.0 - cf
        
        # 위험자산 가중치
        if risky_ratio > 0 and len(available_risky) > 0:
            top_count = sum([top_assets.loc[date_idx, t] for t in available_risky if t in top_assets.columns])
            if top_count > 0:
                for ticker in available_risky:
                    if ticker in top_assets.columns and top_assets.loc[date_idx, ticker]:
                        weights_risky.loc[date_idx, ticker] = risky_ratio / top_count
        
        # 현금 가중치
        if cf > 0 and len(available_cash) > 0:
            weights_cash.loc[date_idx, :] = cf / len(available_cash)
    
    return weights_risky, weights_cash


def backtest_returns(
    monthly_returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    transaction_cost: float = 0.001
) -> pd.Series:
    """
    백테스트 수익률 계산
    """
    # 공통 컬럼 찾기
    common_cols = [col for col in weights_df.columns if col in monthly_returns.columns]
    
    if not common_cols:
        raise ValueError("weights_df와 monthly_returns의 공통 컬럼이 없습니다.")
    
    # 해당 월의 수익률
    strategy_returns = (weights_df[common_cols] * monthly_returns[common_cols]).sum(axis=1)
    
    # 거래 비용 반영
    weight_changes = weights_df[common_cols].diff().abs().sum(axis=1)
    trading_costs = weight_changes * transaction_cost
    
    # 최종 수익률
    final_returns = strategy_returns - trading_costs
    
    return final_returns


def calculate_performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    성과 지표 계산
    """
    # NaN 값 제거
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0:
        return {
            'Total Return (%)': 0.0,
            'CAGR (%)': 0.0,
            'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Max Drawdown (%)': 0.0,
            'Win Rate (%)': 0.0,
            'Avg Monthly Return (%)': 0.0,
            'Monthly Volatility (%)': 0.0
        }
    
    # 누적 수익률
    cum_strategy = (1 + strategy_returns).cumprod()
    total_return = float((cum_strategy.iloc[-1] - 1) * 100)
    
    # 연평균 수익률 (CAGR)
    num_years = len(strategy_returns) / 12
    if num_years > 0:
        cagr = float(((cum_strategy.iloc[-1]) ** (1 / num_years) - 1) * 100)
    else:
        cagr = 0.0
    
    # 변동성
    volatility = float(strategy_returns.std() * np.sqrt(12) * 100)
    
    # Sharpe Ratio
    excess_return = float(strategy_returns.mean() * 12 - risk_free_rate)
    sharpe_ratio = float(excess_return / (volatility / 100) if volatility > 0 else 0)
    
    # Sortino Ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = float(downside_returns.std() * np.sqrt(12)) if len(downside_returns) > 0 else 0.0
    sortino_ratio = float(excess_return / downside_std if downside_std > 0 else 0)
    
    # 최대 낙폭
    running_max = cum_strategy.expanding().max()
    drawdown = float(((cum_strategy / running_max) - 1).min() * 100)
    
    # 승률
    win_rate = float((strategy_returns > 0).sum() / len(strategy_returns) * 100)
    
    metrics = {
        'Total Return (%)': total_return,
        'CAGR (%)': cagr,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': drawdown,
        'Win Rate (%)': win_rate,
        'Avg Monthly Return (%)': float(strategy_returns.mean() * 100),
        'Monthly Volatility (%)': float(strategy_returns.std() * 100)
    }
    
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        
        # 공통 기간으로 정렬
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 0:
            strategy_returns_aligned = strategy_returns[common_idx]
            benchmark_returns_aligned = benchmark_returns[common_idx]
            
            cum_benchmark = (1 + benchmark_returns_aligned).cumprod()
            benchmark_return = float((cum_benchmark.iloc[-1] - 1) * 100)
            
            # 알파/베타
            try:
                covariance = float(np.cov(strategy_returns_aligned, benchmark_returns_aligned)[0, 1])
                benchmark_var = float(np.var(benchmark_returns_aligned))
                beta = float(covariance / benchmark_var if benchmark_var > 0 else 0)
                alpha = float((strategy_returns_aligned.mean() - beta * benchmark_returns_aligned.mean()) * 100 * 12)
            except:
                beta = 0.0
                alpha = 0.0
            
            # 초과 수익률
            try:
                excess = float((cum_strategy.iloc[-1] / cum_benchmark.iloc[-1] - 1) * 100)
            except:
                excess = 0.0
            
            metrics['Benchmark Return (%)'] = benchmark_return
            metrics['Alpha (%)'] = alpha
            metrics['Beta'] = beta
            metrics['Excess Return (%)'] = excess
    
    return metrics


def format_metrics_display(metrics: Dict) -> Dict:
    """
    성과 지표를 표시 형식으로 변환
    """
    formatted = {}
    
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            if 'Rate' in key or 'Return' in key or 'Volatility' in key or 'Drawdown' in key:
                formatted[key] = f"{value:.2f}%"
            elif 'Ratio' in key or 'Beta' in key or 'Alpha' in key:
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = value
    
    return formatted
