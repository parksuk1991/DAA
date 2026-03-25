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
    
    Parameters:
    -----------
    ticker_list : List[str]
        다운로드할 티커 리스트
    start_date : str
        시작 날짜 ('YYYY-MM-DD')
    end_date : str
        종료 날짜 ('YYYY-MM-DD')
    progress_bar : streamlit progress bar (optional)
        
    Returns:
    --------
    pd.DataFrame
        일일 수정 종가 데이터
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
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


def calculate_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    일일 가격을 월별 수익률로 변환
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        일일 수정 종가 데이터
        
    Returns:
    --------
    pd.DataFrame
        월말 기준 월별 수익률
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
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        월별 수익률
    periods : List[int]
        모멘텀 계산 기간 (월)
    weights : List[int]
        각 기간의 가중치
        
    Returns:
    --------
    pd.DataFrame
        13612W 모멘텀 값
    """
    momentum = pd.DataFrame(index=returns_df.index)
    
    for col in returns_df.columns:
        weighted_mom = pd.Series(0.0, index=returns_df.index)
        
        for period, weight in zip(periods, weights):
            # 누적 수익률 계산
            cum_return = (1 + returns_df[col]).rolling(period).apply(
                lambda x: (x.prod() - 1), raw=False
            )
            weighted_mom += cum_return * weight
        
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
    
    Parameters:
    -----------
    momentum_df : pd.DataFrame
        모멘텀 DataFrame
    threshold : float
        Bad 판정 기준값 (기본값: 0)
        
    Returns:
    --------
    pd.DataFrame
        Bad 자산 여부 (True/False)
    """
    bad_assets = momentum_df <= threshold
    return bad_assets


def count_breadth_bad(
    bad_assets_df: pd.DataFrame,
    canary_tickers: List[str]
) -> pd.Series:
    """
    카나리 유니버스의 Bad 자산 개수 계산 (Breadth)
    
    Parameters:
    -----------
    bad_assets_df : pd.DataFrame
        Bad 자산 여부
    canary_tickers : List[str]
        카나리 유니버스 티커 리스트
        
    Returns:
    --------
    pd.Series
        각 시점의 Bad 자산 개수
    """
    canary_bad = bad_assets_df[canary_tickers].sum(axis=1)
    return canary_bad


def calculate_cash_fraction(
    breadth_bad: pd.Series,
    breadth_param: int = 2
) -> pd.Series:
    """
    현금 비율 계산 (CF = b / B)
    
    Parameters:
    -----------
    breadth_bad : pd.Series
        Bad 자산 개수
    breadth_param : int
        Breadth 파라미터 (B)
        
    Returns:
    --------
    pd.Series
        각 시점의 현금 비율 (0~1)
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
    
    Parameters:
    -----------
    momentum_df : pd.DataFrame
        모멘텀 데이터
    top_n : int
        선택할 상위 자산 개수
    risky_tickers : List[str]
        위험자산 티커 (선택사항)
        
    Returns:
    --------
    pd.DataFrame
        상위 자산 여부 (True/False)
    """
    top_assets = pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)
    
    if risky_tickers:
        momentum_risky = momentum_df[risky_tickers]
    else:
        momentum_risky = momentum_df
    
    for date in momentum_risky.index:
        top_indices = momentum_risky.loc[date].nlargest(top_n).index
        top_assets.loc[date, top_indices] = True
    
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
    
    Parameters:
    -----------
    top_assets : pd.DataFrame
        상위 자산 여부
    cash_fraction : pd.Series
        현금 비율
    risky_tickers : List[str]
        위험자산 티커
    cash_tickers : List[str]
        현금/채권 티커
    top_n : int
        선택된 상위 자산 개수
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (위험자산 가중치, 현금 가중치)
    """
    weights_risky = pd.DataFrame(0.0, index=top_assets.index, columns=risky_tickers)
    weights_cash = pd.DataFrame(0.0, index=top_assets.index, columns=cash_tickers)
    
    for date in top_assets.index:
        # 현금 비율
        cf = cash_fraction.loc[date]
        risky_ratio = 1 - cf
        
        # 위험자산 가중치
        if risky_ratio > 0:
            top_count = top_assets.loc[date].sum()
            if top_count > 0:
                for ticker in risky_tickers:
                    if top_assets.loc[date, ticker]:
                        weights_risky.loc[date, ticker] = risky_ratio / top_count
        
        # 현금 가중치
        if cf > 0:
            weights_cash.loc[date, :] = cf / len(cash_tickers)
    
    return weights_risky, weights_cash


def backtest_returns(
    monthly_returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    transaction_cost: float = 0.001
) -> pd.Series:
    """
    백테스트 수익률 계산
    
    Parameters:
    -----------
    monthly_returns : pd.DataFrame
        월별 수익률
    weights_df : pd.DataFrame
        포트폴리오 가중치
    transaction_cost : float
        거래 비용
        
    Returns:
    --------
    pd.Series
        월별 포트폴리오 수익률
    """
    # 해당 월의 수익률
    strategy_returns = (weights_df * monthly_returns).sum(axis=1)
    
    # 거래 비용 반영 (가중치 변화율)
    weight_changes = weights_df.diff().abs().sum(axis=1)
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
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        전략 월별 수익률
    benchmark_returns : pd.Series
        벤치마크 월별 수익률 (선택사항)
    risk_free_rate : float
        무위험 수익률 (연율)
        
    Returns:
    --------
    Dict
        성과 지표 모음
    """
    # 누적 수익률
    cum_strategy = (1 + strategy_returns).cumprod()
    total_return = (cum_strategy.iloc[-1] - 1) * 100
    
    # 연평균 수익률 (CAGR)
    num_years = len(strategy_returns) / 12
    cagr = ((cum_strategy.iloc[-1]) ** (1 / num_years) - 1) * 100
    
    # 변동성
    volatility = strategy_returns.std() * np.sqrt(12) * 100
    
    # Sharpe Ratio
    excess_return = strategy_returns.mean() * 12 - risk_free_rate
    sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0
    
    # Sortino Ratio (하방 편차만 고려)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(12)
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
    
    # 최대 낙폭
    cum_strategy_expanded = cum_strategy.expand(len(cum_strategy))
    running_max = cum_strategy.expanding().max()
    drawdown = (cum_strategy / running_max - 1).min() * 100
    
    # 승률
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100
    
    metrics = {
        'Total Return (%)': total_return,
        'CAGR (%)': cagr,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': drawdown,
        'Win Rate (%)': win_rate,
        'Avg Monthly Return (%)': strategy_returns.mean() * 100,
        'Monthly Volatility (%)': strategy_returns.std() * 100
    }
    
    if benchmark_returns is not None:
        # 벤치마크 지표
        cum_benchmark = (1 + benchmark_returns).cumprod()
        benchmark_return = (cum_benchmark.iloc[-1] - 1) * 100
        
        # 알파/베타
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_var = np.var(benchmark_returns)
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
        
        # 초과 수익률
        excess = (cum_strategy.iloc[-1] / cum_benchmark.iloc[-1] - 1) * 100
        
        metrics['Benchmark Return (%)'] = benchmark_return
        metrics['Alpha (%)'] = alpha * 100 * 12
        metrics['Beta'] = beta
        metrics['Excess Return (%)'] = excess
    
    return metrics


def format_metrics_display(metrics: Dict) -> Dict:
    """
    성과 지표를 표시 형식으로 변환
    
    Parameters:
    -----------
    metrics : Dict
        성과 지표
        
    Returns:
    --------
    Dict
        포매팅된 지표
    """
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
