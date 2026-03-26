"""
DAA 백테스트 유틸리티 - 최종 수정
"""

import pandas as pd
import numpy as np
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
    """가격 데이터 다운로드"""
    all_data = {}
    
    for i, ticker in enumerate(ticker_list):
        try:
            if progress_bar:
                progress_bar.progress(min((i + 1) / len(ticker_list), 0.99))
            
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
            
            if data is not None and len(data) > 0:
                all_data[ticker] = data
        except:
            continue
    
    if not all_data:
        raise ValueError("데이터를 받아올 수 없습니다.")
    
    df = pd.DataFrame(all_data)
    df = df.ffill().bfill()
    
    return df


def calculate_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """월별 수익률"""
    try:
        monthly_prices = price_df.resample('M').last()
        monthly_returns = monthly_prices.pct_change()
        return monthly_returns
    except:
        return pd.DataFrame()


def calculate_momentum(
    returns_df: pd.DataFrame,
    periods: List[int] = None,
    weights: List[int] = None
) -> pd.DataFrame:
    """13612W 모멘텀"""
    if periods is None:
        periods = [1, 3, 6, 12]
    if weights is None:
        weights = [12, 4, 2, 1]
    
    try:
        momentum = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
        
        for col in returns_df.columns:
            weighted_mom = pd.Series(0.0, index=returns_df.index)
            
            for period, weight in zip(periods, weights):
                cum_return = (1 + returns_df[col]).rolling(period).apply(
                    lambda x: float(np.prod(x) - 1) if len(x) == period else np.nan,
                    raw=False
                )
                weighted_mom = weighted_mom + (cum_return * weight)
            
            total_weight = sum(weights)
            momentum[col] = weighted_mom / total_weight
        
        return momentum
    except:
        return pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)


def get_bad_assets(momentum_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Bad 자산 식별"""
    try:
        bad_assets = momentum_df <= threshold
        return bad_assets
    except:
        return pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)


def count_breadth_bad(bad_assets_df: pd.DataFrame, canary_tickers: List[str]) -> pd.Series:
    """카나리 Bad 개수"""
    try:
        available = [t for t in canary_tickers if t in bad_assets_df.columns]
        
        if available:
            canary_bad = bad_assets_df[available].sum(axis=1).astype(int)
        else:
            canary_bad = pd.Series(0, index=bad_assets_df.index, dtype=int)
        
        return canary_bad
    except:
        return pd.Series(0, index=bad_assets_df.index, dtype=int)


def calculate_cash_fraction(breadth_bad: pd.Series, breadth_param: int = 2) -> pd.Series:
    """현금 비율"""
    try:
        cash_fraction = (breadth_bad / float(breadth_param)).clip(0, 1)
        return cash_fraction
    except:
        return pd.Series(0.0, index=breadth_bad.index)


def select_top_assets(
    momentum_df: pd.DataFrame,
    top_n: int = 6,
    risky_tickers: List[str] = None
) -> pd.DataFrame:
    """상위 N개 자산"""
    try:
        if risky_tickers:
            available = [t for t in risky_tickers if t in momentum_df.columns]
            momentum_risky = momentum_df[available]
        else:
            momentum_risky = momentum_df
        
        top_assets = pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)
        
        for date_idx in momentum_risky.index:
            row = momentum_risky.loc[date_idx]
            top_count = min(top_n, len(available) if risky_tickers else len(momentum_risky.columns))
            top_indices = row.nlargest(top_count).index
            top_assets.loc[date_idx, top_indices] = True
        
        return top_assets
    except:
        return pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)


def calculate_portfolio_weights(
    top_assets: pd.DataFrame,
    cash_fraction: pd.Series,
    risky_tickers: List[str],
    cash_tickers: List[str],
    top_n: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """포트폴리오 가중치 - 핵심 수정: scalar 할당 문제 해결"""
    try:
        available_risky = [t for t in risky_tickers if t in top_assets.columns]
        available_cash = [t for t in cash_tickers if t in top_assets.columns]
        
        weights_risky = pd.DataFrame(0.0, index=top_assets.index, columns=available_risky)
        weights_cash = pd.DataFrame(0.0, index=top_assets.index, columns=available_cash)
        
        for date_idx in top_assets.index:
            cf_val = float(cash_fraction.loc[date_idx])
            risky_ratio = 1.0 - cf_val
            
            # 위험자산 가중치 - .at[] 메서드 사용 (scalar 할당용)
            if risky_ratio > 0 and len(available_risky) > 0:
                row_assets = top_assets.loc[date_idx, available_risky]
                top_count = int(row_assets.sum())
                
                if top_count > 0:
                    weight_val = risky_ratio / top_count
                    for ticker in available_risky:
                        if top_assets.at[date_idx, ticker]:  # .at[] 사용
                            weights_risky.at[date_idx, ticker] = weight_val  # .at[] 사용
            
            # 현금 가중치 - for 루프로 각각 할당
            if cf_val > 0 and len(available_cash) > 0:
                weight_val = cf_val / len(available_cash)
                for ticker in available_cash:
                    weights_cash.at[date_idx, ticker] = weight_val  # .at[] 사용
        
        return weights_risky, weights_cash
    
    except Exception as e:
        print(f"Error in calculate_portfolio_weights: {str(e)}")
        return (
            pd.DataFrame(0.0, index=top_assets.index, columns=risky_tickers),
            pd.DataFrame(0.0, index=top_assets.index, columns=cash_tickers)
        )


def backtest_returns(
    monthly_returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    transaction_cost: float = 0.001
) -> pd.Series:
    """백테스트 수익률"""
    try:
        common_cols = [col for col in weights_df.columns if col in monthly_returns.columns]
        
        if not common_cols:
            return pd.Series(0.0, index=weights_df.index)
        
        strategy_returns = (weights_df[common_cols] * monthly_returns[common_cols]).sum(axis=1)
        weight_changes = weights_df[common_cols].diff().abs().sum(axis=1)
        trading_costs = weight_changes * transaction_cost
        
        final_returns = strategy_returns - trading_costs
        
        return final_returns
    except:
        return pd.Series(0.0, index=weights_df.index)


def calculate_performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.02
) -> Dict:
    """성과 지표"""
    try:
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {
                'Total Return (%)': 0.0, 'CAGR (%)': 0.0, 'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0, 'Max Drawdown (%)': 0.0,
                'Win Rate (%)': 0.0, 'Avg Monthly Return (%)': 0.0, 'Monthly Volatility (%)': 0.0
            }
        
        cum_strategy = (1 + strategy_returns).cumprod()
        total_return = float((cum_strategy.iloc[-1] - 1) * 100)
        
        num_years = max(len(strategy_returns) / 12, 0.01)
        cagr = float(((cum_strategy.iloc[-1]) ** (1.0 / num_years) - 1) * 100)
        
        volatility = float(strategy_returns.std() * np.sqrt(12) * 100)
        
        excess_return = float(strategy_returns.mean() * 12 - risk_free_rate)
        sharpe_ratio = float(excess_return / (volatility / 100) if volatility > 1e-6 else 0.0)
        
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = float(downside_returns.std() * np.sqrt(12)) if len(downside_returns) > 0 else 0.0
        sortino_ratio = float(excess_return / downside_std if downside_std > 1e-6 else 0.0)
        
        running_max = cum_strategy.expanding().max()
        drawdown = float(((cum_strategy / running_max) - 1).min() * 100)
        
        win_rate = float((strategy_returns > 0).sum() / len(strategy_returns) * 100)
        
        metrics = {
            'Total Return (%)': total_return, 'CAGR (%)': cagr, 'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio, 'Sortino Ratio': sortino_ratio, 'Max Drawdown (%)': drawdown,
            'Win Rate (%)': win_rate, 'Avg Monthly Return (%)': float(strategy_returns.mean() * 100),
            'Monthly Volatility (%)': float(strategy_returns.std() * 100)
        }
        
        if benchmark_returns is not None:
            try:
                benchmark_returns = benchmark_returns.dropna()
                common_idx = strategy_returns.index.intersection(benchmark_returns.index)
                
                if len(common_idx) > 0:
                    strategy_ret_aligned = strategy_returns[common_idx]
                    benchmark_ret_aligned = benchmark_returns[common_idx]
                    
                    cum_benchmark = (1 + benchmark_ret_aligned).cumprod()
                    benchmark_return = float((cum_benchmark.iloc[-1] - 1) * 100)
                    
                    try:
                        cov_matrix = np.cov(strategy_ret_aligned.values, benchmark_ret_aligned.values)
                        covariance = float(cov_matrix[0, 1])
                        benchmark_var = float(np.var(benchmark_ret_aligned.values))
                        beta = float(covariance / benchmark_var if benchmark_var > 1e-6 else 0.0)
                        alpha = float((strategy_ret_aligned.mean() - beta * benchmark_ret_aligned.mean()) * 100 * 12)
                    except:
                        beta = 0.0
                        alpha = 0.0
                    
                    try:
                        excess = float((cum_strategy.iloc[-1] / cum_benchmark.iloc[-1] - 1) * 100)
                    except:
                        excess = 0.0
                    
                    metrics.update({
                        'Benchmark Return (%)': benchmark_return,
                        'Alpha (%)': alpha,
                        'Beta': beta,
                        'Excess Return (%)': excess
                    })
            except:
                pass
        
        return metrics
    
    except Exception as e:
        return {
            'Total Return (%)': 0.0, 'CAGR (%)': 0.0, 'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0, 'Max Drawdown (%)': 0.0,
            'Win Rate (%)': 0.0, 'Avg Monthly Return (%)': 0.0, 'Monthly Volatility (%)': 0.0
        }


def format_metrics_display(metrics: Dict) -> Dict:
    """지표 포매팅"""
    formatted = {}
    
    for key, value in metrics.items():
        try:
            if isinstance(value, (float, np.floating, int)):
                if any(x in key for x in ['Rate', 'Return', 'Volatility', 'Drawdown']):
                    formatted[key] = f"{float(value):.2f}%"
                elif any(x in key for x in ['Ratio', 'Beta', 'Alpha']):
                    formatted[key] = f"{float(value):.4f}"
                else:
                    formatted[key] = f"{float(value):.2f}"
            else:
                formatted[key] = str(value)
        except:
            formatted[key] = "N/A"
    
    return formatted
