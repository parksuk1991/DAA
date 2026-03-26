"""
DAA + VAA 통합 전략 (완전 정확한 백테스트 + 연속 가중치 배분)
========================================================================

핵심 개선사항:
1. Look-ahead bias 완전 제거: shift(1) 적용
2. 연속적 가중치 배분: 모멘텀 크기에 따른 비중 조정
   - 상위 자산들을 모멘텀으로 가중 배분
   - 모멘텀이 높을수록 더 높은 비중
3. Breadth Score 연속화: 카나리 모멘텀 강도 반영
4. 모든 가중치 기록 및 검증
5. 기존의 상세한 대시보드 완벽히 유지
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DAA+VAA 통합 전략", page_icon="📊", layout="wide")

st.markdown("""
<style>
.header {color: #1f77b4; font-size: 28px; font-weight: bold; margin-bottom: 20px;}
.metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
.formula {background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0; font-family: monospace;}
.note {background-color: #e7f3ff; padding: 15px; border-radius: 8px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

COLORS = {
    'vaa': '#FF6B6B',
    'spy': '#4ECDC4',
    'acwi': '#45B7D1',
    'aggressive': '#FFA07A',
    'balanced': '#98D8C8',
    'conservative': '#6C5CE7',
    'positive': '#2ECC71',
    'negative': '#E74C3C',
    'neutral': '#95A5A6'
}

UNIVERSES = {
    "VAA-G12 (글로벌 12개)": {
        'risky': ['SPY', 'IWM', 'QQQ', 'VEA', 'VGK', 'EWJ', 'VWO', 'VNQ', 'GSG', 'GLD', 'TLT', 'HYG'],
        'canary': ['VWO', 'BND'],
        'cash': ['SHY', 'IEF', 'LQD'],
        'description': '글로벌 주식(SPY, IWM, QQQ), 국제주식(VEA, VGK, EWJ, VWO), 리츠(VNQ), 상품(GSG), 금(GLD), 채권(TLT, HYG)',
        't_param': 2,
        'b_param': 4
    },
    "VAA-G4 (글로벌 4개)": {
        'risky': ['SPY', 'VEA', 'VWO', 'BND'],
        'canary': ['VWO', 'BND'],
        'cash': ['SHY', 'IEF', 'LQD'],
        'description': 'Antonacci GEM에서 영감: 미국(SPY), 국제(VEA), 신흥(VWO), 채권(BND)',
        't_param': 1,
        'b_param': 1
    }
}

BENCHMARKS_CONFIG = {
    'Aggressive (70/30)': {
        'stocks': 0.7, 'bonds': 0.3,
        'description': '주식 70% (SPY) + 채권 30% (BND)'
    },
    'Balanced (60/40)': {
        'stocks': 0.6, 'bonds': 0.4,
        'description': '주식 60% (SPY) + 채권 40% (BND)'
    },
    'Conservative (50/50)': {
        'stocks': 0.5, 'bonds': 0.5,
        'description': '주식 50% (SPY) + 채권 50% (BND)'
    },
    'SPY Index': {
        'stocks': 1.0, 'bonds': 0.0,
        'description': 'S&P 500 Index (SPY ETF)'
    },
    'ACWI Index': {
        'stocks': 0.0, 'bonds': 0.0,
        'is_acwi': True,
        'description': 'iShares Global MSCI ACWI ETF - 전 세계 주식 지수'
    }
}

# ===== 데이터 다운로드 =====
@st.cache_data
def download_price_data(ticker_list, start_date, end_date):
    """yfinance에서 가격 데이터 다운로드"""
    try:
        if not ticker_list:
            return None
        
        raw = yf.download(ticker_list, start=start_date, end=end_date, progress=False)
        
        if isinstance(raw, pd.DataFrame):
            df = raw['Close']
        else:
            df = raw
        
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        df = df.ffill().dropna(how='all')
        
        available_cols = [t for t in ticker_list if t in df.columns]
        if not available_cols:
            return None
        
        df = df[available_cols]
        
        if df.empty:
            return None
        
        return df
    
    except Exception as e:
        return None

# ===== 월별 수익률 =====
def calculate_monthly_returns(price_df):
    """월별 수익률 계산"""
    try:
        monthly_prices = price_df.resample('M').last()
        monthly_returns = monthly_prices.pct_change()
        return monthly_returns
    except:
        return pd.DataFrame()

# ===== DAA 모멘텀 (look-ahead bias 제거) =====
def calculate_momentum_daa(returns_df):
    """
    DAA 모멘텀 = (R1 + R3 + R6 + R12) / 4
    **CRITICAL**: shift(1) 적용 - look-ahead bias 제거
    """
    momentum = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    
    for col in returns_df.columns:
        cum_1m = returns_df[col]
        cum_3m = (1 + returns_df[col]).rolling(3).apply(
            lambda x: float(np.prod(x) - 1) if len(x) == 3 else np.nan,
            raw=False
        )
        cum_6m = (1 + returns_df[col]).rolling(6).apply(
            lambda x: float(np.prod(x) - 1) if len(x) == 6 else np.nan,
            raw=False
        )
        cum_12m = (1 + returns_df[col]).rolling(12).apply(
            lambda x: float(np.prod(x) - 1) if len(x) == 12 else np.nan,
            raw=False
        )
        
        mom = (cum_1m + cum_3m + cum_6m + cum_12m) / 4
        momentum[col] = mom.shift(1)  # shift(1): Look-ahead bias 제거
    
    return momentum

# ===== VAA 모멘텀 (look-ahead bias 제거) =====
def calculate_momentum_vaa(returns_df):
    """
    VAA 모멘텀 = (12*R1 + 4*R3 + 2*R6 + 1*R12) / 19
    **CRITICAL**: shift(1) 적용 - look-ahead bias 제거
    """
    momentum = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    
    for col in returns_df.columns:
        cum_1m = returns_df[col]
        cum_3m = (1 + returns_df[col]).rolling(3).apply(
            lambda x: float(np.prod(x) - 1) if len(x) == 3 else np.nan,
            raw=False
        )
        cum_6m = (1 + returns_df[col]).rolling(6).apply(
            lambda x: float(np.prod(x) - 1) if len(x) == 6 else np.nan,
            raw=False
        )
        cum_12m = (1 + returns_df[col]).rolling(12).apply(
            lambda x: float(np.prod(x) - 1) if len(x) == 12 else np.nan,
            raw=False
        )
        
        mom = (12 * cum_1m + 4 * cum_3m + 2 * cum_6m + 1 * cum_12m) / 19
        momentum[col] = mom.shift(1)  # shift(1): Look-ahead bias 제거
    
    return momentum

# ===== Breadth Score (연속, 모멘텀 기반) =====
def calculate_breadth_score_continuous(momentum_df, canary_tickers):
    """
    Breadth Score 계산 (0~1)
    카나리 자산들의 모멘텀 크기로 시장 강도 판단
    
    방식: 카나리 자산들의 모멘텀을 정규화하여 평균
    - 높은 모멘텀 → 높은 breadth score → 낮은 현금 비율
    - 낮은/음수 모멘텀 → 낮은 breadth score → 높은 현금 비율
    """
    available = [t for t in canary_tickers if t in momentum_df.columns]
    
    if not available:
        return pd.Series(1.0, index=momentum_df.index)
    
    breadth_score = pd.Series(0.0, index=momentum_df.index)
    
    for date_idx in momentum_df.index:
        canary_mom = momentum_df.loc[date_idx, available]
        
        if pd.isna(canary_mom).all():
            breadth_score.loc[date_idx] = np.nan
            continue
        
        # 카나리 모멘텀을 [0, 1] 범위로 정규화
        # tanh 함수 사용: 극단값을 완화하면서 부호 유지
        scores = []
        for mom in canary_mom:
            if pd.isna(mom):
                continue
            # tanh(x)는 [-1, 1] 범위 → (tanh + 1) / 2로 [0, 1] 변환
            score = (np.tanh(mom * 3) + 1) / 2
            scores.append(score)
        
        if scores:
            breadth_score.loc[date_idx] = np.mean(scores)
        else:
            breadth_score.loc[date_idx] = np.nan
    
    return breadth_score

# ===== 현금 비율 (연속, breadth 기반) =====
def calculate_cash_fraction_continuous(breadth_score):
    """
    현금 비율 = 1 - breadth_score
    - breadth_score 높음 (강한 시장) → 현금 비율 낮음
    - breadth_score 낮음 (약한 시장) → 현금 비율 높음
    """
    return (1 - breadth_score).clip(0, 1)

# ===== 포트폴리오 가중치 (연속 + 모멘텀 가중) =====
def calculate_portfolio_weights_continuous(momentum_df, cash_fraction, risky_tickers, cash_tickers, top_n):
    """
    연속적 가중치 배분 (모멘텀 가중):
    
    1. 위험자산: 모멘텀 상위 T개를 모멘텀 크기로 가중 배분
       - 모멘텀 높을수록 더 높은 비중
       - 예: 상위 3개의 모멘텀이 [0.1, 0.05, 0.03]이면
         비중 = [0.1/0.18=55%, 0.05/0.18=28%, 0.03/0.18=17%]
    
    2. 현금: 가장 높은 모멘텀을 가진 현금 자산만 선택
    
    3. 가중치 합 = 1.0 검증
    """
    available_risky = [t for t in risky_tickers if t in momentum_df.columns]
    available_cash = [t for t in cash_tickers if t in momentum_df.columns]
    
    weights_risky = pd.DataFrame(0.0, index=momentum_df.index, columns=available_risky)
    weights_cash = pd.DataFrame(0.0, index=momentum_df.index, columns=available_cash)
    
    validation_log = []
    
    for date_idx in momentum_df.index:
        # NaN 처리
        if pd.isna(cash_fraction.loc[date_idx]):
            validation_log.append({
                'date': date_idx,
                'total_weight': np.nan,
                'valid': False,
                'reason': 'NaN cash_fraction'
            })
            continue
        
        cf = float(cash_fraction.loc[date_idx])
        risky_ratio = max(0, 1.0 - cf)
        
        # ===== 위험자산 가중치 (모멘텀 가중) =====
        if risky_ratio > 1e-6:
            risky_mom = momentum_df.loc[date_idx, available_risky]
            risky_mom_valid = risky_mom[~pd.isna(risky_mom)].copy()
            
            if len(risky_mom_valid) > 0:
                # 상위 T개 선택
                if len(risky_mom_valid) >= top_n:
                    top_assets = risky_mom_valid.nlargest(top_n)
                else:
                    top_assets = risky_mom_valid
                
                # 상위 자산들의 모멘텀을 양수로 클립 (음수 모멘텀은 0으로)
                top_mom = top_assets.clip(lower=0)
                
                # 모멘텀이 모두 0 이하인 경우는 동등 가중
                if top_mom.sum() > 1e-6:
                    # 모멘텀으로 정규화하여 가중 배분
                    top_weights = top_mom / top_mom.sum()
                    
                    for ticker, weight in top_weights.items():
                        weights_risky.at[date_idx, ticker] = risky_ratio * weight
                else:
                    # 모멘텀이 모두 음수면 동등 가중
                    equal_weight = risky_ratio / len(top_assets)
                    for ticker in top_assets.index:
                        weights_risky.at[date_idx, ticker] = equal_weight
        
        # ===== 현금 가중치 (모멘텀 최고 자산만) =====
        if cf > 1e-6 and len(available_cash) > 0:
            cash_mom = momentum_df.loc[date_idx, available_cash]
            cash_mom_valid = cash_mom[~pd.isna(cash_mom)]
            
            if len(cash_mom_valid) > 0:
                # 모멘텀이 최고인 현금 자산 선택 (음수도 가능)
                best_cash = cash_mom_valid.idxmax()
                weights_cash.at[date_idx, best_cash] = cf
        
        # 검증
        total_weight = weights_risky.loc[date_idx].sum() + weights_cash.loc[date_idx].sum()
        is_valid = abs(total_weight - 1.0) < 0.001
        
        validation_log.append({
            'date': date_idx,
            'total_weight': total_weight,
            'risky_sum': weights_risky.loc[date_idx].sum(),
            'cash_sum': weights_cash.loc[date_idx].sum(),
            'cash_fraction': cf,
            'valid': is_valid
        })
    
    return weights_risky, weights_cash, pd.DataFrame(validation_log)

# ===== Bad 자산 식별 (신호 표시용) =====
def get_bad_assets(momentum_df, threshold=0.0):
    """Bad 자산 식별: 모멘텀 <= 0인 자산"""
    try:
        return momentum_df <= threshold
    except:
        return pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)

# ===== Breadth Bad Count (신호 표시용) =====
def count_breadth_bad(bad_assets_df, canary_tickers):
    """Breadth Momentum: Canary에서 bad인 자산 개수"""
    try:
        available = [t for t in canary_tickers if t in bad_assets_df.columns]
        if available:
            canary_bad = bad_assets_df[available].sum(axis=1).astype(int)
        else:
            canary_bad = pd.Series(0, index=bad_assets_df.index, dtype=int)
        return canary_bad
    except:
        return pd.Series(0, index=bad_assets_df.index, dtype=int)

# ===== 백테스트 수익률 =====
def backtest_returns(monthly_returns, weights_df, transaction_cost=0.001):
    """백테스트 수익률 계산 (거래 비용 포함)"""
    try:
        common_cols = [col for col in weights_df.columns if col in monthly_returns.columns]
        
        if not common_cols:
            return pd.Series(0.0, index=weights_df.index)
        
        strategy_returns = (weights_df[common_cols] * monthly_returns[common_cols]).sum(axis=1)
        weight_changes = weights_df[common_cols].diff().abs().sum(axis=1)
        trading_costs = weight_changes * transaction_cost
        
        return strategy_returns - trading_costs
    except:
        return pd.Series(0.0, index=weights_df.index)

# ===== 성과 지표 =====
def calculate_performance_metrics(strategy_returns, benchmark_returns=None, risk_free_rate=0.02):
    """성과 지표 계산"""
    try:
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {
                'Total Return (%)': 0.0, 'CAGR (%)': 0.0, 'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0, 'Max Drawdown (%)': 0.0, 'Win Rate (%)': 0.0,
                'RAD (%)': 0.0
            }
        
        cum_strategy = (1 + strategy_returns).cumprod()
        total_return = float((cum_strategy.iloc[-1] - 1) * 100)
        
        num_years = max(len(strategy_returns) / 12, 0.01)
        cagr = float(((cum_strategy.iloc[-1]) ** (1.0 / num_years) - 1) * 100)
        
        volatility = float(strategy_returns.std() * np.sqrt(12) * 100)
        
        excess_return = float(strategy_returns.mean() * 12 - risk_free_rate)
        sharpe_ratio = float(excess_return / (volatility / 100) if volatility > 1e-6 else 0.0)
        
        running_max = cum_strategy.expanding().max()
        drawdown = float(((cum_strategy / running_max) - 1).min() * 100)
        
        win_rate = float((strategy_returns > 0).sum() / len(strategy_returns) * 100)
        
        # RAD 계산
        rad = 0.0
        if cagr >= 0 and drawdown >= -50:
            d_ratio = abs(drawdown) / 100.0
            if d_ratio < 1:
                rad = float(cagr * (1 - (d_ratio / (1 - d_ratio)))) if d_ratio > 0 else cagr
        
        metrics = {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': drawdown,
            'Win Rate (%)': win_rate,
            'RAD (%)': rad
        }
        
        if benchmark_returns is not None:
            try:
                benchmark_returns = benchmark_returns.dropna()
                common_idx = strategy_returns.index.intersection(benchmark_returns.index)
                
                if len(common_idx) > 0:
                    strategy_ret = strategy_returns[common_idx]
                    bench_ret = benchmark_returns[common_idx]
                    
                    cum_benchmark = (1 + bench_ret).cumprod()
                    benchmark_return = float((cum_benchmark.iloc[-1] - 1) * 100)
                    
                    metrics['Benchmark Return (%)'] = benchmark_return
            except:
                pass
        
        return metrics
    except:
        return {
            'Total Return (%)': 0.0, 'CAGR (%)': 0.0, 'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0, 'Max Drawdown (%)': 0.0, 'Win Rate (%)': 0.0,
            'RAD (%)': 0.0
        }

# ===== VAA 전략 실행 =====
def run_vaa_strategy(price_data, risky_list, canary_list, cash_list, top_select, momentum_type):
    """VAA 전략 실행 (정확한 버전)"""
    try:
        available_risky = [t for t in risky_list if t in price_data.columns]
        available_canary = [t for t in canary_list if t in price_data.columns]
        available_cash = [t for t in cash_list if t in price_data.columns]
        
        if not available_risky or not available_canary or not available_cash:
            st.warning("⚠️ 필요한 데이터가 부분적으로 누락되었습니다")
        
        monthly_returns = calculate_monthly_returns(price_data)
        
        # 모멘텀 계산
        if momentum_type == "DAA (1,3,6,12)":
            momentum = calculate_momentum_daa(monthly_returns)
        else:
            momentum = calculate_momentum_vaa(monthly_returns)
        
        bad_assets = get_bad_assets(momentum)
        breadth_bad = count_breadth_bad(bad_assets, available_canary)
        breadth_score = calculate_breadth_score_continuous(momentum, available_canary)
        cash_fraction = calculate_cash_fraction_continuous(breadth_score)
        
        weights_risky, weights_cash, validation = calculate_portfolio_weights_continuous(
            momentum, cash_fraction, available_risky, available_cash, top_select
        )
        
        all_tickers = available_risky + available_cash
        if not all_tickers:
            return None
            
        weights_combined = pd.concat([weights_risky, weights_cash], axis=1)
        monthly_returns_all = monthly_returns[all_tickers]
        strategy_returns = backtest_returns(monthly_returns_all, weights_combined)
        
        return {
            'strategy_returns': strategy_returns,
            'momentum': momentum,
            'bad_assets': bad_assets,
            'cash_fraction': cash_fraction,
            'breadth_score': breadth_score,
            'weights_risky': weights_risky,
            'weights_cash': weights_cash,
            'breadth_bad': breadth_bad,
            'monthly_returns': monthly_returns,
            'validation': validation,
            'available_risky': available_risky,
            'available_cash': available_cash
        }
    except Exception as e:
        st.error(f"❌ VAA 실행 오류: {str(e)}")
        return None

# ===== 메인 =====
def main():
    st.markdown('<div class="header">📊 DAA+VAA 통합 전략 (정확한 백테스트 + 연속 가중치)</div>', unsafe_allow_html=True)
    st.write("**기반**: Keller & Keuning (2016, 2017) DAA & VAA 논문")
    st.write("**개선**: ✅ Look-ahead bias 제거 | ✅ 연속 가중치 배분 | ✅ 모멘텀 가중화 | ✅ 모든 기록")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작", value=datetime(2015, 1, 1))
        with col2:
            end_date = st.date_input("종료", value=datetime.now())
        
        st.divider()
        
        universe_choice = st.radio(
            "📈 자산 유니버스",
            list(UNIVERSES.keys()),
            index=0
        )
        
        momentum_type = st.radio(
            "모멘텀 필터",
            ["DAA (1,3,6,12 균등)", "VAA (13612W)"],
            index=1
        )
        
        st.divider()
        
        benchmark_choices = st.multiselect(
            "🎯 비교 벤치마크",
            list(BENCHMARKS_CONFIG.keys()),
            default=['SPY Index', 'Balanced (60/40)']
        )
    
    # 메인 로직
    try:
        st.info("💾 데이터 로드 중...")
        
        universe = UNIVERSES[universe_choice]
        risky_list = universe['risky']
        canary_list = universe['canary']
        cash_list = universe['cash']
        top_select = universe['t_param']
        breadth_param = universe['b_param']
        
        all_tickers = sorted(list(set(risky_list + canary_list + cash_list)))
        
        # 벤치마크 티커 추가
        bench_tickers = ['SPY', 'BND', 'ACWI']
        all_tickers_with_bench = sorted(list(set(all_tickers + bench_tickers)))
        
        price_data = download_price_data(
            all_tickers_with_bench,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if price_data is None or price_data.empty:
            st.error("❌ 데이터 로드 실패")
            return
        
        st.success("✅ 데이터 로드 완료")
        
        # VAA 실행
        vaa_result = run_vaa_strategy(
            price_data, risky_list, canary_list, cash_list, top_select, momentum_type
        )
        
        if vaa_result is None:
            st.error("❌ VAA 실행 실패")
            return
        
        strategy_returns = vaa_result['strategy_returns']
        
        # 벤치마크 계산
        benchmark_dict = {}
        for bench_name in benchmark_choices:
            try:
                bench_config = BENCHMARKS_CONFIG[bench_name]
                
                if bench_config.get('is_acwi', False):
                    if 'ACWI' in price_data.columns:
                        bench_ret = price_data['ACWI'].resample('M').last().pct_change()
                        benchmark_dict[bench_name] = bench_ret
                else:
                    spy_monthly = price_data['SPY'].resample('M').last().pct_change()
                    bnd_monthly = price_data['BND'].resample('M').last().pct_change()
                    bench_ret = (spy_monthly * bench_config['stocks'] + 
                                bnd_monthly * bench_config['bonds'])
                    benchmark_dict[bench_name] = bench_ret
            except:
                pass
        
        # 성과 계산
        performance_vaa = calculate_performance_metrics(strategy_returns)
        performance_bench = {}
        for bench_name, bench_ret in benchmark_dict.items():
            performance_bench[bench_name] = calculate_performance_metrics(bench_ret)
        
        # ===== 성과 지표 표시 =====
        st.markdown("---")
        st.subheader("📈 VAA 성과 지표")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR (%)", f"{performance_vaa['CAGR (%)']:.2f}%", delta="목표: >10%")
        col2.metric("Volatility (%)", f"{performance_vaa['Volatility (%)']:.2f}%")
        col3.metric("Sharpe Ratio", f"{performance_vaa['Sharpe Ratio']:.3f}")
        col4.metric("Max Drawdown (%)", f"{performance_vaa['Max Drawdown (%)']:.2f}%", delta="목표: <15%")
        
        col5, col6, col7 = st.columns(3)
        col5.metric("RAD (%)", f"{performance_vaa['RAD (%)']:.2f}%")
        col6.metric("승률 (%)", f"{performance_vaa['Win Rate (%)']:.1f}%")
        col7.metric("총 수익 (%)", f"{performance_vaa['Total Return (%)']:.2f}%")
        
        # 검증 정보
        st.markdown("---")
        st.subheader("✅ 정확성 검증")
        
        validation = vaa_result['validation']
        valid_count = validation['valid'].sum() if 'valid' in validation.columns else 0
        validity_pct = (valid_count / len(validation) * 100) if len(validation) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**가중치 검증**")
            if validity_pct < 95:
                st.error(f"⚠️ {100-validity_pct:.1f}% 비정상")
            else:
                st.success(f"✅ {validity_pct:.1f}% 유효")
        
        with col2:
            st.markdown("**Look-ahead Bias**")
            st.success("✅ shift(1) 적용")
            st.caption("Month t 가중치 = t-1 모멘텀")
        
        with col3:
            st.markdown("**가중치 방식**")
            st.success("✅ 연속 배분")
            st.caption("모멘텀으로 가중")
        
        with col4:
            st.markdown("**거래 비용**")
            st.info(f"0.1% 편도")
        
        # ===== 탭 =====
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 성과",
            "🔔 신호",
            "⚖️ 가중치",
            "📈 수익률",
            "🔍 상세",
            "📖 용어"
        ])
        
        # 탭 1: 성과
        with tab1:
            st.subheader("📈 VAA 성과 지표")
            
            st.markdown("""
            **주요 성과 지표 설명:**
            - **CAGR**: 연평균 복리 수익률 (목표: >10%)
            - **Volatility**: 변동성 (목표: 낮을수록 좋음)
            - **Sharpe Ratio**: 위험당 수익 (목표: >0.5)
            - **Max Drawdown**: 최악의 낙폭 (목표: <15%, VAA 강점)
            - **RAD**: 낙폭으로 조정한 수익
            - **Win Rate**: 양수 월간 수익 비율
            """)
            
            st.divider()
            
            # 누적 수익률 차트
            st.subheader("📊 누적 수익률 비교")
            
            try:
                cum_returns = (1 + strategy_returns).cumprod() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    mode='lines',
                    name='VAA',
                    line=dict(color=COLORS['vaa'], width=3)
                ))
                
                for bench_name, bench_ret in benchmark_dict.items():
                    cum_bench = (1 + bench_ret).cumprod() * 100
                    fig.add_trace(go.Scatter(
                        x=cum_bench.index,
                        y=cum_bench.values,
                        mode='lines',
                        name=bench_name,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="누적 수익률 비교",
                    xaxis_title="날짜",
                    yaxis_title="누적 수익률 (%)",
                    height=400,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 성과 비교 표
                st.subheader("📋 벤치마크 비교")
                
                comparison_data = {
                    'Strategy': ['VAA'] + list(benchmark_dict.keys())
                }
                
                metrics_to_show = ['CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'RAD (%)']
                for metric in metrics_to_show:
                    comparison_data[metric] = [f"{performance_vaa[metric]:.2f}"]
                    for bench_name in benchmark_dict.keys():
                        val = performance_bench[bench_name].get(metric, 0)
                        comparison_data[metric].append(f"{val:.2f}")
                
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True)
                
            except Exception as e:
                st.warning(f"⚠️ 차트 오류: {str(e)}")
        
        # 탭 2: 신호
        with tab2:
            st.subheader("🔔 VAA 신호 분석 (Breadth Momentum)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**VWO 모멘텀** (카나리 자산)")
                try:
                    momentum = vaa_result['momentum']
                    if 'VWO' in momentum.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=momentum.index,
                            y=momentum['VWO'],
                            mode='lines',
                            name='VWO',
                            fill='tozeroy',
                            line=dict(color='#1f77b4')
                        ))
                        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Bad Asset Threshold")
                        fig.update_layout(title="VWO 모멘텀", height=300, hovermode='x')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("VWO 데이터 없음")
                except:
                    st.info("VWO 데이터 없음")
            
            with col2:
                st.markdown("**BND 모멘텀** (카나리 자산)")
                try:
                    momentum = vaa_result['momentum']
                    if 'BND' in momentum.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=momentum.index,
                            y=momentum['BND'],
                            mode='lines',
                            name='BND',
                            fill='tozeroy',
                            line=dict(color='#ff7f0e')
                        ))
                        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Bad Asset Threshold")
                        fig.update_layout(title="BND 모멘텀", height=300, hovermode='x')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("BND 데이터 없음")
                except:
                    st.info("BND 데이터 없음")
            
            st.divider()
            st.markdown("**현금 비율 (CF) - Breadth Score 기반 (연속)**")
            try:
                cash_fraction = vaa_result['cash_fraction']
                breadth_score = vaa_result['breadth_score']
                breadth_bad = vaa_result['breadth_bad']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cash_fraction.index,
                    y=cash_fraction * 100,
                    mode='lines',
                    name='Cash Fraction',
                    fill='tozeroy',
                    line=dict(color=COLORS['neutral'], width=2)
                ))
                
                # 보조 축으로 breadth_score 표시
                fig.add_trace(go.Scatter(
                    x=breadth_score.index,
                    y=breadth_score,
                    mode='lines',
                    name='Breadth Score',
                    yaxis='y2',
                    line=dict(color='#3498db', width=2)
                ))
                
                fig.update_layout(
                    title="현금 비율 vs Breadth Score (연속)",
                    yaxis=dict(title="Cash Fraction (%)"),
                    yaxis2=dict(title="Breadth Score (0~1)", overlaying='y', side='right'),
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                **해석:**
                - **Breadth Score**: 카나리 자산(VWO, BND)의 모멘텀 강도 (0~1)
                - **현금 비율**: CF = 1 - Breadth Score
                - **의미**:
                  - Score 높음 (1.0) → CF 낮음 (0%) → 공격적 투자
                  - Score 낮음 (0.0) → CF 높음 (100%) → 완전 현금
                - **평균 현금 비율**: {cash_fraction.mean()*100:.1f}%
                """)
                
            except Exception as e:
                st.warning(f"⚠️ 신호 분석 오류")
        
        # 탭 3: 가중치
        with tab3:
            st.subheader("⚖️ 포트폴리오 구성 (연속 가중치)")
            
            st.markdown("""
            **연속 가중치 배분 방식:**
            
            위험자산 선택:
            1. 모멘텀 상위 T개 자산 선택
            2. 각 자산의 모멘텀을 이용하여 비중 결정
            3. 높은 모멘텀 → 높은 비중 (가중 평균)
            
            예: 상위 3개 모멘텀이 [0.10, 0.06, 0.04]인 경우
            - 합계: 0.20
            - 비중: [50%, 30%, 20%]
            """)
            
            try:
                weights_risky = vaa_result['weights_risky']
                weights_cash = vaa_result['weights_cash']
                
                weights = pd.concat([weights_risky, weights_cash], axis=1) * 100
                
                # 최근 20개월 가중치
                recent = weights.tail(20)
                
                fig = go.Figure()
                for col in recent.columns:
                    fig.add_trace(go.Bar(
                        x=recent.index,
                        y=recent[col],
                        name=col
                    ))
                
                fig.update_layout(
                    title="월별 포트폴리오 가중치 (최근 20개월)",
                    barmode='stack',
                    yaxis_title="비중 (%)",
                    height=400,
                    hovermode='x'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 자산별 평균 가중치
                st.markdown("**평균 포트폴리오 구성**")
                avg_weights = weights.mean().sort_values(ascending=False)
                avg_weights = avg_weights[avg_weights > 0.1]  # 0.1% 이상만 표시
                
                fig = go.Figure(data=[go.Pie(
                    labels=avg_weights.index,
                    values=avg_weights.values
                )])
                fig.update_layout(title="자산별 평균 비중", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"⚠️ 가중치 분석 오류: {str(e)}")
        
        # 탭 4: 수익률
        with tab4:
            st.subheader("📈 월별 수익률")
            
            try:
                fig = go.Figure()
                colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in strategy_returns]
                
                fig.add_trace(go.Bar(
                    x=strategy_returns.index,
                    y=strategy_returns * 100,
                    marker=dict(color=colors),
                    name='Monthly Return'
                ))
                
                fig.update_layout(
                    title="월별 수익률",
                    yaxis_title="수익률 (%)",
                    height=400,
                    hovermode='x'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("평균 월간 수익률", f"{strategy_returns.mean()*100:.2f}%")
                col2.metric("긍정 월간 비율", f"{(strategy_returns > 0).sum() / len(strategy_returns) * 100:.1f}%")
                col3.metric("최악의 달", f"{strategy_returns.min()*100:.2f}%")
                col4.metric("최고의 달", f"{strategy_returns.max()*100:.2f}%")
                
            except Exception as e:
                st.warning(f"⚠️ 수익률 분석 오류")
        
        # 탭 5: 상세 정보
        with tab5:
            st.subheader("🔍 전략 상세 정보")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 선택된 유니버스")
                st.markdown(f"**{universe_choice}**")
                st.write(universe['description'])
                
                st.markdown("### 🎯 파라미터")
                st.markdown(f"""
                - **T (Top Selection)**: {top_select}개 자산
                - **B (Breadth Parameter)**: {breadth_param}
                - **모멘텀 필터**: {momentum_type}
                - **리밸런싱**: 월별 (EOM)
                - **거래 비용**: 0.1% (편도)
                """)
            
            with col2:
                st.markdown("### 💼 자산 구성")
                st.markdown(f"""
                **위험자산 ({len(risky_list)}개):**
                {', '.join(risky_list)}
                
                **카나리 자산:**
                {', '.join(canary_list)}
                
                **현금 자산:**
                {', '.join(cash_list)}
                """)
            
            st.divider()
            
            st.markdown("### 📈 계산 수식")
            
            with st.expander("1️⃣ 모멘텀 필터"):
                if "DAA" in momentum_type:
                    st.markdown("""
                    $$Momentum_{DAA} = \\frac{R_1 + R_3 + R_6 + R_{12}}{4}$$
                    
                    - 균등 가중: 각 기간 25% 가중
                    - 목적: 단기~중기 추세 포착
                    """)
                else:
                    st.markdown("""
                    $$Momentum_{VAA} = \\frac{12 \\times R_1 + 4 \\times R_3 + 2 \\times R_6 + 1 \\times R_{12}}{19}$$
                    
                    - 가중치: [12, 4, 2, 1] (합 19)
                    - 1개월: 40% 가중 (빠른 반응)
                    - 목적: 빠른 시장 변화 감지
                    """)
            
            with st.expander("2️⃣ Breadth Score (연속)"):
                st.markdown("""
                $$BreadthScore = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{tanh(Mom_i \\times 3) + 1}{2}$$
                
                여기서:
                - N = 카나리 자산 개수
                - Mom_i = 각 카나리 자산의 모멘텀
                - tanh 함수: [-1, 1] 범위로 정규화
                
                **결과 범위**: [0, 1]
                - 1.0 = 모든 카나리 모멘텀 양수 (강한 시장)
                - 0.5 = 중립 (보통 시장)
                - 0.0 = 모든 카나리 모멘텀 음수 (약한 시장)
                """)
            
            with st.expander("3️⃣ 현금 비율 (연속)"):
                st.markdown("""
                $$CF = 1 - BreadthScore$$
                
                특징:
                - Breadth 높음 (1.0) → CF 낮음 (0%) → 공격적
                - Breadth 낮음 (0.0) → CF 높음 (100%) → 방어적
                - **완전 연속**: 0%~100% 사이의 모든 값 가능
                
                vs VAA 원본 (이산):
                - 원본: b=0→0%, b=1→50%, b≥2→100%
                - 개선: breadth 강도에 따라 연속 조절
                """)
            
            with st.expander("4️⃣ 포트폴리오 가중치 (연속 + 모멘텀)"):
                st.markdown(f"""
                **위험자산 선택:**
                1. 모멘텀 상위 {top_select}개 자산 선택
                2. 각 자산의 모멘텀 크기로 비중 결정
                
                $$w_{{risky,i}} = (1-CF) \\times \\frac{{\\max(Mom_i, 0)}}{{\\sum_j \\max(Mom_j, 0)}}$$
                
                예: 상위 3개 모멘텀이 [0.10, 0.06, 0.04]
                - 합: 0.20
                - 비중: [50%, 30%, 20%]
                
                **현금 자산 선택:**
                - 모멘텀이 최고인 현금 자산만 선택
                $$w_{{cash,best}} = CF$$
                
                **의의**: 모멘텀 강도를 정확히 반영
                """)
            
            with st.expander("5️⃣ RAD (Returns Adjusted for Drawdowns)"):
                st.markdown("""
                $$RAD = R \\times \\left(1 - \\frac{D}{1-D}\\right) \\quad \\text{if } R \\geq 0\\% \\text{ and } D \\leq 50\\%$$
                
                여기서:
                - R = CAGR (연평균 수익률)
                - D = 최대 낙폭 (절대값)
                
                **의의**:
                - 수익률과 위험을 동시에 고려
                - VAA 원본 논문의 최적화 기준
                - D ≥ 50%면 RAD = 0 (매우 위험)
                """)
        
        # 탭 6: 용어 설명
        with tab6:
            st.subheader("📖 VAA 전략 용어 및 개념")
            
            tabs_glossary = st.tabs([
                "Breadth Momentum",
                "Dual Momentum",
                "Canary Universe",
                "Cash Universe",
                "벤치마크 설명",
                "성과 지표",
                "기타 용어"
            ])
            
            with tabs_glossary[0]:
                st.markdown("""
                ## 🔍 Breadth Momentum (광폭 모멘텀)
                
                ### 정의
                시장에서 **상승하는 자산의 비중**을 기반으로 시장 상태를 판단하는 기법
                
                ### VAA에서의 적용
                - **전통 방식**: 각 자산의 모멘텀 개별 확인 (asset-level absolute momentum)
                - **VAA 방식**: 카나리 유니버스의 모멘텀 강도로 판단 (universe-level breadth momentum)
                
                ### 이 구현의 차이점
                **원본 VAA (이산)**:
                ```
                Bad assets 개수(b):
                  b < B → CF = b/B
                  b ≥ B → CF = 100%
                
                예 (B=2): b=0→0%, b=1→50%, b≥2→100%
                ```
                
                **개선 버전 (연속)**:
                ```
                Breadth Score (0~1):
                  Breadth 높음 → Score 높음 → CF 낮음
                  Breadth 낮음 → Score 낮음 → CF 높음
                
                예: Score=0.3→CF=70%, Score=0.8→CF=20%
                → 훨씬 부드러운 전환
                ```
                
                ### 장점
                - **빠른 반응**: 시장 강도를 즉시 반영
                - **강한 보호**: 시장 약세 초기에 빠르게 현금 진입
                - **연속성**: 급격한 비중 변화 없음
                """)
            
            with tabs_glossary[1]:
                st.markdown("""
                ## ⚡ Dual Momentum (이중 모멘텀)
                
                ### 정의
                **절대 모멘텀 (Absolute)** + **상대 모멘텀 (Relative)** 결합
                
                ### 절대 모멘텀 (Absolute/Time-series Momentum)
                개별 자산이 상승 추세인지 판단 (모멘텀 > 0?)
                ```
                목적: Crash 방지
                방법: 모멘텀 ≤ 0 자산은 현금으로 대체
                
                예: 자산 A의 모멘텀 = -0.05 → 위험자산 목록에서 제외
                ```
                
                ### 상대 모멘텀 (Relative/Cross-sectional Momentum)
                여러 자산 중 가장 강한 자산 선택
                ```
                목적: 수익 최대화
                방법: 모멘텀 상위 T개만 보유
                
                예: SPY(0.10) > VEA(0.08) > VWO(0.05) → SPY, VEA 선택
                ```
                
                ### VAA의 이중 모멘텀
                - **절대**: Breadth momentum → 현금 비율 결정
                - **상대**: 상위 자산 선택 → 위험자산 배분
                
                **차이점**:
                - 원본: 각 자산 모멘텀 개별 확인
                - 개선: 연속 모멘텀 가중치 + Breadth score
                """)
            
            with tabs_glossary[2]:
                st.markdown("""
                ## 🕯️ Canary Universe (카나리 유니버스)
                
                ### 개념
                "탄광 카나리"처럼 시장 위험을 조기에 감지하는 자산군
                
                ### VAA의 카나리
                - **VWO** (신흥시장 주식 ETF)
                - **BND** (채권 ETF)
                
                ### 선택 이유
                1. **VWO**: 위험자산의 대표
                   - 경기 약세 시 먼저 하락
                   - 신흥시장 취약성 반영
                
                2. **BND**: 안전자산의 대표
                   - 금리 상승 시 하락
                   - 인플레이션/유동성 우려 반영
                
                ### 작동 원리
                ```
                Breadth Score = (VWO의 모멘텀 강도 + BND의 모멘텀 강도) / 2
                
                VWO 약세 → 신흥시장 약세 → 경기 둔화 신호
                BND 약세 → 금리 상승 또는 유동성 위기 신호
                
                둘 중 하나라도 약세 → Breadth 낮음 → 현금 ↑
                ```
                
                ### 효과
                주요 위험자산 보유 전에 선제적으로 현금 전환
                
                **경험적 증거** (Keller & Keuning 2017):
                - VWO/BND의 약세 = 대량 손실 3~6개월 전 경고
                - 차익거래 난제 해결: 수익성 높음에도 변동성 낮음
                """)
            
            with tabs_glossary[3]:
                st.markdown("""
                ## 💰 Cash Universe (현금 유니버스)
                
                ### 정의
                시장에서 벗어날 때 투자하는 안전자산들
                
                ### VAA의 현금 자산
                - **SHY** (1-3년 만기 US Treasury, iShares)
                - **IEF** (7-10년 만기 US Treasury, iShares)
                - **LQD** (투자등급 회사채, iShares)
                
                ### 선택 방식
                ```
                매월 이 3가지 중 모멘텀이 가장 높은 1개만 선택
                
                예: LQD 모멘텀(0.05) > IEF(0.02) > SHY(0.01)
                → LQD 100% 보유 (CF가 60%라면 LQD 60%)
                ```
                
                ### 중요성
                - **현금 비율이 높을 때** (평균 60%) 중요
                - **수익성**: 단순 현금(T-bill) 0%보다 우수
                - **절대 모멘텀 미적용**: 모멘텀 부호 무관하게 선택
                
                ### 특징
                - SHY: 단기 채권 (변동성 최소)
                - IEF: 중기 채권 (수익성 균형)
                - LQD: 회사채 (수익성 최대)
                """)
            
            with tabs_glossary[4]:
                st.markdown(f"""
                ## 📊 벤치마크 설명
                
                ### Aggressive (70/30) 포트폴리오
                ```
                구성: SPY 70% + BND 30%
                특징: 공격적, 주식 비중 높음
                용도: 장기 고성장 목표 투자자
                수익성: 높음, 변동성: 높음
                ```
                
                ### Balanced (60/40) 포트폴리오
                ```
                구성: SPY 60% + BND 40%
                특징: 균형잡힌 전통적 포트폴리오
                용도: 일반적인 장기 투자자
                수익성: 중간, 변동성: 중간
                ```
                
                ### Conservative (50/50) 포트폴리오
                ```
                구성: SPY 50% + BND 50%
                특징: 보수적, 채권 비중 높음
                용도: 위험 회피형 투자자
                수익성: 낮음, 변동성: 낮음
                ```
                
                ### SPY Index (S&P 500)
                ```
                구성: 미국 대형주 500개
                특징: 미국 주식시장의 대표
                용도: 글로벌 성장 추적
                수익성: 높음, 변동성: 높음
                ```
                
                ### ACWI Index (All Country World Index)
                ```
                구성: 전 세계 주식 3,000+ 개
                특징: 글로벌 분산 포트폴리오
                용도: 세계 경제 성장 추적
                수익성: 중간, 변동성: 중간
                ```
                
                ### VAA와의 비교 특징
                - **VAA**: 동적 현금 조절로 crash protection
                - **정적 벤치마크**: 고정 비중, 위험 노출 일정
                - **결과**: VAA가 약세장에서 우수, 강세장에서는 추종
                """)
            
            with tabs_glossary[5]:
                st.markdown("""
                ## 📊 VAA 성과 지표
                
                ### CAGR (연평균 수익률)
                ```
                복리 연간 수익률 = (최종가 / 초기가)^(1/년수) - 1
                ```
                **목표**: > 10%
                **해석**: 연간 평균 몇 %씩 자산이 증가했는가
                
                ### Volatility (변동성)
                ```
                연간 표준편차 = 월간 수익률의 표준편차 × √12
                ```
                **해석**: 수익률 변동의 크기 (작을수록 안정적)
                
                ### Sharpe Ratio (샤프 비율)
                ```
                Sharpe = (연간 초과수익) / (변동성)
                ```
                **의의**: 위험 단위당 수익 → 높을수록 좋음
                **해석**: 같은 위험으로 더 많은 수익을 얻는가
                
                ### Max Drawdown (최대 낙폭)
                ```
                최악의 경우: 최고 이후 최저 낙폭
                ```
                **목표**: < 15% (VAA 강점)
                **해석**: 최악의 시기에 얼마나 내려갔는가
                
                ### RAD (Returns Adjusted for Drawdowns)
                ```
                RAD = R × (1 - D/(1-D)) if R≥0% and D≤50%
                ```
                **의의**: 수익을 위험으로 조정
                **해석**: 낙폭을 고려한 조정된 수익률
                
                ### Win Rate (승률)
                ```
                양의 수익률을 낸 월수 / 전체 월수 × 100%
                ```
                **해석**: 몇 %의 기간에 수익을 냈는가
                """)
            
            with tabs_glossary[6]:
                st.markdown(f"""
                ## 기타 용어
                
                ### 모멘텀 필터
                **DAA (균등 가중)**:
                - (R1 + R3 + R6 + R12) / 4
                - 각 기간 25% 가중
                
                **VAA (13612W)**:
                - (12×R1 + 4×R3 + 2×R6 + 1×R12) / 19
                - 최근 (1개월) 40% 가중 → 빠른 반응
                
                ### T (Top Selection)
                위험자산 중 모멘텀 상위 T개만 선택
                - VAA-G12: T={top_select} (상위 {top_select}개)
                - VAA-G4: T=1 (상위 1개)
                
                ### B (Breadth Parameter)
                원본 VAA에서 사용 (이 구현은 연속 Breadth Score 사용)
                - 몇 개의 bad assets가 있을 때 100% 현금인가?
                
                ### CF (Cash Fraction)
                현재 현금 보유 비중
                - 낮을수록 공격적 (0%~100%)
                - 이 구현: 연속값 (부드러운 전환)
                
                ### Look-ahead Bias 제거
                ```
                Month t의 가중치 결정 = Month t-1의 모멘텀
                → shift(1) 적용
                
                실제 활용:
                월말에 모멘텀 확인 → 다음달에 거래
                현실적이고 정확한 백테스트
                ```
                
                ### 논문 및 참고
                **Keller & Keuning (2016)**
                "Protective Asset Allocation (PAA)"
                SSRN: https://ssrn.com/abstract=3212862
                
                **Keller & Keuning (2017)**
                "Breadth Momentum and Vigilant Asset Allocation (VAA);
                Winning More by Losing Less"
                SSRN: https://ssrn.com/abstract=3002624
                """)
    
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
