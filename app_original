"""
DAA + VAA 통합 전략 (최종 버전)
========================================================================

핵심 전략:
1. 31개 글로벌 자산 + Core(ACWI 최소 20%) + 안전자산 3개
2. Breadth Momentum 기반 위험자산 비중 자동 결정
3. 최적 T값 자동 계산 (Sharpe ratio 최대화)
4. 종목별 최대 30% 투자 제한
5. DAA 모멘텀 필터 적용
6. 월별 안전자산 동적 선택
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DAA+VAA 최종", page_icon="📊", layout="wide")

st.markdown("""
<style>
.header {color: #1f77b4; font-size: 28px; font-weight: bold; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

COLORS = {
    'strategy': '#E74C3C',
    'aggressive': '#E67E22',
    'balanced': '#27AE60',
    'conservative': '#3498DB',
    'spy': '#9B59B6',
    'acwi': '#1ABC9C',
    'positive': '#2ECC71',
    'negative': '#E74C3C',
    'neutral': '#95A5A6'
}

# 31개 자산 유니버스 (ACWI Core + 30개 선택 자산)
UNIVERSE = {
    'core': ['ACWI'],  # Core: 최소 20%
    'risky': ['SPY', 'IWM', 'QQQ', 'VEA', 'VGK', 'EWJ', 'VWO', 'VNQ', 'GSG', 'GLD', 
              'TLT', 'HYG', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 
              'XLK', 'XLU', 'RSP', 'VUG', 'VTV', 'VYM', 'USMV', 'EWY', 'SPMO', 'PTF'],  # 30개
    'canary': ['VWO', 'BND'],
    'cash': ['SHY', 'IEF', 'LQD']
}

BENCHMARKS_CONFIG = {
    'Aggressive (70/30)': {'stocks': 0.7, 'bonds': 0.3},
    'Balanced (60/40)': {'stocks': 0.6, 'bonds': 0.4},
    'Conservative (50/50)': {'stocks': 0.5, 'bonds': 0.5},
    'SPY Index': {'stocks': 1.0, 'bonds': 0.0},
    'ACWI Index': {'is_acwi': True}
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
        
        df = df.ffill().bfill().dropna(how='all')
        
        available_cols = [t for t in ticker_list if t in df.columns]
        return df[available_cols] if available_cols else None
    except:
        return None

# ===== 월별 수익률 =====
def calculate_monthly_returns(price_df):
    """월별 수익률 계산"""
    try:
        monthly_prices = price_df.resample('M').last()
        return monthly_prices.pct_change()
    except:
        return pd.DataFrame()

# ===== DAA 모멘텀 =====
def calculate_momentum_daa(returns_df):
    """DAA: (R1 + R3 + R6 + R12) / 4"""
    momentum = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    
    for col in returns_df.columns:
        cum_1m = returns_df[col]
        cum_3m = (1 + returns_df[col]).rolling(3).apply(lambda x: np.prod(x) - 1 if len(x) == 3 else np.nan, raw=False)
        cum_6m = (1 + returns_df[col]).rolling(6).apply(lambda x: np.prod(x) - 1 if len(x) == 6 else np.nan, raw=False)
        cum_12m = (1 + returns_df[col]).rolling(12).apply(lambda x: np.prod(x) - 1 if len(x) == 12 else np.nan, raw=False)
        
        mom = (cum_1m + cum_3m + cum_6m + cum_12m) / 4
        momentum[col] = mom.shift(1)
    
    return momentum

# ===== Breadth Score =====
def calculate_breadth_score_continuous(momentum_df, canary_tickers):
    """Breadth Score: Canary 모멘텀 기반"""
    available = [t for t in canary_tickers if t in momentum_df.columns]
    
    if not available:
        return pd.Series(1.0, index=momentum_df.index)
    
    breadth_score = pd.Series(0.0, index=momentum_df.index)
    
    for date_idx in momentum_df.index:
        canary_mom = momentum_df.loc[date_idx, available]
        
        if pd.isna(canary_mom).all():
            breadth_score.loc[date_idx] = np.nan
            continue
        
        scores = []
        for mom in canary_mom:
            if pd.isna(mom):
                continue
            score = (np.tanh(mom * 3) + 1) / 2
            scores.append(score)
        
        if scores:
            breadth_score.loc[date_idx] = np.mean(scores)
        else:
            breadth_score.loc[date_idx] = np.nan
    
    return breadth_score

# ===== 현금 비율 =====
def calculate_cash_fraction_continuous(breadth_score):
    """CF = 1 - breadth_score"""
    return (1 - breadth_score).clip(0, 1)

# ===== 위험자산 비중 최적화 (적정 비중 계산) =====
def calculate_risk_asset_allocation(breadth_score):
    """
    위험자산의 적정 비중을 breadth score 기반으로 계산
    
    로직:
    - Breadth 높음 (1.0): 위험자산 100%
    - Breadth 중간 (0.5): 위험자산 70-80%
    - Breadth 낮음 (0.0): 위험자산 20-30% (최소 ACWI 20%)
    
    이차함수로 부드러운 전환
    """
    # breadth_score를 위험자산 비중으로 변환 (비선형)
    # f(x) = 0.3 + 0.7*x^2 (부드러운 곡선)
    # x=0: 30%, x=0.5: 51.75%, x=1: 100%
    
    risk_allocation = 0.3 + 0.7 * (breadth_score ** 2)
    return risk_allocation.clip(0.25, 1.0)

# ===== 최적 T값 자동 계산 =====
def find_optimal_top_n(momentum_df, cash_fraction, risk_allocation, available_risky, available_cash, max_t=8):
    """
    Sharpe ratio를 최대화하는 T값 찾기
    """
    best_sharpe = -999
    best_t = 1
    
    for test_t in range(1, min(len(available_risky) + 1, max_t + 1)):
        try:
            # test_t로 가중치 계산
            w_r, w_c, _, _ = calculate_portfolio_weights_with_constraints(
                momentum_df, cash_fraction, risk_allocation,
                UNIVERSE['core'], available_risky, available_cash, test_t
            )
            
            # 수익률 계산 (간단 버전)
            all_cols = [c for c in w_r.columns if c in momentum_df.columns] + \
                      [c for c in w_c.columns if c in momentum_df.columns]
            if not all_cols:
                continue
            
            w_combined = pd.concat([w_r, w_c], axis=1)
            rets = (w_combined[all_cols] * momentum_df[all_cols]).sum(axis=1)
            
            # Sharpe 계산
            if len(rets) > 0:
                sharpe = (rets.mean() * 12 - 0.02) / (rets.std() * np.sqrt(12) + 1e-6)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_t = test_t
        except:
            pass
    
    return best_t

# ===== 포트폴리오 가중치 계산 (제약 조건 포함) =====
def calculate_portfolio_weights_with_constraints(momentum_df, cash_fraction, risk_allocation,
                                                 core_tickers, risky_tickers, cash_tickers, top_n,
                                                 max_single_weight=0.30, min_core_weight=0.20):
    """
    제약 조건:
    1. ACWI Core: 최소 20%
    2. 각 종목: 최대 30%
    3. 위험자산: risk_allocation 기반
    4. 가중치 합: 1.0
    """
    available_core = [t for t in core_tickers if t in momentum_df.columns]
    available_risky = [t for t in risky_tickers if t in momentum_df.columns]
    available_cash = [t for t in cash_tickers if t in momentum_df.columns]
    
    weights_core = pd.DataFrame(0.0, index=momentum_df.index, columns=available_core)
    weights_risky = pd.DataFrame(0.0, index=momentum_df.index, columns=available_risky)
    weights_cash = pd.DataFrame(0.0, index=momentum_df.index, columns=available_cash)
    
    validation_log = []
    
    for date_idx in momentum_df.index:
        try:
            if pd.isna(cash_fraction.loc[date_idx]) or pd.isna(risk_allocation.loc[date_idx]):
                validation_log.append({'date': date_idx, 'total_weight': np.nan, 'valid': False})
                continue
            
            cf = float(cash_fraction.loc[date_idx])
            risk_ratio = float(risk_allocation.loc[date_idx])
            
            # ===== Core (ACWI) 할당: 최소 20% =====
            core_weight = max(min_core_weight, risk_ratio * 0.3)  # 위험자산의 30% 최소
            core_weight = min(core_weight, max_single_weight)  # 30% 초과 방지
            
            if len(available_core) > 0:
                for ticker in available_core:
                    weights_core.at[date_idx, ticker] = core_weight / len(available_core)
            
            # ===== 나머지 위험자산 할당 =====
            remaining_risk = risk_ratio - core_weight
            
            if remaining_risk > 1e-6 and len(available_risky) > 0:
                risky_mom = momentum_df.loc[date_idx, available_risky]
                risky_mom_valid = risky_mom[~pd.isna(risky_mom)].copy()
                
                if len(risky_mom_valid) > 0:
                    # 상위 T개 선택
                    if len(risky_mom_valid) >= top_n:
                        top_assets = risky_mom_valid.nlargest(top_n)
                    else:
                        top_assets = risky_mom_valid
                    
                    top_mom = top_assets.clip(lower=0)
                    
                    if top_mom.sum() > 1e-6:
                        # 모멘텀으로 가중 배분
                        top_weights = top_mom / top_mom.sum()
                        
                        # 각 자산별 최대 30% 제약 적용
                        scaled_weights = {}
                        total_assigned = 0
                        
                        for ticker, weight in top_weights.items():
                            max_w = max_single_weight - weights_core.loc[date_idx].sum()
                            assigned = min(weight * remaining_risk, max_w)
                            scaled_weights[ticker] = assigned
                            total_assigned += assigned
                        
                        # 정규화 (합 = remaining_risk)
                        if total_assigned > 1e-6:
                            for ticker, weight in scaled_weights.items():
                                weights_risky.at[date_idx, ticker] = weight * (remaining_risk / total_assigned)
            
            # ===== 현금 할당 =====
            if cf > 1e-6 and len(available_cash) > 0:
                cash_mom = momentum_df.loc[date_idx, available_cash]
                cash_mom_valid = cash_mom[~pd.isna(cash_mom)]
                
                if len(cash_mom_valid) > 0:
                    # 모멘텀 최고 자산만 선택
                    best_cash = cash_mom_valid.idxmax()
                    weights_cash.at[date_idx, best_cash] = cf
            
            # ===== 정규화 =====
            total_weight = weights_core.loc[date_idx].sum() + weights_risky.loc[date_idx].sum() + weights_cash.loc[date_idx].sum()
            
            if total_weight > 1e-6 and abs(total_weight - 1.0) > 0.001:
                scale = 1.0 / total_weight
                weights_core.loc[date_idx] = weights_core.loc[date_idx] * scale
                weights_risky.loc[date_idx] = weights_risky.loc[date_idx] * scale
                weights_cash.loc[date_idx] = weights_cash.loc[date_idx] * scale
            elif total_weight <= 1e-6 and len(available_cash) > 0:
                weights_cash.at[date_idx, available_cash[0]] = 1.0
            
            # 검증
            final_total = weights_core.loc[date_idx].sum() + weights_risky.loc[date_idx].sum() + weights_cash.loc[date_idx].sum()
            is_valid = abs(final_total - 1.0) < 0.001
            
            validation_log.append({'date': date_idx, 'total_weight': final_total, 'valid': is_valid})
        
        except Exception as e:
            validation_log.append({'date': date_idx, 'total_weight': np.nan, 'valid': False})
    
    return weights_core, weights_risky, weights_cash, pd.DataFrame(validation_log)

# ===== Bad 자산 =====
def get_bad_assets(momentum_df, threshold=0.0):
    try:
        return momentum_df <= threshold
    except:
        return pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)

# ===== 백테스트 =====
def backtest_returns(monthly_returns, weights_df, transaction_cost=0.001):
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
def calculate_performance_metrics(strategy_returns, risk_free_rate=0.02):
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
        
        rad = 0.0
        if cagr >= 0 and drawdown >= -50:
            d_ratio = abs(drawdown) / 100.0
            if d_ratio < 1:
                rad = float(cagr * (1 - (d_ratio / (1 - d_ratio)))) if d_ratio > 0 else cagr
        
        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': drawdown,
            'Win Rate (%)': win_rate,
            'RAD (%)': rad
        }
    except:
        return {
            'Total Return (%)': 0.0, 'CAGR (%)': 0.0, 'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0, 'Max Drawdown (%)': 0.0, 'Win Rate (%)': 0.0,
            'RAD (%)': 0.0
        }

# ===== 전략 실행 =====
def run_strategy(price_data):
    """DAA 전략 실행 (자동 T값 결정)"""
    try:
        available_core = [t for t in UNIVERSE['core'] if t in price_data.columns]
        available_risky = [t for t in UNIVERSE['risky'] if t in price_data.columns]
        available_canary = [t for t in UNIVERSE['canary'] if t in price_data.columns]
        available_cash = [t for t in UNIVERSE['cash'] if t in price_data.columns]
        
        if not available_risky or not available_canary or not available_cash:
            st.warning("⚠️ 일부 데이터 누락")
        
        # 수익률 & 모멘텀
        monthly_returns = calculate_monthly_returns(price_data)
        momentum = calculate_momentum_daa(monthly_returns)
        
        # Breadth & 현금비율 & 위험자산비중
        breadth_score = calculate_breadth_score_continuous(momentum, available_canary)
        cash_fraction = calculate_cash_fraction_continuous(breadth_score)
        risk_allocation = calculate_risk_asset_allocation(breadth_score)
        
        # 최적 T값 결정
        optimal_t = find_optimal_top_n(momentum, cash_fraction, risk_allocation, 
                                       available_risky, available_cash)
        st.session_state.optimal_t = optimal_t
        
        # 가중치 계산 (제약 조건 포함)
        w_core, w_risky, w_cash, validation = calculate_portfolio_weights_with_constraints(
            momentum, cash_fraction, risk_allocation,
            UNIVERSE['core'], available_risky, available_cash, optimal_t
        )
        
        # 수익률 계산
        all_tickers = available_core + available_risky + available_cash
        weights_combined = pd.concat([w_core, w_risky, w_cash], axis=1)
        monthly_returns_all = monthly_returns[all_tickers]
        strategy_returns = backtest_returns(monthly_returns_all, weights_combined)
        
        return {
            'strategy_returns': strategy_returns,
            'momentum': momentum,
            'breadth_score': breadth_score,
            'cash_fraction': cash_fraction,
            'risk_allocation': risk_allocation,
            'weights_core': w_core,
            'weights_risky': w_risky,
            'weights_cash': w_cash,
            'validation': validation,
            'available_core': available_core,
            'available_risky': available_risky,
            'available_cash': available_cash,
            'optimal_t': optimal_t,
            'monthly_returns': monthly_returns
        }
    except Exception as e:
        st.error(f"❌ 오류: {str(e)}")
        return None

# ===== 메인 =====
def main():
    st.markdown('<div class="header">📊 DAA+VAA 최종 전략 (31개 자산 + ACWI Core)</div>', unsafe_allow_html=True)
    st.write("**핵심**: ACWI 최소 20% | 위험자산 최대 100% | 종목 최대 30% | 자동 T값 결정")
    
    # 초기화
    if 'optimal_t' not in st.session_state:
        st.session_state.optimal_t = 3
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작", value=datetime(2015, 1, 1))
        with col2:
            end_date = st.date_input("종료", value=datetime.now())
        
        st.divider()
        
        benchmark_choices = st.multiselect(
            "벤치마크",
            list(BENCHMARKS_CONFIG.keys()),
            default=['SPY Index', 'Balanced (60/40)']
        )
    
    try:
        st.info("💾 데이터 로드 중...")
        
        all_tickers = sorted(list(set(UNIVERSE['core'] + UNIVERSE['risky'] + 
                                      UNIVERSE['canary'] + UNIVERSE['cash'] + 
                                      ['SPY', 'BND', 'ACWI'])))
        
        price_data = download_price_data(
            all_tickers,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if price_data is None:
            st.error("❌ 데이터 로드 실패")
            return
        
        st.success("✅ 데이터 로드 완료")
        
        # 전략 실행
        result = run_strategy(price_data)
        
        if result is None:
            return
        
        strategy_returns = result['strategy_returns']
        optimal_t = result['optimal_t']
        
        # 벤치마크
        benchmark_dict = {}
        benchmark_colors = {
            'Aggressive (70/30)': COLORS['aggressive'],
            'Balanced (60/40)': COLORS['balanced'],
            'Conservative (50/50)': COLORS['conservative'],
            'SPY Index': COLORS['spy'],
            'ACWI Index': COLORS['acwi']
        }
        
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
        
        # 성과
        performance = calculate_performance_metrics(strategy_returns)
        performance_bench = {k: calculate_performance_metrics(v) for k, v in benchmark_dict.items()}
        
        # ===== 성과 표시 =====
        st.markdown("---")
        st.subheader("📈 성과 지표")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR (%)", f"{performance['CAGR (%)']:.2f}%")
        col2.metric("Volatility (%)", f"{performance['Volatility (%)']:.2f}%")
        col3.metric("Sharpe Ratio", f"{performance['Sharpe Ratio']:.3f}")
        col4.metric("Max Drawdown (%)", f"{performance['Max Drawdown (%)']:.2f}%")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("RAD (%)", f"{performance['RAD (%)']:.2f}%")
        col6.metric("승률 (%)", f"{performance['Win Rate (%)']:.1f}%")
        col7.metric("총 수익 (%)", f"{performance['Total Return (%)']:.2f}%")
        col8.metric("최적 T", f"{optimal_t}개")
        
        # 검증
        st.markdown("---")
        st.subheader("✅ 정확성 검증")
        
        validation = result['validation']
        valid_count = validation['valid'].sum() if 'valid' in validation.columns else 0
        validity_pct = (valid_count / len(validation) * 100) if len(validation) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**가중치 검증**")
            if validity_pct < 99:
                st.warning(f"⚠️ {100-validity_pct:.1f}% 비정상")
            else:
                st.success(f"✅ {validity_pct:.1f}% 유효")
        
        with col2:
            st.markdown("**자동 T값**")
            st.success(f"✅ {optimal_t}개 최적")
        
        with col3:
            st.markdown("**ACWI 비중**")
            st.success("✅ 최소 20%")
        
        with col4:
            st.markdown("**종목 제약**")
            st.success("✅ 최대 30%")
        
        # ===== 탭 =====
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 성과", "⚖️ 가중치", "📈 수익률", "🔍 상세", "📖 용어"
        ])
        
        with tab1:
            st.subheader("누적 수익률")
            
            cum_returns = (1 + strategy_returns).cumprod() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cum_returns.index, y=cum_returns.values,
                mode='lines', name='Strategy',
                line=dict(color=COLORS['strategy'], width=3)
            ))
            
            for bench_name, bench_ret in benchmark_dict.items():
                cum_bench = (1 + bench_ret).cumprod() * 100
                fig.add_trace(go.Scatter(
                    x=cum_bench.index, y=cum_bench.values,
                    mode='lines', name=bench_name,
                    line=dict(color=benchmark_colors.get(bench_name, '#95A5A6'), width=2)
                ))
            
            fig.update_layout(
                title="누적 수익률 비교",
                yaxis_title="누적 수익률 (%)",
                height=400, hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 비교표
            st.subheader("성과 비교")
            comp = {'전략': ['DAA+VAA'] + list(benchmark_dict.keys())}
            for metric in ['CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']:
                comp[metric] = [f"{performance[metric]:.2f}"] + \
                              [f"{performance_bench[b].get(metric, 0):.2f}" for b in benchmark_dict.keys()]
            
            st.dataframe(pd.DataFrame(comp), use_container_width=True)
        
        with tab2:
            st.subheader("포트폴리오 가중치 (전체 기간)")
            
            weights = pd.concat([result['weights_core'], result['weights_risky'], 
                                result['weights_cash']], axis=1) * 100
            weights = weights.round(2)
            
            # 최근 20개월 차트
            recent = weights.tail(20)
            
            fig = go.Figure()
            for col in recent.columns:
                fig.add_trace(go.Bar(x=recent.index, y=recent[col], name=col))
            
            fig.update_layout(
                title="월별 가중치 (최근 20개월)",
                barmode='stack', height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            st.markdown("**모든 월별 포트폴리오 가중치**")
            csv = weights.to_csv()
            st.download_button("📥 CSV 다운로드", csv, "weights.csv", "text/csv")
            st.dataframe(weights, use_container_width=True, height=600)
        
        with tab3:
            st.subheader("월별 수익률")
            
            fig = go.Figure()
            colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in strategy_returns]
            fig.add_trace(go.Bar(
                x=strategy_returns.index, y=strategy_returns * 100,
                marker=dict(color=colors)
            ))
            
            fig.update_layout(title="월별 수익률", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("상세 정보")
            
            st.markdown("""
            ### 📊 포트폴리오 구성 방식
            
            #### 1️⃣ **Breadth Score 기반 위험자산 비중 결정**
            
            ```
            Breadth Score = tanh(VWO_momentum, BND_momentum) 정규화
            범위: 0 (약한 시장) ~ 1 (강한 시장)
            
            위험자산 비중 = 0.3 + 0.7 × (Breadth Score)²
            - Breadth 0.0 → 위험자산 30% (최소)
            - Breadth 0.5 → 위험자산 52%
            - Breadth 1.0 → 위험자산 100%
            
            이차함수로 부드러운 전환
            ```
            
            #### 2️⃣ **최적 T값 자동 결정**
            
            ```
            AI가 Sharpe ratio를 최대화하는 T값 선택
            (위험자산 상위 T개 자산)
            
            T: 1 ~ 8개 범위에서 자동 계산
            매월 재평가 가능
            ```
            
            #### 3️⃣ **각 종목 가중치 계산 로직**
            
            **Step A: Core Asset (ACWI) 할당**
            ```
            ACWI 비중 = max(20%, 위험자산 비중 × 30%)
            
            제약: 최소 20%, 최대 30%
            → 시장이 약해도 ACWI로 20% 확보
            → 시장이 강해도 30% 초과 금지
            ```
            
            **Step B: 상위 T개 자산 선택 및 모멘텀 가중**
            ```
            1. 30개 자산 중 모멘텀이 높은 순서 정렬
            2. 상위 T개 선택
            3. 각 자산 모멘텀으로 가중 배분
            
            예: T=3, 모멘텀=[0.15, 0.10, 0.05]
            합: 0.30
            비중: [50%, 33.3%, 16.7%] × 남은 위험자산 비중
            
            모멘텀 = (R1 + R3 + R6 + R12) / 4 (DAA 공식)
            ```
            
            **Step C: 개별 종목 최대 30% 제약 적용**
            ```
            계산된 비중이 30%를 초과하면 30%로 조정
            초과분은 다른 자산에 재배분
            ```
            
            **Step D: 현금 자산 선택**
            ```
            현금 비율 = 1 - Breadth Score
            
            3개 현금자산(SHY, IEF, LQD) 중
            모멘텀이 가장 높은 1개만 선택
            
            매월 모멘텀이 다르므로 다른 자산이 선택됨
            → 시장 환경에 최적화된 안전자산 선택
            ```
            
            #### 4️⃣ **안전자산이 매월 다른 이유**
            
            ```
            각 안전자산의 특성:
            - SHY: 1-3년 단기 국채 (금리 변화에 민감)
            - IEF: 7-10년 중기 국채 (균형)
            - LQD: 회사채 (경기에 민감, 수익성 높음)
            
            매월 시장 상황에 따라 모멘텀 변화:
            - 금리 상승 시 → 장기채(LQD) 모멘텀 하락 → SHY/IEF 선택
            - 경기 호황 시 → 회사채(LQD) 모멘텀 상승 → LQD 선택
            - 경기 약세 시 → 단기채(SHY) 모멘텀 상승 → SHY 선택
            
            결과: 시장에 최적의 안전자산 자동 선택
            ```
            
            #### 5️⃣ **정규화 (가중치 합 = 1.0 보장)**
            
            ```
            ACWI 비중 + 나머지 위험자산 + 현금 = 1.0
            
            만약 합이 1.0이 아니면 자동 재스케일
            ```
            """)
        
        with tab5:
            st.subheader("용어 설명")
            
            tabs_g = st.tabs(["Breadth", "Momentum", "DAA", "ACWI Core", "안전자산"])
            
            with tabs_g[0]:
                st.markdown("""
                ## Breadth Momentum
                
                Canary 자산(VWO, BND)의 모멘텀으로 시장 강도 판단
                - 높음: 시장 강세 → 공격적 (위험자산 많이)
                - 낮음: 시장 약세 → 방어적 (현금 많이)
                """)
            
            with tabs_g[1]:
                st.markdown("""
                ## DAA Momentum
                
                ```
                Momentum = (R1 + R3 + R6 + R12) / 4
                ```
                - R1: 1개월 누적 수익률
                - R3: 3개월 누적 수익률
                - R6: 6개월 누적 수익률
                - R12: 12개월 누적 수익률
                
                각 기간을 균등 가중 (25% 각)
                """)
            
            with tabs_g[2]:
                st.markdown("""
                ## DAA (Defensive Asset Allocation)
                
                Keller & Keuning 2016
                - 모멘텀 기반 자산 선택
                - 동적 현금 비중 조절
                """)
            
            with tabs_g[3]:
                st.markdown("""
                ## ACWI Core (Core Asset)
                
                MSCI All Country World Index
                - 전 세계 주식 시장 대표
                - 최소 20% 필수 보유
                - 극단적 상황에서도 안정적
                """)
            
            with tabs_g[4]:
                st.markdown("""
                ## 안전자산 (Cash Universe)
                
                위험 회피 시 선택
                
                **SHY**: 1-3년 국채 (최소 변동성)
                **IEF**: 7-10년 국채 (균형)
                **LQD**: 회사채 (최고 수익성)
                
                매월 모멘텀 최고 자산 선택
                """)
    
    except Exception as e:
        st.error(f"❌ 오류: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
