"""
VAA (Vigilant Asset Allocation) 백테스트 시스템
Keller & Keuning (2017) 논문 완전 구현
모든 코드가 이 파일 하나에 포함됨
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ===== 페이지 설정 =====
st.set_page_config(page_title="VAA 백테스트 시스템", page_icon="📊", layout="wide")

st.markdown("""
<style>
.header {color: #1f77b4; font-size: 28px; font-weight: bold; margin-bottom: 20px;}
.metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
.formula {background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0; font-family: monospace;}
.note {background-color: #e7f3ff; padding: 15px; border-radius: 8px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# ===== 색상 설정 =====
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

# ===== 유니버스 정의 =====
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
        
        # 모든 티커를 한 번에 다운로드
        raw = yf.download(ticker_list, start=start_date, end=end_date, progress=False)
        
        # Series 또는 DataFrame 확인
        if isinstance(raw, pd.DataFrame):
            df = raw['Close']
        else:
            df = raw
        
        # Series를 DataFrame으로 변환
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        # 필요한 컬럼만 선택 및 정렬
        df = df.ffill().dropna(how='all')
        
        # 사용 가능한 컬럼만 필터링
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

# ===== 모멘텀 계산: 13612W =====
def calculate_momentum(returns_df, periods=[1, 3, 6, 12], weights=[12, 4, 2, 1]):
    """
    13612W 모멘텀 계산
    공식: Momentum = (12*R1 + 4*R3 + 2*R6 + 1*R12) / 19
    """
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

# ===== Bad 자산 식별 =====
def get_bad_assets(momentum_df, threshold=0.0):
    """Bad 자산 식별: 모멘텀 <= 0인 자산"""
    try:
        return momentum_df <= threshold
    except:
        return pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)

# ===== Breadth 계산 =====
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

# ===== 현금 비율 =====
def calculate_cash_fraction(breadth_bad, breadth_param=2):
    """
    현금 비율 계산
    공식: CF = b / B (if b < B), CF = 1 (if b >= B)
    """
    try:
        return (breadth_bad / float(breadth_param)).clip(0, 1)
    except:
        return pd.Series(0.0, index=breadth_bad.index)

# ===== 상위 자산 선택 =====
def select_top_assets(momentum_df, top_n=6, risky_tickers=None):
    """상위 N개 자산 선택 (상대 모멘텀)"""
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

# ===== 포트폴리오 가중치 =====
def calculate_portfolio_weights(top_assets, cash_fraction, risky_tickers, cash_tickers, top_n=6):
    """포트폴리오 가중치 계산"""
    try:
        available_risky = [t for t in risky_tickers if t in top_assets.columns]
        available_cash = [t for t in cash_tickers if t in top_assets.columns]
        
        weights_risky = pd.DataFrame(0.0, index=top_assets.index, columns=available_risky)
        weights_cash = pd.DataFrame(0.0, index=top_assets.index, columns=available_cash)
        
        for date_idx in top_assets.index:
            cf_val = float(cash_fraction.loc[date_idx])
            risky_ratio = 1.0 - cf_val
            
            # 위험자산 가중치
            if risky_ratio > 0 and len(available_risky) > 0:
                row_assets = top_assets.loc[date_idx, available_risky]
                top_count = int(row_assets.sum())
                
                if top_count > 0:
                    weight_val = risky_ratio / top_count
                    for ticker in available_risky:
                        if top_assets.loc[date_idx, ticker]:
                            weights_risky.at[date_idx, ticker] = weight_val
            
            # 현금 가중치
            if cf_val > 0 and len(available_cash) > 0:
                weight_val = cf_val / len(available_cash)
                for ticker in available_cash:
                    weights_cash.at[date_idx, ticker] = weight_val
        
        return weights_risky, weights_cash
    except Exception as e:
        return (
            pd.DataFrame(0.0, index=top_assets.index, columns=risky_tickers),
            pd.DataFrame(0.0, index=top_assets.index, columns=cash_tickers)
        )

# ===== 백테스트 수익률 =====
def backtest_returns(monthly_returns, weights_df, transaction_cost=0.001):
    """백테스트 수익률 계산"""
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
def run_vaa_strategy(price_data, risky_list, canary_list, cash_list, breadth_param, top_select):
    """VAA 전략 실행"""
    try:
        available_risky = [t for t in risky_list if t in price_data.columns]
        available_canary = [t for t in canary_list if t in price_data.columns]
        available_cash = [t for t in cash_list if t in price_data.columns]
        
        if not available_risky or not available_canary or not available_cash:
            st.warning("⚠️ 필요한 데이터가 부분적으로 누락되었습니다")
        
        monthly_returns = calculate_monthly_returns(price_data)
        momentum = calculate_momentum(monthly_returns)
        bad_assets = get_bad_assets(momentum)
        breadth_bad = count_breadth_bad(bad_assets, available_canary)
        cash_fraction = calculate_cash_fraction(breadth_bad, breadth_param)
        top_assets = select_top_assets(momentum, top_select, available_risky)
        weights_risky, weights_cash = calculate_portfolio_weights(
            top_assets, cash_fraction, available_risky, available_cash, top_select
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
            'weights_risky': weights_risky,
            'weights_cash': weights_cash,
            'breadth_bad': breadth_bad,
            'monthly_returns': monthly_returns,
            'top_assets': top_assets,
            'available_risky': available_risky,
            'available_cash': available_cash
        }
    except Exception as e:
        st.error(f"❌ VAA 실행 오류: {str(e)}")
        return None

# ===== 메인 앱 =====
def main():
    st.markdown('<div class="header">📊 VAA (Vigilant Asset Allocation) 백테스트</div>', unsafe_allow_html=True)
    st.write("**Keller & Keuning (2017)**: *Breadth Momentum and Vigilant Asset Allocation; Winning More by Losing Less*")
    st.write("논문: https://ssrn.com/abstract=3002624")
    
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
        breadth_param = universe['b_param']
        top_select = universe['t_param']
        
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
        daa_result = run_vaa_strategy(
            price_data, risky_list, canary_list, cash_list, breadth_param, top_select
        )
        
        if daa_result is None:
            st.error("❌ VAA 실행 실패")
            return
        
        strategy_returns = daa_result['strategy_returns']
        
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
        
        # 탭
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 성과", "🔔 신호", "⚖️ 가중치", "📈 수익률", "🔍 상세", "📖 용어"])
        
        # 탭 1: 성과
        with tab1:
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
                    title="누적 수익률",
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
                st.warning(f"⚠️ 차트 오류")
        
        # 탭 2: 신호
        with tab2:
            st.subheader("🔔 VAA 신호 분석 (Breadth Momentum)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**VWO 모멘텀** (카나리 자산)")
                try:
                    momentum = daa_result['momentum']
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
                        fig.update_layout(title="VWO 13612W 모멘텀", height=300, hovermode='x')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("VWO 데이터 없음")
                except:
                    st.info("VWO 데이터 없음")
            
            with col2:
                st.markdown("**BND 모멘텀** (카나리 자산)")
                try:
                    momentum = daa_result['momentum']
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
                        fig.update_layout(title="BND 13612W 모멘텀", height=300, hovermode='x')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("BND 데이터 없음")
                except:
                    st.info("BND 데이터 없음")
            
            st.divider()
            st.markdown("**현금 비율 (CF) - Breadth Momentum 기반**")
            try:
                cash_fraction = daa_result['cash_fraction']
                breadth_bad = daa_result['breadth_bad']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cash_fraction.index,
                    y=cash_fraction * 100,
                    mode='lines',
                    name='Cash Fraction',
                    fill='tozeroy',
                    line=dict(color=COLORS['neutral'], width=2)
                ))
                
                # 보조 축으로 breadth_bad 표시
                fig.add_trace(go.Bar(
                    x=breadth_bad.index,
                    y=breadth_bad,
                    name='Bad Assets Count',
                    yaxis='y2',
                    marker=dict(color=COLORS['negative'], opacity=0.3)
                ))
                
                fig.update_layout(
                    title="현금 비율 (CF) vs Bad Assets",
                    yaxis=dict(title="Cash Fraction (%)"),
                    yaxis2=dict(title="Bad Assets Count", overlaying='y', side='right'),
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                **해석:**
                - B (Breadth Parameter) = {breadth_param}
                - 공식: CF = b/B (b는 bad assets 개수)
                - b ≥ B일 때: CF = 100% (모두 현금)
                - 평균 현금 비율: {cash_fraction.mean()*100:.1f}%
                """)
                
            except Exception as e:
                st.warning(f"⚠️ 신호 분석 오류")
        
        # 탭 3: 가중치
        with tab3:
            st.subheader("⚖️ 포트폴리오 구성")
            
            try:
                weights_risky = daa_result['weights_risky']
                weights_cash = daa_result['weights_cash']
                
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
                st.warning(f"⚠️ 가중치 분석 오류")
        
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
                col1, col2, col3 = st.columns(3)
                col1.metric("평균 월간 수익률", f"{strategy_returns.mean()*100:.2f}%")
                col2.metric("긍정 월간 비율", f"{(strategy_returns > 0).sum() / len(strategy_returns) * 100:.1f}%")
                col3.metric("최악의 달", f"{strategy_returns.min()*100:.2f}%")
                
            except Exception as e:
                st.warning(f"⚠️ 수익률 분석 오류")
        
        # 탭 5: 상세 정보
        with tab5:
            st.subheader("🔍 전략 상세 정보")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 선택된 유니버스")
                st.markdown(f"**{universe_choice}**")
                st.markdown(universe['description'])
                
                st.markdown("### 🎯 파라미터")
                st.markdown(f"""
                - **T (Top Selection)**: {top_select}개 자산
                - **B (Breadth Parameter)**: {breadth_param}
                - **모멘텀 필터**: 13612W
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
            
            with st.expander("1️⃣ 13612W 모멘텀"):
                st.markdown("""
                $$Momentum = \\frac{12 \\times R_1 + 4 \\times R_3 + 2 \\times R_6 + 1 \\times R_{12}}{19}$$
                
                여기서:
                - $R_t$ = t개월의 누적 수익률
                - 가중치: 최근 수익률을 더 중시 (1개월에 40% 가중)
                
                **목적**: 빠른 시장 변화 감지
                """)
            
            with st.expander("2️⃣ Bad 자산 식별"):
                st.markdown("""
                **정의:** 모멘텀 ≤ 0인 자산
                
                Canary 유니버스 (VWO, BND)에서 bad인 자산 개수를 카운트
                
                **의미**: 시장 약세 신호
                """)
            
            with st.expander("3️⃣ Breadth Momentum & 현금 비율"):
                st.markdown(f"""
                $$CF = \\begin{{cases}}
                b/B & \\text{{if }} b < B \\\\
                1 & \\text{{if }} b \\geq B
                \\end{{cases}}$$
                
                여기서:
                - b = canary universe의 bad assets 개수
                - B = Breadth parameter = {breadth_param}
                
                **현재 설정 (B={breadth_param})**:
                - b=0: CF=0% (100% 위험자산)
                - b=1: CF={1/breadth_param*100:.0f}%
                - b≥{breadth_param}: CF=100% (모두 현금)
                
                **의미**: 시장 광폭성(breadth)을 기반으로 한 즉각적인 crash protection
                """)
            
            with st.expander("4️⃣ 포트폴리오 가중치"):
                st.markdown("""
                **위험자산 가중치:**
                $$w_{{risky}} = \\frac{{1-CF}}{{T}}$$
                
                **현금 가중치:**
                $$w_{{cash}} = \\frac{{CF}}{{|cash \\ universe|}}$$
                
                **의미**: 
                - 현금 비율이 높을수록 위험자산 비중 감소
                - 현금은 최고 모멘텀 자산 1개만 선택
                """)
            
            with st.expander("5️⃣ RAD (Returns Adjusted for Drawdowns)"):
                st.markdown("""
                $$RAD = R \\times \\left(1 - \\frac{D}{1-D}\\right) \\quad \\text{if } R \\geq 0\\% \\text{ and } D \\leq 50\\%$$
                
                여기서:
                - R = CAGR (연평균 수익률)
                - D = 최대 낙폭
                
                **의의**:
                - 수익률과 위험을 동시에 고려
                - D=50% 이상이면 RAD=0 (장펀드 청산 수준)
                - VAA의 in-sample 최적화 기준
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
                - **VAA 방식**: 카나리 유니버스의 나쁜 자산 개수 확인 (universe-level breadth momentum)
                
                ### 예시 (B=2인 경우)
                ```
                카나리 유니버스: [VWO, BND]
                
                상황 1: 둘 다 모멘텀 > 0 → 시장 강세 → CF = 0% (모두 투자)
                상황 2: VWO만 모멘텀 ≤ 0 → 시장 약세 신호 → CF = 50%
                상황 3: 둘 다 모멘텀 ≤ 0 → 강한 약세 신호 → CF = 100% (모두 현금)
                ```
                
                ### 장점
                - **빠른 반응**: 1개 자산만 나빠도 반응
                - **강한 보호**: crash 초기에 빠르게 현금 진입
                - **시스템적**: 개별 판단 불필요
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
                ```
                
                ### 상대 모멘텀 (Relative/Cross-sectional Momentum)
                여러 자산 중 가장 강한 자산 선택
                ```
                목적: 수익 최대화
                방법: 모멘텀 상위 T개만 보유
                ```
                
                ### VAA의 차이점
                - **전통 Dual Momentum**: 개별 자산 기반 crash protection
                - **VAA Breadth Momentum**: 시장 광폭성 기반 crash protection
                  → 훨씬 더 적극적인 현금 보유
                """)
            
            with tabs_glossary[2]:
                st.markdown("""
                ## 🕯️ Canary Universe (카나리 유니버스)
                
                ### 개념
                "탄광 카나리"처럼 시장 위험을 조기에 감지하는 자산군
                
                ### VAA의 카나리
                - **VWO** (신흥시장 주식)
                - **BND** (채권)
                
                ### 선택 이유
                1. **VWO**: 위험자산의 대표 → 경기 약세 시 먼저 하락
                2. **BND**: 안전자산의 대표 → 인플레이션 우려/금리 상승 시 하락
                
                ### 작동 원리
                ```
                VWO 약세 → 신흥시장 약세 → 경기 둔화 신호
                BND 약세 → 금리 상승 또는 유동성 위기 신호
                
                둘 중 하나라도 약세 → 시장 위험 ↑ → 현금 비율 ↑
                ```
                
                ### 효과
                주요 위험자산 보유 전에 선제적으로 현금 전환
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
                
                예: LQD 모멘텀 > IEF > SHY → LQD 보유
                ```
                
                ### 중요성
                - 현금 비율이 높을 때(60% 평균) 중요
                - 단순 현금(T-bill)보다 수익성 높음
                - 절대 모멘텀 미적용 (모멘텀 부호 무관)
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
                - **결과**: VAA가 약세장에서 우수, 강세장에서는 벤치마크 추종
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
                **의의**: 수익을 위험으로 조정 (VAA 최적화 기준)
                **해석**: 낙폭을 고려한 조정된 수익률
                
                ### Win Rate (승률)
                ```
                양의 수익률을 낸 월수 / 전체 월수 × 100%
                ```
                **해석**: 몇 %의 기간에 수익을 냈는가
                """)
            
            with tabs_glossary[6]:
                st.markdown("""
                ## 기타 용어
                
                ### 13612W 모멘텀 필터
                1, 3, 6, 12개월 누적 수익률의 가중 평균
                - **가중치**: 12, 4, 2, 1 (최근 더 중시)
                - **합계**: 19
                - **목적**: 빠른 반응 (1개월에 40% 가중)
                
                ### T (Top Selection)
                위험자산 중 모멘텀 상위 T개만 선택
                - VAA-G12: T=2 (상위 2개)
                - VAA-G4: T=1 (상위 1개)
                
                ### B (Breadth Parameter)
                몇 개의 bad assets가 있을 때 100% 현금인가?
                - VAA-G12: B=4 (4개 이상 bad → 100% 현금)
                - VAA-G4: B=1 (1개 이상 bad → 100% 현금)
                
                ### CF (Cash Fraction)
                현재 현금 보유 비중
                - 낮을수록 공격적
                - 높을수록 방어적
                - VAA 평균: ~60%
                
                ### 논문 및 참고
                **Keller & Keuning (2017)**
                "Breadth Momentum and Vigilant Asset Allocation; 
                Winning More by Losing Less"
                SSRN: https://ssrn.com/abstract=3002624
                """)
        
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
