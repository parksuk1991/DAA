"""
DAA 백테스트 시스템 - 올인원 Streamlit 앱
모든 코드가 이 파일 하나에 포함됨
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import warnings
import time

warnings.filterwarnings('ignore')

# ===== 페이지 설정 =====
st.set_page_config(page_title="DAA 백테스트", page_icon="📊", layout="wide")

st.markdown("""
<style>
.header {color: #1f77b4; font-size: 24px; font-weight: bold; margin-bottom: 20px;}
.metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ===== 색상 설정 =====
COLORS = {
    'daa': '#FF6B6B',
    'benchmark': '#4ECDC4',
    'positive': '#2ECC71',
    'negative': '#E74C3C',
    'canary_bad': '#F7B731'
}

# ===== 자산 유니버스 =====
UNIVERSES = {
    "DAA-G12 (12개)": {
        'risky': ['SPY', 'IWM', 'QQQ', 'VEA', 'VGK', 'EWJ', 'VWO', 'VNQ', 'GSG', 'GLD', 'TLT', 'HYG'],
        'canary': ['VWO', 'BND'],
        'cash': ['SHY', 'IEF', 'LQD']
    },
    "DAA-U6 (6개)": {
        'risky': ['VTV', 'VUG', 'VBV', 'VBR', 'BIL', 'LQD'],
        'canary': ['VWO', 'BND'],
        'cash': ['SHY', 'IEF', 'LQD']
    },
    "DAA-G4 (4개)": {
        'risky': ['SPY', 'VEA', 'VWO', 'BND'],
        'canary': ['VWO', 'BND'],
        'cash': ['SHY', 'IEF', 'LQD']
    }
}

BENCHMARKS = {
    'Balanced 60/40': {'stocks': 0.6, 'bonds': 0.4},
    'Aggressive 70/30': {'stocks': 0.7, 'bonds': 0.3},
    'Conservative 50/50': {'stocks': 0.5, 'bonds': 0.5}
}

@st.cache_data
def download_price_data(ticker_list, start_date, end_date):
    """yfinance에서 가격 데이터 다운로드 (rate limit 처리)"""
    try:
        all_data = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(ticker_list):
            try:
                progress_bar.progress(min((i + 1) / len(ticker_list), 0.99))
                
                # Rate limit 처리: 각 다운로드 사이에 딜레이 추가
                for attempt in range(3):  # 3회 시도
                    try:
                        data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
                        if data is not None and len(data) > 0:
                            all_data[ticker] = data
                        break  # 성공하면 루프 탈출
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(1)  # 1초 대기
                        else:
                            raise e
            except Exception as e:
                # 단일 티커 실패해도 계속 진행
                continue
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data)
        df = df.ffill().bfill()
        
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

# ===== 모멘텀 계산 =====
def calculate_momentum(returns_df, periods=[1, 3, 6, 12], weights=[12, 4, 2, 1]):
    """13612W 모멘텀 계산"""
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
    """Bad 자산 식별"""
    try:
        return momentum_df <= threshold
    except:
        return pd.DataFrame(False, index=momentum_df.index, columns=momentum_df.columns)

# ===== Breadth 계산 =====
def count_breadth_bad(bad_assets_df, canary_tickers):
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

# ===== 현금 비율 =====
def calculate_cash_fraction(breadth_bad, breadth_param=2):
    """현금 비율 계산"""
    try:
        return (breadth_bad / float(breadth_param)).clip(0, 1)
    except:
        return pd.Series(0.0, index=breadth_bad.index)

# ===== 상위 자산 선택 =====
def select_top_assets(momentum_df, top_n=6, risky_tickers=None):
    """상위 N개 자산 선택"""
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
                'Sharpe Ratio': 0.0, 'Max Drawdown (%)': 0.0, 'Win Rate (%)': 0.0
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
        
        metrics = {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': drawdown,
            'Win Rate (%)': win_rate
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
            'Sharpe Ratio': 0.0, 'Max Drawdown (%)': 0.0, 'Win Rate (%)': 0.0
        }

# ===== DAA 전략 실행 =====
def run_daa_strategy(price_data, risky_list, canary_list, cash_list, breadth_param, top_select):
    """DAA 전략 실행"""
    try:
        # 사용 가능한 컬럼만 필터링
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
            'breadth_bad': breadth_bad
        }
    except Exception as e:
        st.error(f"❌ DAA 실행 오류: {str(e)}")
        return None

# ===== 메인 앱 =====
def main():
    st.markdown('<div class="header">📊 DAA 백테스트 시스템</div>', unsafe_allow_html=True)
    st.write("Keller & Keuning의 Defensive Asset Allocation 전략")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작", value=datetime(2015, 1, 1))
        with col2:
            end_date = st.date_input("종료", value=datetime.now())
        
        universe_choice = st.radio(
            "자산 유니버스",
            list(UNIVERSES.keys()),
            index=0
        )
        
        breadth_param = st.radio("Breadth Parameter", [1, 2], index=1)
        top_select = st.slider("Top Selection", 1, 12, 6)
        
        benchmark_choice = st.selectbox(
            "벤치마크",
            list(BENCHMARKS.keys()) + ["None"],
            index=0
        )
    
    # 메인 로직
    try:
        st.info("💾 데이터 로드 중...")
        
        universe = UNIVERSES[universe_choice]
        risky_list = universe['risky']
        canary_list = universe['canary']
        cash_list = universe['cash']
        
        all_tickers = sorted(list(set(risky_list + canary_list + cash_list)))
        
        price_data = download_price_data(
            all_tickers,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if price_data is None or price_data.empty:
            st.error("❌ 데이터 로드 실패")
            return
        
        st.success("✅ 데이터 로드 완료")
        
        # DAA 실행
        daa_result = run_daa_strategy(
            price_data, risky_list, canary_list, cash_list, breadth_param, top_select
        )
        
        if daa_result is None:
            st.error("❌ DAA 실행 실패")
            return
        
        strategy_returns = daa_result['strategy_returns']
        
        # 벤치마크
        benchmark_returns = None
        if benchmark_choice != "None" and 'SPY' in price_data.columns and 'BND' in price_data.columns:
            try:
                bm_weights = BENCHMARKS[benchmark_choice]
                spy_monthly = price_data['SPY'].resample('M').last().pct_change()
                bnd_monthly = price_data['BND'].resample('M').last().pct_change()
                benchmark_returns = spy_monthly * bm_weights['stocks'] + bnd_monthly * bm_weights['bonds']
            except:
                pass
        
        # 성과 계산
        performance = calculate_performance_metrics(strategy_returns, benchmark_returns)
        
        # 탭
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 성과", "🔔 신호", "⚖️ 가중치", "📈 수익률", "ℹ️ 정보"])
        
        # 탭 1: 성과
        with tab1:
            st.subheader("성과 지표")
            
            col1, col2, col3, col4 = st.columns(4)
            
            try:
                col1.metric("CAGR (%)", f"{performance['CAGR (%)']:.2f}%")
            except:
                col1.metric("CAGR (%)", "N/A")
            
            try:
                col2.metric("Volatility (%)", f"{performance['Volatility (%)']:.2f}%")
            except:
                col2.metric("Volatility (%)", "N/A")
            
            try:
                col3.metric("Sharpe Ratio", f"{performance['Sharpe Ratio']:.4f}")
            except:
                col3.metric("Sharpe Ratio", "N/A")
            
            try:
                col4.metric("Max Drawdown (%)", f"{performance['Max Drawdown (%)']:.2f}%")
            except:
                col4.metric("Max Drawdown (%)", "N/A")
            
            col5, col6, col7 = st.columns(3)
            
            try:
                col5.metric("승률", f"{performance['Win Rate (%)']:.2f}%")
            except:
                col5.metric("승률", "N/A")
            
            try:
                col6.metric("총 수익", f"{performance['Total Return (%)']:.2f}%")
            except:
                col6.metric("총 수익", "N/A")
            
            try:
                col7.metric("벤치마크", f"{performance.get('Benchmark Return (%)', 0):.2f}%")
            except:
                col7.metric("벤치마크", "N/A")
            
            # 누적 수익률 차트
            st.markdown("#### 누적 수익률")
            
            try:
                cum_returns = (1 + strategy_returns).cumprod() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    mode='lines',
                    name='DAA',
                    line=dict(color=COLORS['daa'], width=3)
                ))
                
                if benchmark_returns is not None:
                    cum_benchmark = (1 + benchmark_returns).cumprod() * 100
                    fig.add_trace(go.Scatter(
                        x=cum_benchmark.index,
                        y=cum_benchmark.values,
                        mode='lines',
                        name=benchmark_choice,
                        line=dict(color=COLORS['benchmark'], width=3)
                    ))
                
                fig.update_layout(
                    title="누적 수익률",
                    xaxis_title="날짜",
                    yaxis_title="누적 수익률 (%)",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ 차트 오류")
        
        # 탭 2: 신호
        with tab2:
            st.subheader("DAA 신호")
            
            try:
                momentum = daa_result['momentum']
                cash_fraction = daa_result['cash_fraction']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        if 'VWO' in momentum.columns:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=momentum.index,
                                y=momentum['VWO'],
                                mode='lines',
                                name='VWO',
                                line=dict(color='blue')
                            ))
                            fig.add_hline(y=0, line_dash="dash", line_color="red")
                            fig.update_layout(title="VWO 모멘텀", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("VWO 데이터 없음")
                
                with col2:
                    try:
                        if 'BND' in momentum.columns:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=momentum.index,
                                y=momentum['BND'],
                                mode='lines',
                                name='BND',
                                line=dict(color='green')
                            ))
                            fig.add_hline(y=0, line_dash="dash", line_color="red")
                            fig.update_layout(title="BND 모멘텀", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("BND 데이터 없음")
                
                st.markdown("#### 현금 비율")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cash_fraction.index,
                    y=cash_fraction * 100,
                    mode='lines+markers',
                    name='Cash Fraction',
                    fill='tozeroy',
                    line=dict(color=COLORS['canary_bad'])
                ))
                fig.update_layout(title="동적 현금 비율", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"⚠️ 신호 분석 오류")
        
        # 탭 3: 가중치
        with tab3:
            st.subheader("포트폴리오 가중치")
            
            try:
                weights_risky = daa_result['weights_risky']
                weights_cash = daa_result['weights_cash']
                
                weights = pd.concat([weights_risky, weights_cash], axis=1) * 100
                recent = weights.tail(10)
                
                fig = go.Figure()
                for col in recent.columns:
                    fig.add_trace(go.Bar(x=recent.index, y=recent[col], name=col))
                
                fig.update_layout(title="월별 가중치", barmode='stack', height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ 가중치 차트 오류")
        
        # 탭 4: 수익률
        with tab4:
            st.subheader("월별 수익률")
            
            try:
                fig = go.Figure()
                colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in strategy_returns]
                
                fig.add_trace(go.Bar(
                    x=strategy_returns.index,
                    y=strategy_returns * 100,
                    marker=dict(color=colors),
                    name='Monthly Return'
                ))
                
                fig.update_layout(title="월별 수익률", yaxis_title="수익률 (%)", height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ 수익률 차트 오류")
        
        # 탭 5: 정보
        with tab5:
            st.subheader("전략 정보")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**자산 유니버스**")
                st.write(f"위험자산: {len(risky_list)}개")
                st.write(f"카나리: {len(canary_list)}개")
                st.write(f"현금: {len(cash_list)}개")
            
            with col2:
                st.write("**파라미터**")
                st.write(f"Breadth (B): {breadth_param}")
                st.write(f"Top Selection (T): {top_select}")
                st.write(f"모멘텀: 1,3,6,12개월")
            
            st.write("**논문**")
            st.write("Keller & Keuning (2018)")
            st.write("'Breadth Momentum and the Canary Universe'")
            st.write("SSRN: https://ssrn.com/abstract=3212862")
    
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
