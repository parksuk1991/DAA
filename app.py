"""
DAA 백테스트 Streamlit 앱 - 완전 수정
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

from config import (
    RISKY_UNIVERSE_G12, RISKY_UNIVERSE_U6, RISKY_UNIVERSE_G4,
    CANARY_UNIVERSE, CASH_UNIVERSE, BENCHMARKS, COLORS
)
from utils import download_price_data
from daa_strategy import DAAStrategy, DAAConfig, DAABacktest


# ===== 페이지 설정 =====
st.set_page_config(
    page_title="DAA 백테스트",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .header {color: #1f77b4; font-size: 24px; font-weight: bold; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)


# ===== 캐시 =====
@st.cache_data
def load_price_data(tickers_tuple, start, end):
    """가격 데이터 로드"""
    try:
        progress_bar = st.progress(0)
        data = download_price_data(list(tickers_tuple), start, end, progress_bar)
        return data
    except Exception as e:
        st.error(f"데이터 로드 오류: {str(e)}")
        return None


@st.cache_data
def run_daa_backtest(price_data_key, risky_tuple, canary_tuple, cash_tuple, breadth_param, top_select):
    """DAA 백테스트"""
    try:
        config = DAAConfig(
            risky_tickers=list(risky_tuple),
            canary_tickers=list(canary_tuple),
            cash_tickers=list(cash_tuple),
            breadth_parameter=breadth_param,
            top_selection=top_select
        )
        
        daa = DAAStrategy(config).fit(price_data_key)
        return daa
    except Exception as e:
        st.error(f"DAA 계산 오류: {str(e)}")
        return None


# ===== 메인 =====
def main():
    st.markdown('<div class="header">📊 DAA 백테스트 시스템</div>', unsafe_allow_html=True)
    
    st.write("Keller & Keuning의 DAA 전략 - VWO(신흥시장)과 BND(채권)의 모멘텀으로 자동 Crash Protection")
    
    # ===== 사이드바 =====
    with st.sidebar:
        st.header("⚙️ 설정")
        
        st.subheader("📅 분석 기간")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작", value=datetime(2015, 1, 1))
        with col2:
            end_date = st.date_input("종료", value=datetime.now())
        
        st.subheader("🎯 자산 선택")
        universe_choice = st.radio(
            "자산 유니버스",
            ["DAA-G12 (12개)", "DAA-U6 (6개)", "DAA-G4 (4개)"],
            index=0
        )
        
        if "G12" in universe_choice:
            risky_universe = RISKY_UNIVERSE_G12
        elif "U6" in universe_choice:
            risky_universe = RISKY_UNIVERSE_U6
        else:
            risky_universe = RISKY_UNIVERSE_G4
        
        st.subheader("⚡ DAA 파라미터")
        breadth_param = st.radio("Breadth Parameter", [1, 2], index=1)
        top_select = st.slider("Top Selection", 1, 12, 6)
        
        st.subheader("🎲 벤치마크")
        benchmark_choice = st.selectbox(
            "벤치마크",
            list(BENCHMARKS.keys()) + ["None"],
            index=1
        )
    
    # ===== 메인 로직 =====
    try:
        st.info("💾 데이터 로드 중...")
        
        # 모든 필요한 티커
        all_tickers = list(risky_universe.keys()) + list(CANARY_UNIVERSE.keys()) + list(CASH_UNIVERSE.keys())
        unique_tickers = sorted(list(set(all_tickers)))
        
        # 데이터 로드
        price_data = load_price_data(
            tuple(unique_tickers),
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if price_data is None or price_data.empty:
            st.error("데이터를 로드할 수 없습니다.")
            return
        
        st.success("✅ 데이터 로드 완료")
        
        # DAA 실행
        risky_list = list(risky_universe.keys())
        canary_list = list(CANARY_UNIVERSE.keys())
        cash_list = list(CASH_UNIVERSE.keys())
        
        daa_strategy = run_daa_backtest(
            price_data,
            tuple(risky_list),
            tuple(canary_list),
            tuple(cash_list),
            breadth_param,
            top_select
        )
        
        if daa_strategy is None:
            st.error("DAA 전략 계산에 실패했습니다.")
            return
        
        # ===== 벤치마크 =====
        benchmark_returns = None
        
        if benchmark_choice != "None":
            try:
                bm_weights = BENCHMARKS.get(benchmark_choice, {'stocks': 0.6, 'bonds': 0.4})
                
                if 'SPY' in price_data.columns and 'BND' in price_data.columns:
                    spy_prices = price_data['SPY']
                    bnd_prices = price_data['BND']
                    
                    spy_monthly = spy_prices.resample('M').last().pct_change()
                    bnd_monthly = bnd_prices.resample('M').last().pct_change()
                    
                    bm_returns = (
                        spy_monthly * float(bm_weights['stocks']) +
                        bnd_monthly * float(bm_weights['bonds'])
                    )
                    benchmark_returns = bm_returns
            except Exception as e:
                st.warning(f"⚠️ 벤치마크 계산 오류: {str(e)}")
                benchmark_returns = None
        
        # ===== 백테스트 =====
        daa_backtest = DAABacktest(daa_strategy)
        daa_performance = daa_backtest.run(benchmark_returns)
        
        if not daa_performance:
            st.error("성과 계산에 실패했습니다.")
            return
        
        # ===== 탭 =====
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 성과 비교",
            "🔔 신호 분석",
            "⚖️ 가중치",
            "📈 월별 수익률",
            "ℹ️ 정보"
        ])
        
        # ===== 탭 1: 성과 비교 =====
        with tab1:
            st.subheader("성과 지표")
            
            col1, col2, col3, col4 = st.columns(4)
            
            metrics_to_show = [
                ('CAGR (%)', 'CAGR (%)', col1),
                ('Volatility (%)', 'Volatility (%)', col2),
                ('Sharpe Ratio', 'Sharpe Ratio', col3),
                ('Max Drawdown (%)', 'Max Drawdown (%)', col4)
            ]
            
            for display_name, metric_key, col in metrics_to_show:
                if metric_key in daa_performance:
                    try:
                        value = float(daa_performance[metric_key])
                        if '%' in display_name:
                            col.metric(display_name, f"{value:.2f}%")
                        else:
                            col.metric(display_name, f"{value:.4f}")
                    except:
                        col.metric(display_name, "N/A")
            
            col5, col6, col7 = st.columns(3)
            
            for col, (name, key) in zip(
                [col5, col6, col7],
                [
                    ('승률', 'Win Rate (%)'),
                    ('총 수익', 'Total Return (%)'),
                    ('월간 변동성', 'Monthly Volatility (%)')
                ]
            ):
                if key in daa_performance:
                    try:
                        value = float(daa_performance[key])
                        col.metric(name, f"{value:.2f}%")
                    except:
                        col.metric(name, "N/A")
            
            # 누적 수익률
            st.markdown("#### 누적 수익률")
            
            try:
                cum_returns_daa = daa_strategy.get_cumulative_returns() * 100
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=cum_returns_daa.index,
                    y=cum_returns_daa.values,
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
                    hovermode='x unified',
                    height=450,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"차트 생성 오류: {str(e)}")
        
        # ===== 탭 2: 신호 분석 =====
        with tab2:
            st.subheader("DAA 신호")
            
            try:
                signals = daa_strategy.get_signals()
                
                if signals is not None and not signals.empty:
                    # VWO, BND 모멘텀
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        try:
                            if 'VWO_mom' in signals.columns:
                                fig_vwo = go.Figure()
                                fig_vwo.add_trace(go.Scatter(
                                    x=signals.index,
                                    y=signals['VWO_mom'],
                                    mode='lines',
                                    name='VWO Momentum',
                                    line=dict(color='blue')
                                ))
                                fig_vwo.add_hline(y=0, line_dash="dash", line_color="red")
                                fig_vwo.update_layout(title="VWO 모멘텀", height=350)
                                st.plotly_chart(fig_vwo, use_container_width=True)
                        except Exception as e:
                            st.warning(f"VWO 차트 오류: {str(e)}")
                    
                    with col2:
                        try:
                            if 'BND_mom' in signals.columns:
                                fig_bnd = go.Figure()
                                fig_bnd.add_trace(go.Scatter(
                                    x=signals.index,
                                    y=signals['BND_mom'],
                                    mode='lines',
                                    name='BND Momentum',
                                    line=dict(color='green')
                                ))
                                fig_bnd.add_hline(y=0, line_dash="dash", line_color="red")
                                fig_bnd.update_layout(title="BND 모멘텀", height=350)
                                st.plotly_chart(fig_bnd, use_container_width=True)
                        except Exception as e:
                            st.warning(f"BND 차트 오류: {str(e)}")
                    
                    # 현금 비율
                    try:
                        if 'Cash_Fraction' in signals.columns:
                            st.markdown("#### 현금 비율")
                            
                            fig_cf = go.Figure()
                            fig_cf.add_trace(go.Scatter(
                                x=signals.index,
                                y=signals['Cash_Fraction'] * 100,
                                mode='lines+markers',
                                name='Cash Fraction',
                                fill='tozeroy',
                                line=dict(color=COLORS['canary_bad'])
                            ))
                            fig_cf.update_layout(
                                title="동적 현금 비율",
                                xaxis_title="날짜",
                                yaxis_title="현금 비율 (%)",
                                height=350
                            )
                            st.plotly_chart(fig_cf, use_container_width=True)
                    except Exception as e:
                        st.warning(f"현금 비율 차트 오류: {str(e)}")
                else:
                    st.warning("신호 데이터가 없습니다.")
            except Exception as e:
                st.warning(f"신호 분석 오류: {str(e)}")
        
        # ===== 탭 3: 가중치 =====
        with tab3:
            st.subheader("포트폴리오 가중치")
            
            try:
                weights = daa_strategy.get_weights()
                weights_pct = weights * 100
                recent_weights = weights_pct.tail(10)
                
                fig = go.Figure()
                for col in recent_weights.columns:
                    fig.add_trace(go.Bar(
                        x=recent_weights.index,
                        y=recent_weights[col],
                        name=col
                    ))
                
                fig.update_layout(
                    title="월별 가중치",
                    barmode='stack',
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"가중치 차트 오류: {str(e)}")
        
        # ===== 탭 4: 월별 수익률 =====
        with tab4:
            st.subheader("월별 수익률")
            
            try:
                returns = daa_strategy.get_returns()
                
                fig = go.Figure()
                colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in returns]
                
                fig.add_trace(go.Bar(
                    x=returns.index,
                    y=returns * 100,
                    marker=dict(color=colors),
                    name='Monthly Return'
                ))
                
                fig.update_layout(
                    title="월별 수익률",
                    xaxis_title="날짜",
                    yaxis_title="수익률 (%)",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"수익률 차트 오류: {str(e)}")
        
        # ===== 탭 5: 정보 =====
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
                st.write(f"Breadth Parameter (B): {breadth_param}")
                st.write(f"Top Selection (T): {top_select}")
                st.write(f"모멘텀 기간: 1, 3, 6, 12개월")
            
            st.write("**논문**")
            st.write("Keller & Keuning (2018)")
            st.write("'Breadth Momentum and the Canary Universe'")
            st.write("SSRN: https://ssrn.com/abstract=3212862")
    
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.info("설정을 다시 확인하고 시도해주세요.")


if __name__ == "__main__":
    main()
