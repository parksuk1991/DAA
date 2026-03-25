"""
DAA 백테스트 Streamlit 앱 - 최종 수정 (완전한 오류 처리)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

from config import RISKY_UNIVERSE_G12, RISKY_UNIVERSE_U6, RISKY_UNIVERSE_G4, CANARY_UNIVERSE, CASH_UNIVERSE, BENCHMARKS, COLORS
from utils import download_price_data
from daa_strategy import DAAStrategy, DAAConfig, DAABacktest


st.set_page_config(page_title="DAA 백테스트", page_icon="📊", layout="wide")

st.markdown("""
<style>
.metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
.header {color: #1f77b4; font-size: 24px; font-weight: bold; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_price_data(tickers_tuple, start, end):
    """가격 데이터"""
    try:
        progress_bar = st.progress(0)
        data = download_price_data(list(tickers_tuple), start, end, progress_bar)
        return data
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {str(e)}")
        return None


@st.cache_data
def run_daa(price_data_key, risky_tuple, canary_tuple, cash_tuple, breadth_param, top_select):
    """DAA 실행"""
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
        st.error(f"❌ DAA 계산 실패: {str(e)}")
        return None


def main():
    st.markdown('<div class="header">📊 DAA 백테스트 시스템</div>', unsafe_allow_html=True)
    st.write("Keller & Keuning의 DAA 전략")
    
    with st.sidebar:
        st.header("⚙️ 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작", value=datetime(2015, 1, 1))
        with col2:
            end_date = st.date_input("종료", value=datetime.now())
        
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
        
        breadth_param = st.radio("Breadth Parameter", [1, 2], index=1)
        top_select = st.slider("Top Selection", 1, 12, 6)
        
        benchmark_choice = st.selectbox(
            "벤치마크",
            list(BENCHMARKS.keys()) + ["None"],
            index=1
        )
    
    try:
        st.info("💾 데이터 로드 중...")
        
        all_tickers = list(risky_universe.keys()) + list(CANARY_UNIVERSE.keys()) + list(CASH_UNIVERSE.keys())
        unique_tickers = sorted(list(set(all_tickers)))
        
        price_data = load_price_data(
            tuple(unique_tickers),
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if price_data is None or price_data.empty:
            st.error("❌ 데이터 로드 실패")
            return
        
        st.success("✅ 데이터 로드 완료")
        
        risky_list = list(risky_universe.keys())
        canary_list = list(CANARY_UNIVERSE.keys())
        cash_list = list(CASH_UNIVERSE.keys())
        
        daa_strategy = run_daa(
            price_data,
            tuple(risky_list),
            tuple(canary_list),
            tuple(cash_list),
            breadth_param,
            top_select
        )
        
        if daa_strategy is None:
            st.error("❌ DAA 계산 실패")
            return
        
        # 벤치마크
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
            except:
                pass
        
        # 백테스트
        daa_backtest = DAABacktest(daa_strategy)
        daa_performance = daa_backtest.run(benchmark_returns)
        
        # 탭
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 성과", "🔔 신호", "⚖️ 가중치", "📈 수익률", "ℹ️ 정보"])
        
        # 탭 1: 성과
        with tab1:
            st.subheader("성과 지표")
            
            col1, col2, col3, col4 = st.columns(4)
            
            try:
                if 'CAGR (%)' in daa_performance:
                    col1.metric("CAGR (%)", f"{float(daa_performance['CAGR (%)']):,.2f}%")
            except:
                col1.metric("CAGR (%)", "N/A")
            
            try:
                if 'Volatility (%)' in daa_performance:
                    col2.metric("Volatility (%)", f"{float(daa_performance['Volatility (%)']):,.2f}%")
            except:
                col2.metric("Volatility (%)", "N/A")
            
            try:
                if 'Sharpe Ratio' in daa_performance:
                    col3.metric("Sharpe Ratio", f"{float(daa_performance['Sharpe Ratio']):,.4f}")
            except:
                col3.metric("Sharpe Ratio", "N/A")
            
            try:
                if 'Max Drawdown (%)' in daa_performance:
                    col4.metric("Max Drawdown (%)", f"{float(daa_performance['Max Drawdown (%)']):,.2f}%")
            except:
                col4.metric("Max Drawdown (%)", "N/A")
            
            col5, col6, col7 = st.columns(3)
            
            try:
                if 'Win Rate (%)' in daa_performance:
                    col5.metric("승률", f"{float(daa_performance['Win Rate (%)']):,.2f}%")
            except:
                col5.metric("승률", "N/A")
            
            try:
                if 'Total Return (%)' in daa_performance:
                    col6.metric("총 수익", f"{float(daa_performance['Total Return (%)']):,.2f}%")
            except:
                col6.metric("총 수익", "N/A")
            
            try:
                if 'Monthly Volatility (%)' in daa_performance:
                    col7.metric("월간 변동성", f"{float(daa_performance['Monthly Volatility (%)']):,.2f}%")
            except:
                col7.metric("월간 변동성", "N/A")
            
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
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ 차트 오류: {str(e)}")
        
        # 탭 2: 신호
        with tab2:
            st.subheader("DAA 신호")
            
            try:
                signals = daa_strategy.get_signals()
                
                if signals is not None and not signals.empty:
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
                                fig_vwo.update_layout(title="VWO 모멘텀", height=300)
                                st.plotly_chart(fig_vwo, use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ VWO 차트 오류")
                    
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
                                fig_bnd.update_layout(title="BND 모멘텀", height=300)
                                st.plotly_chart(fig_bnd, use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ BND 차트 오류")
                    
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
                            fig_cf.update_layout(title="현금 비율", height=300)
                            st.plotly_chart(fig_cf, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ 현금 비율 차트 오류")
                else:
                    st.warning("⚠️ 신호 데이터 없음")
            except Exception as e:
                st.warning(f"⚠️ 신호 분석 오류")
        
        # 탭 3: 가중치
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
                
                fig.update_layout(title="월별 가중치", barmode='stack', height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ 가중치 차트 오류")
        
        # 탭 4: 수익률
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
                st.write(f"Breadth Parameter (B): {breadth_param}")
                st.write(f"Top Selection (T): {top_select}")
                st.write(f"모멘텀 기간: 1, 3, 6, 12개월")
            
            st.write("**논문**")
            st.write("Keller & Keuning (2018)")
            st.write("SSRN: https://ssrn.com/abstract=3212862")
    
    except Exception as e:
        st.error(f"❌ 오류: {str(e)}")
        st.info("설정을 다시 확인해주세요.")


if __name__ == "__main__":
    main()
