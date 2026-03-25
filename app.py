"""
DAA 백테스트 Streamlit 애플리케이션
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

from config import (
    RISKY_UNIVERSE_G12, RISKY_UNIVERSE_U6, RISKY_UNIVERSE_G4,
    CANARY_UNIVERSE, CASH_UNIVERSE, CASH_UNIVERSE_AGGRESSIVE,
    DAA_PARAMS, BENCHMARKS, COLORS
)
from utils import download_price_data
from daa_strategy import DAAStrategy, DAAConfig, DAABacktest, DAAComparison

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="DAA 백테스트 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 스타일 설정 =====
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header {
        color: #1f77b4;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== 캐시 설정 =====
@st.cache_data
def load_price_data(tickers, start, end):
    """가격 데이터 로드 (캐싱)"""
    progress_bar = st.progress(0)
    data = download_price_data(tickers, start, end, progress_bar)
    return data

@st.cache_data
def run_daa_backtest(
    price_data,
    risky_tickers,
    canary_tickers,
    cash_tickers,
    breadth_param,
    top_select
):
    """DAA 백테스트 실행 (캐싱)"""
    config = DAAConfig(
        risky_tickers=risky_tickers,
        canary_tickers=canary_tickers,
        cash_tickers=cash_tickers,
        breadth_parameter=breadth_param,
        top_selection=top_select
    )
    
    daa = DAAStrategy(config).fit(price_data)
    return daa

# ===== 메인 앱 =====
def main():
    # 헤더
    st.markdown('<div class="header">📊 DAA (Defensive Asset Allocation) 백테스트 시스템</div>', 
                unsafe_allow_html=True)
    
    st.write("""
    Wouter J. Keller와 Jan Willem Keuning의 논문을 기반한 카나리 유니버스 동적 자산배분 전략
    
    **핵심 개념**: VWO(신흥시장)와 BND(본드)의 모멘텀으로 Crash Protection 신호 생성
    """)
    
    # ===== 사이드바 설정 =====
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 기간 설정
        st.subheader("📅 분석 기간")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "시작 날짜",
                value=datetime(2015, 1, 1),
                min_value=datetime(2005, 1, 1)
            )
        with col2:
            end_date = st.date_input(
                "종료 날짜",
                value=datetime.now(),
                min_value=start_date
            )
        
        # 자산 유니버스 선택
        st.subheader("🎯 자산 유니버스")
        universe_choice = st.radio(
            "위험자산 선택",
            options=[
                "DAA-G12 (12개 글로벌 자산)",
                "DAA-U6 (6개 US 자산)",
                "DAA-G4 (4개 자산)",
                "커스텀"
            ],
            index=0
        )
        
        # 자산 할당
        if universe_choice == "DAA-G12 (12개 글로벌 자산)":
            risky_tickers = list(RISKY_UNIVERSE_G12.keys())
        elif universe_choice == "DAA-U6 (6개 US 자산)":
            risky_tickers = list(RISKY_UNIVERSE_U6.keys())
        elif universe_choice == "DAA-G4 (4개 자산)":
            risky_tickers = list(RISKY_UNIVERSE_G4.keys())
        else:
            custom_input = st.text_input(
                "티커 입력 (쉼표로 구분)",
                value="SPY,VEA,VWO,QQQ"
            )
            risky_tickers = [t.strip() for t in custom_input.split(",")]
        
        canary_tickers = list(CANARY_UNIVERSE.keys())
        cash_tickers = list(CASH_UNIVERSE.keys())
        
        # DAA 파라미터
        st.subheader("⚡ DAA 파라미터")
        breadth_param = st.radio(
            "Breadth Parameter (B)",
            options=[2, 1],
            format_func=lambda x: f"B={x} ({'기본' if x==2 else '공격적'})",
            index=0
        )
        
        top_select = st.slider(
            "상위 N개 선택 (Top T)",
            min_value=1,
            max_value=len(risky_tickers),
            value=min(6, len(risky_tickers))
        )
        
        # 벤치마크 선택
        st.subheader("🏆 벤치마크")
        benchmark_choice = st.selectbox(
            "비교 대상",
            options=list(BENCHMARKS.keys()) + ["None"]
        )
    
    # ===== 메인 콘텐츠 =====
    # 1. 데이터 로드
    st.markdown("### 📥 데이터 로드 중...")
    
    try:
        # 모든 필요한 티커
        all_tickers = list(set(risky_tickers + canary_tickers + cash_tickers))
        
        with st.spinner("yfinance에서 데이터를 다운로드 중입니다..."):
            price_data = load_price_data(
                all_tickers,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        st.success(f"✅ {len(price_data)} 일 × {len(price_data.columns)} 자산 데이터 로드 완료")
        
        # 2. DAA 백테스트 실행
        st.markdown("### 🔄 DAA 백테스트 실행 중...")
        
        with st.spinner("DAA 전략을 실행 중입니다..."):
            daa_strategy = run_daa_backtest(
                price_data,
                risky_tickers,
                canary_tickers,
                cash_tickers,
                breadth_param,
                top_select
            )
        
        st.success("✅ DAA 백테스트 완료")
        
        # 3. 성과 지표 탭
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📈 성과 비교", "📊 신호 분석", "🎯 가중치", "📋 월별 수익률", "ℹ️ 정보"]
        )
        
        # ===== 탭 1: 성과 비교 =====
        with tab1:
            st.subheader("전략 성과 지표")
            
            # 벤치마크 계산
            benchmark_returns = None
            if benchmark_choice != "None":
                benchmark_weights = BENCHMARKS[benchmark_choice]
                
                if benchmark_weights['stocks'] > 0:
                    stock_prices = price_data['SPY']  # SPY를 주식 대표로 사용
                else:
                    stock_prices = None
                
                if benchmark_weights['bonds'] > 0:
                    bond_prices = price_data['BND']  # BND를 채권 대표로 사용
                else:
                    bond_prices = None
                
                # 월별 수익률
                if stock_prices is not None and bond_prices is not None:
                    stock_monthly = stock_prices.resample('M').last().pct_change()
                    bond_monthly = bond_prices.resample('M').last().pct_change()
                    
                    benchmark_returns = (
                        stock_monthly * benchmark_weights['stocks'] +
                        bond_monthly * benchmark_weights['bonds']
                    )
            
            # DAA 백테스트
            daa_backtest = DAABacktest(daa_strategy)
            daa_performance = daa_backtest.run(benchmark_returns)
            
            # 성과 지표 표시
            col1, col2, col3, col4 = st.columns(4)
            
            metrics_list = [
                ('CAGR (%)', 'CAGR (%)', col1),
                ('Volatility (%)', 'Volatility (%)', col2),
                ('Sharpe Ratio', 'Sharpe Ratio', col3),
                ('Max Drawdown (%)', 'Max Drawdown (%)', col4)
            ]
            
            for display_name, metric_key, col in metrics_list:
                if metric_key in daa_performance:
                    value = daa_performance[metric_key]
                    if '%' in display_name:
                        col.metric(display_name, f"{value:.2f}%")
                    else:
                        col.metric(display_name, f"{value:.4f}")
            
            # 추가 지표
            col5, col6, col7 = st.columns(3)
            
            for col, (name, key) in zip(
                [col5, col6, col7],
                [('승률', 'Win Rate (%)'), ('총 수익', 'Total Return (%)'), ('월간 변동성', 'Monthly Volatility (%)')]
            ):
                if key in daa_performance:
                    value = daa_performance[key]
                    col.metric(name, f"{value:.2f}%")
            
            # 누적 수익률 그래프
            st.markdown("#### 누적 수익률 추이")
            
            cum_returns_daa = daa_strategy.get_cumulative_returns() * 100
            
            fig = go.Figure()
            
            # DAA
            fig.add_trace(go.Scatter(
                x=cum_returns_daa.index,
                y=cum_returns_daa.values,
                mode='lines',
                name='DAA',
                line=dict(color=COLORS['daa'], width=3)
            ))
            
            # 벤치마크
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
                title="누적 수익률 비교",
                xaxis_title="날짜",
                yaxis_title="누적 수익률 (%)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown 차트
            st.markdown("#### 낙폭 (Drawdown) 추이")
            
            cum_returns = (1 + daa_strategy.get_returns()).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown_daa = (cum_returns / running_max - 1) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=drawdown_daa.index,
                y=drawdown_daa.values,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color=COLORS['negative'])
            ))
            
            fig.update_layout(
                title="DAA 낙폭 추이",
                xaxis_title="날짜",
                yaxis_title="낙폭 (%)",
                hovermode='x',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ===== 탭 2: 신호 분석 =====
        with tab2:
            st.subheader("DAA 신호 분석")
            
            signals = daa_strategy.get_signals()
            
            # 카나리 유니버스 모멘텀
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### VWO (신흥시장) 모멘텀")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=signals.index,
                    y=signals['VWO_mom'].values,
                    mode='lines',
                    name='VWO Momentum',
                    fill='tozeroy',
                    line=dict(color=COLORS['canary_good'])
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### BND (본드) 모멘텀")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=signals.index,
                    y=signals['BND_mom'].values,
                    mode='lines',
                    name='BND Momentum',
                    fill='tozeroy',
                    line=dict(color=COLORS['canary_good'])
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # Cash Fraction
            st.markdown("#### 현금 비율 (Cash Fraction)")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=signals.index,
                y=signals['Cash_Fraction'].values * 100,
                fill='tozeroy',
                name='Cash Fraction',
                line=dict(color=COLORS['daa'])
            ))
            fig.update_layout(
                title="동적 현금 배분 비율",
                xaxis_title="날짜",
                yaxis_title="현금 비율 (%)",
                hovermode='x',
                height=400,
                template='plotly_white',
                yaxis=dict(range=[0, 105])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 신호 데이터 테이블
            st.markdown("#### 신호 데이터 (최근 20개월)")
            
            display_signals = signals[['VWO_mom', 'BND_mom', 'Breadth_Bad_Count', 'Cash_Fraction']].tail(20)
            display_signals.columns = ['VWO 모멘텀', 'BND 모멘텀', 'Bad 개수', '현금 비율']
            
            st.dataframe(display_signals.style.format({
                'VWO 모멘텀': '{:.4f}',
                'BND 모멘텀': '{:.4f}',
                'Bad 개수': '{:.0f}',
                '현금 비율': '{:.2%}'
            }), use_container_width=True)
        
        # ===== 탭 3: 포트폴리오 가중치 =====
        with tab3:
            st.subheader("포트폴리오 가중치 분석")
            
            weights = daa_strategy.get_weights()
            
            # 최근 가중치
            st.markdown("#### 최근 10개월 가중치")
            
            recent_weights = weights.tail(10) * 100
            
            fig = go.Figure()
            
            for col in recent_weights.columns:
                fig.add_trace(go.Bar(
                    x=recent_weights.index,
                    y=recent_weights[col],
                    name=col
                ))
            
            fig.update_layout(
                barmode='stack',
                title="월별 포트폴리오 구성",
                xaxis_title="날짜",
                yaxis_title="가중치 (%)",
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 가중치 히트맵
            st.markdown("#### 가중치 히트맵 (최근 1년)")
            
            heatmap_data = weights.tail(12) * 100
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.T.values,
                x=heatmap_data.index,
                y=heatmap_data.columns,
                colorscale='YlOrRd'
            ))
            
            fig.update_layout(
                title="자산별 월별 가중치",
                xaxis_title="날짜",
                yaxis_title="자산",
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 평균 가중치
            st.markdown("#### 평균 가중치")
            
            avg_weights = weights.mean() * 100
            
            fig = px.bar(
                x=avg_weights.index,
                y=avg_weights.values,
                labels={'y': '평균 가중치 (%)'},
                color=avg_weights.values,
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                title="전체 기간 평균 포트폴리오 가중치",
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ===== 탭 4: 월별 수익률 =====
        with tab4:
            st.subheader("월별 수익률 분석")
            
            monthly_returns = daa_strategy.get_returns()
            
            # 월별 수익률 히트맵
            st.markdown("#### 월별 수익률 히트맵")
            
            # 연도별, 월별로 정렬
            monthly_returns_df = monthly_returns.to_frame('Return')
            monthly_returns_df['Year'] = monthly_returns_df.index.year
            monthly_returns_df['Month'] = monthly_returns_df.index.month
            
            pivot_returns = monthly_returns_df.pivot_table(
                values='Return',
                index='Year',
                columns='Month',
                aggfunc='first'
            ) * 100
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_returns.values,
                x=['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'],
                y=pivot_returns.index,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(pivot_returns.values, 2),
                texttemplate='%{text}%',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="월별 수익률 (%)",
                xaxis_title="월",
                yaxis_title="연도",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 월별 수익률 통계
            st.markdown("#### 월별 수익률 통계")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            monthly_returns_pct = monthly_returns * 100
            
            col1.metric("평균 월 수익률", f"{monthly_returns_pct.mean():.2f}%")
            col2.metric("중간값", f"{monthly_returns_pct.median():.2f}%")
            col3.metric("최대 월 수익률", f"{monthly_returns_pct.max():.2f}%")
            col4.metric("최소 월 수익률", f"{monthly_returns_pct.min():.2f}%")
            col5.metric("월간 변동성", f"{monthly_returns_pct.std():.2f}%")
            
            # 수익률 분포
            st.markdown("#### 수익률 분포")
            
            fig = px.histogram(
                monthly_returns_pct,
                nbins=30,
                labels={'value': '월별 수익률 (%)'},
                title="월별 수익률 분포"
            )
            
            fig.update_layout(
                height=400,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ===== 탭 5: 정보 =====
        with tab5:
            st.subheader("📋 전략 정보")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 위험자산 유니버스")
                st.write({
                    '자산 개수': len(risky_tickers),
                    '티커': ", ".join(risky_tickers)
                })
                
                st.markdown("#### 카나리 유니버스")
                st.write({
                    '자산 개수': len(canary_tickers),
                    '티커': ", ".join(canary_tickers)
                })
            
            with col2:
                st.markdown("#### 안전자산 유니버스")
                st.write({
                    '자산 개수': len(cash_tickers),
                    '티커': ", ".join(cash_tickers)
                })
                
                st.markdown("#### DAA 파라미터")
                st.write({
                    'Breadth Parameter (B)': breadth_param,
                    'Top Selection (T)': top_select,
                    'Momentum 기간': [1, 3, 6, 12],
                    'Momentum 가중치': [12, 4, 2, 1]
                })
            
            st.markdown("#### 📚 참고 자료")
            st.write("""
            - **논문**: Keller, W.J., Keuning, J.W. (2018)
            - **제목**: "Breadth Momentum and the Canary Universe: Defensive Asset Allocation (DAA)"
            - **출처**: SSRN 3212862
            - **링크**: https://ssrn.com/abstract=3212862
            
            #### 📖 주요 개념
            
            **카나리 유니버스**: VWO(신흥시장 ETF)와 BND(채권 ETF)로 구성
            - VWO: 신흥시장의 민감성을 나타내는 지표
            - BND: 금리 상승 및 신용 위험 신호
            
            **Breadth Momentum**: 카나리 유니버스의 Bad 자산 개수
            - b=0: 안전 신호 → 100% 위험자산
            - b=1: 경고 신호 → 50/50 배분
            - b=2: 위험 신호 → 100% 안전자산
            
            **Momentum 계산**: 1, 3, 6, 12개월 리턴의 가중평균 (12612W)
            """)
    
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.info("다시 시도하거나 설정을 확인해주세요.")

if __name__ == "__main__":
    main()
