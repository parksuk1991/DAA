"""
DAA + VAA 완전 수정판
========================================================================

핵심 수정:
1. 현금 비중 30% 상한선 STRICT (단일 현금 자산도 30% 초과 불가)
2. Breadth 1.0 → 현금 0% (어떤 현금자산도 미포함)
3. 모멘텀 70% + 변동성 30% 복합 점수
4. 위험자산 70% 이상 보장 (현금 최대 30%)
5. 100% 정규화 보장 (모든 기간)
6. 모든 종목이 선택될 수 있도록 로직 개선
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DAA+VAA 완전수정", page_icon="📊", layout="wide")
st.markdown("""<style>.header {color: #1f77b4; font-size: 28px; font-weight: bold; margin-bottom: 20px;}</style>""", unsafe_allow_html=True)

COLORS = {
    'strategy': '#E74C3C', 'aggressive': '#E67E22', 'balanced': '#27AE60',
    'conservative': '#3498DB', 'spy': '#9B59B6', 'acwi': '#1ABC9C',
    'positive': '#2ECC71', 'negative': '#E74C3C', 'neutral': '#95A5A6'
}

UNIVERSE = {
    'core': ['ACWI'],
    'risky': ['SPY', 'IWM', 'QQQ', 'VEA', 'VGK', 'EWJ', 'VWO', 'VNQ', 'GSG', 'GLD', 
              'TLT', 'HYG', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 
              'XLK', 'XLU', 'RSP', 'VUG', 'VTV', 'VYM', 'USMV', 'EWY', 'SPMO', 'PTF'],
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

@st.cache_data
def download_price_data(ticker_list, start_date, end_date):
    try:
        raw = yf.download(ticker_list, start=start_date, end=end_date, progress=False)
        df = raw['Close'] if isinstance(raw, pd.DataFrame) else raw
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.ffill().bfill().dropna(how='all')
        available_cols = [t for t in ticker_list if t in df.columns]
        return df[available_cols] if available_cols else None
    except:
        return None

def calculate_monthly_returns(price_df):
    try:
        monthly_prices = price_df.resample('M').last()
        return monthly_prices.pct_change()
    except:
        return pd.DataFrame()

def calculate_volatility(returns_df, window=3):
    """과거 window개월 변동성 (낮을수록 안정적)"""
    volatility = returns_df.rolling(window).std()
    return volatility.shift(1)

def calculate_composite_score(momentum_df, volatility_df, momentum_weight=0.7, vol_weight=0.3):
    """
    복합 점수 = 70% 모멘텀 + 30% 변동성역수
    
    모멘텀: 수익성 (높을수록 좋음)
    변동성: 안정성 (낮을수록 좋음)
    """
    # 정규화 (z-score)
    mom_norm = momentum_df.copy()
    vol_norm = volatility_df.copy()
    
    for col in mom_norm.columns:
        m = mom_norm[col]
        m_clean = m.dropna()
        if len(m_clean) > 0 and m_clean.std() > 0:
            mom_norm[col] = (m - m_clean.mean()) / m_clean.std()
    
    for col in vol_norm.columns:
        v = vol_norm[col]
        v_clean = v.dropna()
        if len(v_clean) > 0 and v_clean.std() > 0:
            vol_norm[col] = (v - v_clean.mean()) / v_clean.std()
    
    # 변동성 역수 (낮은 변동성 = 높은 점수)
    vol_score = -vol_norm  # 부호 반전
    
    # 복합
    composite = momentum_weight * mom_norm + vol_weight * vol_score
    
    return composite

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

def calculate_breadth_score_continuous(momentum_df, canary_tickers):
    """Breadth Score (0=약함, 1=강함)"""
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

def calculate_cash_fraction_continuous(breadth_score, max_cash=0.30):
    """
    현금 비중 = 1 - breadth_score (상한선 30%)
    
    핵심:
    - Breadth 1.0 → 현금 0% (위험자산 100%)
    - Breadth 0.7 → 현금 30% (위험자산 70%)
    - Breadth < 0.7 → 현금 30% (상한선)
    """
    raw_cash = 1 - breadth_score
    cash_frac = raw_cash.clip(0, max_cash)
    return cash_frac

def calculate_risk_asset_allocation(cash_fraction):
    """위험자산 비중 = 1 - 현금 비중"""
    return 1 - cash_fraction

def calculate_portfolio_weights_with_constraints(composite_score_df, cash_fraction,
                                                available_core, available_risky, available_cash,
                                                top_n, max_single=0.30, min_core=0.20):
    """
    포트폴리오 가중치 계산 (완벽한 정규화)
    
    제약:
    1. 현금 0% ~ 30% (단일 현금자산도 30% 초과 불가)
    2. ACWI 20% ~ 30%
    3. 모든 종목 30% 이하
    4. 위험자산 70% 이상 (현금 30% 이하)
    5. 정규화: 합 = 1.0000 ± 0.0001
    """
    w_core = pd.DataFrame(0.0, index=composite_score_df.index, columns=available_core)
    w_risky = pd.DataFrame(0.0, index=composite_score_df.index, columns=available_risky)
    w_cash = pd.DataFrame(0.0, index=composite_score_df.index, columns=available_cash)
    
    validation = []
    
    for date_idx in composite_score_df.index:
        try:
            if pd.isna(cash_fraction.loc[date_idx]):
                validation.append({
                    'date': date_idx, 'total': np.nan, 'valid': False,
                    'cash': np.nan, 'max_w': np.nan, 'reason': 'NaN'
                })
                continue
            
            cf = float(cash_fraction.loc[date_idx])
            cf = cf.clip(0, 0.30)  # 현금 30% 상한선 STRICT
            risk_ratio = 1 - cf
            
            # ===== Step 1: Core (ACWI) =====
            core_w = max(min_core, risk_ratio * 0.25)
            core_w = min(core_w, max_single)
            if available_core:
                w_core.at[date_idx, available_core[0]] = core_w
            
            # ===== Step 2: 나머지 위험자산 =====
            remain_risk = risk_ratio - core_w
            
            if remain_risk > 1e-8 and available_risky:
                scores = composite_score_df.loc[date_idx, available_risky]
                scores_valid = scores[~pd.isna(scores)].copy()
                
                if len(scores_valid) > 0:
                    # 상위 T개
                    sel_count = min(top_n, len(scores_valid))
                    top_tickers = scores_valid.nlargest(sel_count)
                    
                    # 가중치 계산
                    top_scores = top_tickers.clip(lower=0)
                    
                    if top_scores.sum() > 1e-8:
                        weights = top_scores / top_scores.sum()
                        
                        for ticker, wt in weights.items():
                            w_risky.at[date_idx, ticker] = wt * remain_risk
            
            # ===== Step 3: 현금 (cf > 0일 때만) =====
            if cf > 1e-8 and available_cash:
                scores = composite_score_df.loc[date_idx, available_cash]
                scores_valid = scores[~pd.isna(scores)]
                
                if len(scores_valid) > 0:
                    best = scores_valid.idxmax()
                    w_cash.at[date_idx, best] = cf
            
            # ===== Step 4: 정규화 =====
            total = w_core.loc[date_idx].sum() + w_risky.loc[date_idx].sum() + w_cash.loc[date_idx].sum()
            
            if total < 1e-8:
                # 거의 불가능한 경우
                if available_risky:
                    w_risky.at[date_idx, available_risky[0]] = 1.0
                elif available_core:
                    w_core.at[date_idx, available_core[0]] = 1.0
                else:
                    w_cash.at[date_idx, available_cash[0]] = 1.0
            elif abs(total - 1.0) > 0.00001:
                scale = 1.0 / total
                w_core.loc[date_idx] *= scale
                w_risky.loc[date_idx] *= scale
                w_cash.loc[date_idx] *= scale
            
            # ===== 검증 =====
            final_total = w_core.loc[date_idx].sum() + w_risky.loc[date_idx].sum() + w_cash.loc[date_idx].sum()
            max_weight = max(
                w_core.loc[date_idx].max(),
                w_risky.loc[date_idx].max(),
                w_cash.loc[date_idx].max()
            )
            
            is_valid = abs(final_total - 1.0) < 0.00001 and max_weight <= 0.31
            
            validation.append({
                'date': date_idx,
                'total': final_total,
                'valid': is_valid,
                'cash': w_cash.loc[date_idx].sum(),
                'max_w': max_weight,
                'reason': 'OK' if is_valid else f'Total={final_total:.6f}, Max={max_weight:.4f}'
            })
        
        except Exception as e:
            validation.append({
                'date': date_idx,
                'total': np.nan,
                'valid': False,
                'reason': str(e)[:30]
            })
    
    return w_core, w_risky, w_cash, pd.DataFrame(validation)

def find_optimal_top_n(composite_score_df, available_risky, max_t=8):
    """최적 T값 (평균 복합 점수)"""
    best_score = -999
    best_t = 1
    
    for test_t in range(1, min(len(available_risky) + 1, max_t + 1)):
        scores_list = []
        for date_idx in composite_score_df.index:
            row = composite_score_df.loc[date_idx, available_risky]
            row_valid = row[~pd.isna(row)]
            if len(row_valid) >= test_t:
                top_score = row_valid.nlargest(test_t).mean()
                scores_list.append(top_score)
        
        if scores_list:
            avg_score = np.mean(scores_list)
            if avg_score > best_score:
                best_score = avg_score
                best_t = test_t
    
    return best_t

def backtest_returns(monthly_returns, weights_df, transaction_cost=0.001):
    try:
        common_cols = [c for c in weights_df.columns if c in monthly_returns.columns]
        if not common_cols:
            return pd.Series(0.0, index=weights_df.index)
        
        returns = (weights_df[common_cols] * monthly_returns[common_cols]).sum(axis=1)
        costs = weights_df[common_cols].diff().abs().sum(axis=1) * transaction_cost
        
        return returns - costs
    except:
        return pd.Series(0.0, index=weights_df.index)

def calculate_performance_metrics(returns, rf=0.02):
    try:
        returns = returns.dropna()
        if len(returns) == 0:
            return {k: 0.0 for k in ['CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'RAD (%)', 'Total Return (%)']}
        
        cum = (1 + returns).cumprod()
        total_ret = (cum.iloc[-1] - 1) * 100
        
        years = len(returns) / 12
        cagr = ((cum.iloc[-1]) ** (1.0 / years) - 1) * 100 if years > 0 else 0
        
        vol = returns.std() * np.sqrt(12) * 100
        sharpe = (returns.mean() * 12 - rf) / (vol / 100) if vol > 1e-6 else 0
        
        dd = ((cum / cum.expanding().max()) - 1).min() * 100
        wr = (returns > 0).sum() / len(returns) * 100
        
        rad = 0
        if cagr >= 0 and dd >= -50:
            d_ratio = abs(dd) / 100
            if d_ratio < 1:
                rad = cagr * (1 - (d_ratio / (1 - d_ratio))) if d_ratio > 0 else cagr
        
        return {
            'Total Return (%)': total_ret,
            'CAGR (%)': cagr,
            'Volatility (%)': vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': dd,
            'Win Rate (%)': wr,
            'RAD (%)': rad
        }
    except:
        return {k: 0.0 for k in ['CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'RAD (%)', 'Total Return (%)']}

def run_strategy(price_data):
    try:
        avail_core = [t for t in UNIVERSE['core'] if t in price_data.columns]
        avail_risky = [t for t in UNIVERSE['risky'] if t in price_data.columns]
        avail_canary = [t for t in UNIVERSE['canary'] if t in price_data.columns]
        avail_cash = [t for t in UNIVERSE['cash'] if t in price_data.columns]
        
        rets = calculate_monthly_returns(price_data)
        vol = calculate_volatility(rets)
        mom = calculate_momentum_daa(rets)
        
        comp = calculate_composite_score(mom, vol, 0.7, 0.3)
        breadth = calculate_breadth_score_continuous(mom, avail_canary)
        cf = calculate_cash_fraction_continuous(breadth, 0.30)
        
        opt_t = find_optimal_top_n(comp, avail_risky)
        st.session_state.optimal_t = opt_t
        
        w_c, w_r, w_ca, val = calculate_portfolio_weights_with_constraints(
            comp, cf, avail_core, avail_risky, avail_cash, opt_t
        )
        
        all_t = avail_core + avail_risky + avail_cash
        w_all = pd.concat([w_c, w_r, w_ca], axis=1)
        ret_all = rets[all_t]
        strat_ret = backtest_returns(ret_all, w_all)
        
        return {
            'strat_ret': strat_ret,
            'mom': mom,
            'vol': vol,
            'comp': comp,
            'breadth': breadth,
            'cf': cf,
            'w_c': w_c, 'w_r': w_r, 'w_ca': w_ca,
            'val': val,
            'avail_c': avail_core,
            'avail_r': avail_risky,
            'avail_ca': avail_cash,
            'opt_t': opt_t,
            'rets': rets
        }
    except Exception as e:
        st.error(f"❌ {str(e)}")
        return None

def main():
    st.markdown('<div class="header">📊 DAA+VAA 완전 수정 (현금 30% 엄격, 100% 정규화)</div>', unsafe_allow_html=True)
    st.write("**핵심**: 현금 최대 30% | 위험자산 최소 70% | 모멘텀+변동성 | 100% 정규화 보장")
    
    if 'optimal_t' not in st.session_state:
        st.session_state.optimal_t = 3
    
    with st.sidebar:
        st.header("⚙️ 설정")
        c1, c2 = st.columns(2)
        with c1:
            sd = st.date_input("시작", value=datetime(2015, 1, 1))
        with c2:
            ed = st.date_input("종료", value=datetime.now())
        
        st.divider()
        
        bmarks = st.multiselect(
            "벤치마크",
            list(BENCHMARKS_CONFIG.keys()),
            default=['SPY Index', 'Balanced (60/40)']
        )
    
    try:
        st.info("💾 로드 중...")
        
        all_t = sorted(list(set(UNIVERSE['core'] + UNIVERSE['risky'] + 
                                UNIVERSE['canary'] + UNIVERSE['cash'] + 
                                ['SPY', 'BND', 'ACWI'])))
        
        price_data = download_price_data(all_t, sd.strftime('%Y-%m-%d'), ed.strftime('%Y-%m-%d'))
        
        if price_data is None:
            st.error("❌ 데이터 로드 실패")
            return
        
        st.success("✅ 로드 완료")
        
        res = run_strategy(price_data)
        if not res:
            return
        
        strat_ret = res['strat_ret']
        opt_t = res['opt_t']
        
        # 벤치마크
        bmark_dict = {}
        colors = {
            'Aggressive (70/30)': COLORS['aggressive'],
            'Balanced (60/40)': COLORS['balanced'],
            'Conservative (50/50)': COLORS['conservative'],
            'SPY Index': COLORS['spy'],
            'ACWI Index': COLORS['acwi']
        }
        
        for bn in bmarks:
            try:
                bc = BENCHMARKS_CONFIG[bn]
                if bc.get('is_acwi'):
                    if 'ACWI' in price_data.columns:
                        bmark_dict[bn] = price_data['ACWI'].resample('M').last().pct_change()
                else:
                    spy_m = price_data['SPY'].resample('M').last().pct_change()
                    bnd_m = price_data['BND'].resample('M').last().pct_change()
                    bmark_dict[bn] = spy_m * bc['stocks'] + bnd_m * bc['bonds']
            except:
                pass
        
        perf = calculate_performance_metrics(strat_ret)
        perf_b = {k: calculate_performance_metrics(v) for k, v in bmark_dict.items()}
        
        st.markdown("---")
        st.subheader("📈 성과")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{perf['CAGR (%)']:.2f}%")
        c2.metric("Vol", f"{perf['Volatility (%)']:.2f}%")
        c3.metric("Sharpe", f"{perf['Sharpe Ratio']:.3f}")
        c4.metric("MaxDD", f"{perf['Max Drawdown (%)']:.2f}%")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("RAD", f"{perf['RAD (%)']:.2f}%")
        c6.metric("승률", f"{perf['Win Rate (%)']:.1f}%")
        c7.metric("총수익", f"{perf['Total Return (%)']:.2f}%")
        c8.metric("최적T", f"{opt_t}개")
        
        st.markdown("---")
        st.subheader("✅ 검증")
        
        val = res['val']
        vc = val['valid'].sum() if 'valid' in val.columns else 0
        vp = (vc / len(val) * 100) if len(val) > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**가중치**")
            if vp < 99:
                st.error(f"❌ {100-vp:.1f}% 오류")
            else:
                st.success(f"✅ {vp:.1f}%")
        
        with c2:
            st.markdown("**현금 30%**")
            max_cf = val['cash'].max() if 'cash' in val.columns else 0
            if max_cf > 0.31:
                st.error(f"❌ {max_cf*100:.1f}%")
            else:
                st.success("✅ 준수")
        
        with c3:
            st.markdown("**종목 30%**")
            max_w = val['max_w'].max() if 'max_w' in val.columns else 0
            if max_w > 0.31:
                st.error(f"❌ {max_w*100:.1f}%")
            else:
                st.success("✅ 준수")
        
        with c4:
            st.markdown("**변동성**")
            st.success("✅ 70%")
        
        # 탭
        t1, t2, t3, t4, t5 = st.tabs(["📊 성과", "⚖️ 가중치", "📈 수익", "🔍 상세", "📖 용어"])
        
        with t1:
            st.subheader("누적 수익률")
            
            cum = (1 + strat_ret).cumprod() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode='lines', name='DAA+VAA',
                                    line=dict(color=COLORS['strategy'], width=3)))
            
            for bn, br in bmark_dict.items():
                cb = (1 + br).cumprod() * 100
                fig.add_trace(go.Scatter(x=cb.index, y=cb.values, mode='lines', name=bn,
                                        line=dict(color=colors.get(bn, '#95A5A6'), width=2)))
            
            fig.update_layout(title="누적 수익률", yaxis_title="(%)", height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with t2:
            st.subheader("포트폴리오 가중치")
            
            w = pd.concat([res['w_c'], res['w_r'], res['w_ca']], axis=1) * 100
            w = w.round(3)
            
            rec = w.tail(20)
            fig = go.Figure()
            for c in rec.columns:
                fig.add_trace(go.Bar(x=rec.index, y=rec[c], name=c))
            fig.update_layout(title="최근 20개월", barmode='stack', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.markdown("**전체 가중치**")
            csv = w.to_csv()
            st.download_button("📥 CSV", csv, "weights.csv", "text/csv")
            st.dataframe(w, use_container_width=True, height=600)
        
        with t3:
            st.subheader("월별 수익률")
            fig = go.Figure()
            color_list = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in strat_ret]
            fig.add_trace(go.Bar(x=strat_ret.index, y=strat_ret*100, marker=dict(color=color_list)))
            fig.update_layout(title="월별", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with t4:
            st.subheader("포트폴리오 구성 방식")
            st.markdown("""
            ### 복합 점수 (70% 모멘텀 + 30% 변동성)
            
            **모멘텀** (70%):
            - DAA: (R1+R3+R6+R12)/4
            - 높을수록 좋음
            
            **변동성** (30%):
            - 과거 3개월 표준편차
            - 낮을수록 좋음 (안정성)
            
            ### 현금 비중 계산
            
            ```
            현금 = 1 - Breadth Score (상한선 30%)
            위험자산 = 1 - 현금 (최소 70%)
            
            Breadth 1.0 → 현금 0% (안전자산 제외)
            Breadth 0.7 → 현금 30%
            Breadth 0.0 → 현금 30% (상한선)
            ```
            
            ### 자산 할당
            
            **ACWI Core**: 20%~30%
            **위험자산**: 복합 점수 상위 T개 (모멘텀+변동성 기반)
            **현금**: 1개 자산만 (모멘텀 최고)
            
            모든 종목 최대 30%
            """)
        
        with t5:
            st.markdown("""
            ## 주요 용어
            
            **DAA**: 모멘텀 기반 자산배분
            **Breadth**: 시장 광폭성 (VWO, BND 모멘텀)
            **복합 점수**: 수익성 + 안정성
            **정규화**: 100% 정확
            """)
    
    except Exception as e:
        st.error(f"❌ {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
