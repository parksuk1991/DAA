"""
DAA 전략 설정
"""

# ===== 기본 자산 유니버스 =====

# 글로벌 12개 자산 유니버스 (DAA-G12)
RISKY_UNIVERSE_G12 = {
    'SPY': 'US Large Cap',
    'IWM': 'US Small Cap',
    'QQQ': 'US Tech',
    'VEA': 'International Developed',
    'VGK': 'Europe',
    'EWJ': 'Japan',
    'VWO': 'Emerging Markets',
    'VNQ': 'US REIT',
    'GSG': 'Commodities',
    'GLD': 'Gold',
    'TLT': 'Long-term Bonds',
    'HYG': 'High Yield Bonds'
}

# US 6개 자산 유니버스 (DAA-U6)
RISKY_UNIVERSE_U6 = {
    'VTV': 'US Large Value',
    'VUG': 'US Large Growth',
    'VBV': 'US Small Value',
    'VBR': 'US Small Growth',
    'BIL': 'Short-term Bonds',
    'LQD': 'Investment Grade Bonds'
}

# Simple 4개 자산 유니버스 (DAA-G4)
RISKY_UNIVERSE_G4 = {
    'SPY': 'US Equities',
    'VEA': 'International Equities',
    'VWO': 'Emerging Markets',
    'BND': 'Bonds'
}

# ===== 카나리 유니버스 =====
CANARY_UNIVERSE = {
    'VWO': 'Emerging Markets Canary',
    'BND': 'Bonds Canary'
}

# ===== 안전자산 (Cash) 유니버스 =====
CASH_UNIVERSE = {
    'SHY': 'Short-term Treasury (1-3y)',
    'IEF': 'Intermediate Treasury (7-10y)',
    'LQD': 'Investment Grade Bonds'
}

# 공격적 현금 유니버스 (DAA Aggressive)
CASH_UNIVERSE_AGGRESSIVE = {
    'SHV': 'Ultra Short Treasury (1-12m)',
    'IEF': 'Intermediate Treasury (7-10y)',
    'UST': 'Leveraged Treasury (2x)'
}

# ===== DAA 전략 파라미터 =====

# 기본 DAA 파라미터
DAA_PARAMS = {
    'momentum_periods': [1, 3, 6, 12],  # 월 단위
    'momentum_weights': [12, 4, 2, 1],  # 13612W 가중치
    'breadth_parameter': 2,              # B=2 (기본), B=1 (공격적)
    'top_selection': 6,                  # T=6 (상위 6개 선택)
    'transaction_cost': 0.001,           # 거래 비용 0.1%
    'rebalance_freq': 'M'                # 월별 리밸런싱
}

# 공격적 DAA 파라미터 (DAA1)
DAA_AGGRESSIVE_PARAMS = {
    'momentum_periods': [1, 3, 6, 12],
    'momentum_weights': [12, 4, 2, 1],
    'breadth_parameter': 1,              # B=1 (공격적 crash protection)
    'top_selection': 2,                  # T=2
    'transaction_cost': 0.001,
    'rebalance_freq': 'M'
}

# ===== 벤치마크 포트폴리오 =====
BENCHMARKS = {
    'Aggressive 70/30': {'stocks': 0.7, 'bonds': 0.3},
    'Balanced 60/40': {'stocks': 0.6, 'bonds': 0.4},
    'Conservative 50/50': {'stocks': 0.5, 'bonds': 0.5},
    'Equal Weight': {'stocks': 0.5, 'bonds': 0.5},
    'Buy & Hold SPY': {'stocks': 1.0, 'bonds': 0.0}
}

# ===== 성과 지표 =====
PERFORMANCE_METRICS = [
    'Total Return',
    'CAGR',
    'Volatility',
    'Sharpe Ratio',
    'Sortino Ratio',
    'Max Drawdown',
    'Win Rate',
    'Avg Cash Ratio'
]

# ===== 색상 설정 =====
COLORS = {
    'daa': '#FF6B6B',           # 빨강
    'benchmark': '#4ECDC4',     # 청록
    'canary_good': '#95E1D3',   # 연한 초록
    'canary_bad': '#F7B731',    # 주황
    'positive': '#2ECC71',      # 초록
    'negative': '#E74C3C'       # 빨강
}

# ===== 데이터 설정 =====
DATA_CONFIG = {
    'cache_dir': './data/cache/',
    'data_source': 'yfinance',
    'download_timeout': 30,
    'max_retries': 3
}

# ===== 분석 기간 =====
ANALYSIS_PERIODS = {
    'IS': ('2000-01-01', '2010-12-31'),    # In-Sample (학습 기간)
    'OS': ('2011-01-01', '2020-12-31'),    # Out-of-Sample (검증 기간)
    'RS': ('2020-01-01', None),             # Recent Sample (최근)
}
