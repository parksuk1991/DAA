# DAA (Defensive Asset Allocation) Backtest System

Wouter J. Keller and Jan Willem Keuning의 "Breadth Momentum and the Canary Universe: Defensive Asset Allocation (DAA)" 논문을 기반으로 한 백테스트 시스템입니다.

## 🎯 주요 특징

- **카나리 유니버스**: VWO (신흥시장) + BND (본드)를 이용한 crash protection
- **Breadth Momentum**: 위험자산의 모멘텀 추적으로 자동 리스크 관리
- **월별 리밸런싱**: 매월 자동으로 자산 배분 조정
- **Streamlit UI**: 직관적인 대시보드로 결과 시각화
- **실시간 백테스트**: yfinance 데이터로 언제든지 실행 가능

## 📊 DAA 전략 규칙

```
카나리 유니버스 (VWO, BND)의 bad 자산 개수 b에 따라:

b=0 (둘 다 good)  → 100% 위험자산
b=1 (하나만 bad)  → 50% 위험자산 + 50% 안전자산  
b=2 (둘 다 bad)   → 100% 안전자산
```

## 🚀 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/daa-backtest.git
cd daa-backtest

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. Streamlit 실행
streamlit run app.py
```

## 📁 프로젝트 구조

```
daa-backtest/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                 # Streamlit 메인 애플리케이션
├── config.py             # 설정 파일
├── daa_strategy.py       # DAA 전략 구현
├── backtest.py           # 백테스트 엔진
├── utils.py              # 유틸리티 함수
└── data/
    └── cache/            # 캐시된 데이터
```

## 📈 사용 예시

Streamlit 대시보드에서:
1. 시작 날짜와 종료 날짜 선택
2. 위험자산 유니버스 선택 (기본: SPY, VEA, VWO, QQQ 등)
3. 벤치마크 선택 (기본: 60/40 포트폴리오)
4. '백테스트 실행' 클릭

## 🔧 주요 설정

`config.py`에서 설정 가능:
- **Momentum 기간**: 1, 3, 6, 12개월
- **Top T**: 선택할 상위 자산 개수
- **Breadth Parameter B**: 2 (기본) 또는 1 (공격적)
- **거래 비용**: 0.1% (기본)

## 📊 결과 해석

### 성과 지표
- **CAGR**: 연평균 수익률
- **Max Drawdown**: 최대 낙폭
- **Sharpe Ratio**: 위험조정 수익률
- **Win Rate**: 승리한 월의 비율
- **Cash Fraction**: 평균 현금 비율

### 차트
- 누적 수익 곡선 (DAA vs 벤치마크)
- 월별 수익률 히트맵
- 현금 비율 변화
- 자산 배분 변화

## 📝 논문 참고

Keller, W.J., Keuning, J.W. (2018). 
"Breadth Momentum and the Canary Universe: Defensive Asset Allocation (DAA)"
SSRN: https://ssrn.com/abstract=3212862

## 🤝 기여

Pull Request는 언제든 환영합니다!

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

## 📧 문의

질문이나 제안사항은 Issues에서 등록해주세요.
