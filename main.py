import pandas as pd
import numpy as np

# 예시 데이터 (주어진 DataFrame 형태)
data = {
    'Date': ['Dec.24', 'Nov.24', 'Oct.24', 'Sep.24', 'Aug.24', 'Jul.24', 'Jun.24', 'May.24', 'Apr.24', 'Mar.24', 'Feb.24', 'Jan.24'],
    'KOSPI': [2428.16, 2455.91, 2556.15, 2593.27, 2674.31, 2770.69, 2797.82, 2636.52, 2692.06, 2746.63, 2642.36, 2497.09],
    'KOSDAQ': [661.33, 678.19, 743.06, 763.88, 767.66, 803.15, 840.44, 839.98, 868.93, 905.5, 862.96, 799.24],
    'NASDAQ': [19403.95, 18239.92, 17910.36, 17136.30, 17194.14, 17879.30, 16828.67, 15605.48, 16396.83, 16274.94, 15361.64, 14843.77],
    'S&P': [6047.15, 5728.8, 5708.75, 5528.93, 5446.68, 5475.09, 5283.4, 5018.39, 5243.77, 5137.08, 4906.19, 4742.83],
    'US_interest': [4.75, 4.75, 5.0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
    'KR_interest': [3, 3.25, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
}

df = pd.DataFrame(data)
df = df.set_index('Date')

# 필요하다면 시간 순서대로 정렬(현재 역순이므로 역순 인덱싱)
df = df.iloc[::-1]

# -----------------------------------------
# 간단한 MDP 정의
# -----------------------------------------

# 상태(state)와 액션(action)을 정의한다.
# 여기서는 상태를 금리 수준과 현재 보유 자산에 의한 조합으로 가정한다.
#
# 상태 정의 (US_interest_regime, KR_interest_regime, currently_held_asset)
# US_interest_regime: low, med, high (예시)
# KR_interest_regime: low, med, high
# currently_held_asset: {None, "KOSPI", "KOSDAQ", "NASDAQ", "S&P"}

# 금리를 범주화하는 함수 정의
def categorize_interest(rate):
    if rate < 5.0:
        return 'low'
    elif rate < 5.5:
        return 'med'
    else:
        return 'high'

# 각 월별 행으로부터 상태 정보를 추출
states = []
for i, row in df.iterrows():
    us_cat = categorize_interest(row['US_interest'])
    kr_cat = categorize_interest(row['KR_interest'])
    # 초기에는 자산을 보유하지 않은 상태(None)로 시작
    states.append((us_cat, kr_cat, None))
    
# 가능한 액션 정의
# 자산 미보유 시: Buy_KOSPI, Buy_KOSDAQ, Buy_NASDAQ, Buy_S&P, Hold, Sell
# 자산 보유 시: Hold, Sell 만 가능(단순화)
actions = ["Buy_KOSPI", "Buy_KOSDAQ", "Buy_NASDAQ", "Buy_S&P", "Hold", "Sell"]

# 실제 MDP에서는 P(s'|s,a)와 R(s,a,s')를 정의해야 한다.
# 여기서는 단순화:
# - 자산 미보유 상태에서 "Buy_X"를 하면 해당 자산 보유 상태로 전환
# - 자산 보유 중 "Hold" 시 다음 시점으로 이동하면서 해당 자산의 수익률을 보상으로 획득
# - "Sell" 시 자산을 매도하고 다시 None 상태로 전환
#
# 각 row(월)를 한 타임스텝으로 보고, 단순히 이전 달에서 다음 달로 100% 전이한다고 가정(결정적 전이)
# 보상은 해당 자산을 hold하는 동안 발생하는 수익률로 간주

unique_interest_states = ['low', 'med', 'high']
possible_assets = [None, "KOSPI", "KOSDAQ", "NASDAQ", "S&P"]

# MDP 전이 정보를 저장할 딕셔너리
# mdp[(state, action)] = [(전이확률, 다음상태, 보상), ...]
mdp = {}
dates = df.index.tolist()

for t in range(len(dates)-1):
    current_date = dates[t]
    next_date = dates[t+1]
    current_row = df.loc[current_date]
    next_row = df.loc[next_date]

    # 현재 상태(자산 미보유/보유 각각에 대해)
    us_cat = categorize_interest(current_row['US_interest'])
    kr_cat = categorize_interest(current_row['KR_interest'])

    current_states = [(us_cat, kr_cat, a) for a in possible_assets]

    # 현재 날짜에서 다음 날짜로 각 자산별 수익률 계산
    def asset_return(asset):
        return (next_row[asset] - current_row[asset]) / current_row[asset] if asset in df.columns else 0
    
    kospi_ret = asset_return('KOSPI')
    kosdaq_ret = asset_return('KOSDAQ')
    nasdaq_ret = asset_return('NASDAQ')
    sp_ret = asset_return('S&P')

    # 다음 상태(금리 범주) 결정
    next_us_cat = categorize_interest(next_row['US_interest'])
    next_kr_cat = categorize_interest(next_row['KR_interest'])

    # 현재 상태별로 가능한 액션에 따른 전이 정의
    for state in current_states:
        (u_cat, k_cat, asset_held) = state

        # 자산 보유 여부에 따른 액션 설정
        if asset_held is None:
            # 자산 미보유 시: 모든 Buy 액션 가능, Hold/Sell 시 변화 없음(자산 없음)
            possible_actions = ["Buy_KOSPI", "Buy_KOSDAQ", "Buy_NASDAQ", "Buy_S&P", "Hold", "Sell"]
        else:
            # 자산 보유 시: Hold나 Sell만 가능(다른 자산 Buy 불가)
            possible_actions = ["Hold", "Sell"]

        # 액션별 전이 정의
        for a in possible_actions:
            if asset_held is None:
                # 현재 자산 미보유인 경우
                if a.startswith("Buy"):
                    new_asset = a.split("_")[1]
                    # 자산을 새로 사면 다음 상태에서 해당 자산 보유
                    next_state = (next_us_cat, next_kr_cat, new_asset)
                    # 구매 시점에는 즉시 보상 없음(다음 스텝에서 수익률 실현)
                    reward = 0.0
                elif a == "Sell":
                    # 이미 자산 미보유인데 Sell -> 변화 없음
                    next_state = (next_us_cat, next_kr_cat, None)
                    reward = 0.0
                else:  # Hold (자산 없음)
                    next_state = (next_us_cat, next_kr_cat, None)
                    reward = 0.0
            else:
                # 자산 보유 중인 경우
                if a == "Hold":
                    # 자산을 계속 보유하면 다음 단계로 이동하며 자산 수익률 반영
                    next_state = (next_us_cat, next_kr_cat, asset_held)
                    if asset_held == "KOSPI":
                        reward = kospi_ret
                    elif asset_held == "KOSDAQ":
                        reward = kosdaq_ret
                    elif asset_held == "NASDAQ":
                        reward = nasdaq_ret
                    elif asset_held == "S&P":
                        reward = sp_ret
                    else:
                        reward = 0.0
                elif a == "Sell":
                    # 자산을 매도하면 다시 None 상태
                    # 여기서는 매도 시점 추가 보상은 없는 것으로 단순화
                    next_state = (next_us_cat, next_kr_cat, None)
                    reward = 0.0

            # 결정적 전이(확률 1.0)로 가정
            mdp.setdefault((state, a), []).append((1.0, next_state, reward))

# 이제 value iteration(가치 반복)을 통해 최적 정책을 구한다.

# MDP 키에서 모든 상태 추출
all_states = set()
for (state, action), transitions in mdp.items():
    all_states.add(state)


    for (_, s_next, _) in transitions:
        all_states.add(s_next)
all_states = list(all_states)
all_states = sorted(all_states, key=lambda s: (s[0], s[1], s[2] if s[2] is not None else ""))

# 가치 함수(V) 초기화
V = {s: 0.0 for s in all_states}
gamma = 0.95
theta = 1e-6

def value_iteration(mdp, gamma, theta):
    V = {s: 0.0 for s in all_states}
    while True:
        delta = 0
        for s in all_states:
            # 해당 상태에서 가능한 액션 탐색
            possible_actions = [a for (st, a) in mdp.keys() if st == s]
            if not possible_actions:
                # terminal 상태일 경우 넘어감
                continue
            # 각 액션에 대한 가치 계산
            A_values = []
            for a in possible_actions:
                transitions = mdp[(s,a)]
                val = 0.0
                for (p, s_next, r) in transitions:
                    val += p * (r + gamma * V[s_next])
                A_values.append(val)
            max_val = max(A_values) if A_values else 0.0
            delta = max(delta, abs(V[s] - max_val))
            V[s] = max_val
        if delta < theta:
            break
    return V

V = value_iteration(mdp, gamma, theta)

# 최적 정책 도출
policy = {}
for s in all_states:
    actions_from_s = [a for (st,a) in mdp.keys() if st == s]
    if not actions_from_s:
        policy[s] = None
    else:
        # Q값을 계산 후, 최대값을 주는 액션 선택
        best_a = None
        best_val = -np.inf
        for a in actions_from_s:
            val = 0.0
            for (p, s_next, r) in mdp[(s,a)]:
                val += p*(r + gamma*V[s_next])
            if val > best_val:
                best_val = val
                best_a = a
        policy[s] = best_a

# 일부 상태와 최적 액션, 가치 출력 (예시)
for i, s in enumerate(all_states[:10]):
    print("상태:", s, "최적 액션:", policy[s], "가치:", V[s])
