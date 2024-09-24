import pandas as pd
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('sawtooth_matrix.csv', header=None)

# x값 생성 (0부터 360까지 1000개의 선형 증가 값)
x_values = np.linspace(0, 360, 360)

dataframes = []

for idx, col in enumerate(df.columns):
    y_values = df[col].values  # 해당 파형의 y값 (1000개 점)

    # 첫 번째 감성값 계산
    emotion1 = min(y_values.max() * 10, 10)

    # 두 번째 감성값 계산
    zero_count = np.sum(y_values == 0)
    if zero_count == 0:
        emotion2 = 0
    elif zero_count == len(y_values):
        emotion2 = 10
    else:
        proportion_zero = zero_count / len(y_values)
        emotion2 = proportion_zero * 10

    # 세 번째 감성값 계산 (8, 9, 10 중 랜덤 선택)
    emotion3 = np.random.choice([1,2])

    # 감성값 리스트 생성
    emotion_values = [emotion1, emotion2, emotion3] + [np.nan] * 357

    # x값 리스트 생성
    x_list = list(x_values)

    # y값 리스트 생성
    y_list = list(y_values)

    # 개별 데이터프레임 생성
    col_df = pd.DataFrame({
        f'emotion_{idx+1}': emotion_values,
        f'x_{idx+1}': x_list,
        f'y_{idx+1}': y_list
    })

    dataframes.append(col_df)

# 모든 데이터프레임을 열 방향으로 연결
output_df = pd.concat(dataframes, axis=1)

# 새로운 CSV 파일로 저장
output_df.to_csv('swt_sample_matrix.csv', index=False)
