import numpy as np
import matplotlib.pyplot as plt

class SQUARE_WAVE():
    def trapezoidal_wave(t, T=2, width_ratio=0.5, amplitude=1, zero_ratio=None):
        if zero_ratio is None:
            zero_ratio = 0.5
        assert zero_ratio < 1, "zero_ratio must be less than 1"
        
        period = np.mod(t, T)
        less_T=(T-T*zero_ratio)
        half_width = less_T * width_ratio / 2
        wave = np.zeros_like(t)
        
        wave[(period >= 0) & (period < half_width)] = (amplitude / half_width) * period[(period >= 0) & (period < half_width)]
        wave[(period >= half_width) & (period < (1-zero_ratio)*T - half_width)] = amplitude
        wave[(period >= (1-zero_ratio)*T - half_width) & (period < (1-zero_ratio)*T)] = amplitude - (amplitude / half_width) * (period[(period >= (1-zero_ratio)*T - half_width) & (period < (1-zero_ratio)*T)] - ((1-zero_ratio)*T - half_width))
        
        return wave

    def half_trapezoidal_wave(t, T=2, width_ratio=0.5, amplitude=1, zero_ratio=None):
        if zero_ratio is None:
            zero_ratio = 0.5
        assert zero_ratio < 1, "zero_ratio must be less than 1"
        
        period = np.mod(t, T)
        less_T=(1-zero_ratio)*T
        half_width = less_T * width_ratio
        wave = np.zeros_like(t)
        
        wave[(period >= 0) & (period < half_width)] = (amplitude / half_width) * period[(period >= 0) & (period < half_width)]
        wave[(period >= half_width) & (period < (1-zero_ratio)*T)] = amplitude
        
        return wave

    def square_wave(t, T=1, duty_cycle=0.5, zero_ratio=None):
        if zero_ratio is None:
            zero_ratio = 0.5
        assert zero_ratio < 1, "zero_ratio must be less than 1"
        
        period = np.mod(t, T)
        wave = np.where((period < T * (1-zero_ratio)) & (period < (1 - zero_ratio)*T), 1.0, 0.0)
        
        return wave

class TRIANGULAR_WAVE():
    def triangular_wave(t, T=1.0, amplitude=1, phase=0.0, zero_ratio=None):
        if zero_ratio is None:
            zero_ratio = 0.5
        assert 0 <= zero_ratio < 1, "zero_ratio must be between 0 and 1"
        less_T = (1 - zero_ratio) * T
        # 현재 주기 내에서의 시간 위치 계산
        t_mod = np.mod(t + phase, T)
        # 활성 구간 마스크 생성
        active_mask = t_mod < less_T
        wave = np.zeros_like(t)
        # 활성 구간에서의 시간 정규화
        t_active = t_mod[active_mask] / less_T
        # 삼각파 생성
        wave[active_mask] = amplitude * (1 - 2 * np.abs(t_active - 0.5))
        return wave

    def sawtooth_wave(t, T=1.0, amplitude=1.0, zero_ratio=None):
        if zero_ratio is None:
            zero_ratio = 0.5
        assert zero_ratio < 1, "zero_ratio must be less than 1"
        
        period = np.mod(t, T)  # 주기 내의 시간
        wave = (2 * amplitude / T) * period - amplitude  # 톱니파 계산
        
        wave[period > (1-zero_ratio)*T] = -amplitude
        
        return wave

    def two_slope_wave(t, T=2, width_ratio=0.5, slope1=1, slope2=2, zero_ratio=None):
        if zero_ratio is None:
            zero_ratio = 0.5
        assert zero_ratio < 1, "zero_ratio must be less than 1"
        
        period = np.mod(t, T)
        less_T=(1-zero_ratio)*T
        half_width = less_T * width_ratio
        wave = np.zeros_like(t)
        
        rising1 = (period >= 0) & (period < half_width)
        wave[rising1] = slope1 * period[rising1]
        rising2 = (period >= half_width) & (period < 2*half_width)
        wave[rising2] = slope1 * (half_width ) + slope2 * (period[rising2] - half_width )
        
        return wave
    
    def half_sine_wave(t, T=1.0, amplitude=1, phase=0.0, zero_ratio=None):
        if zero_ratio is None:
            zero_ratio = 0.5
        assert 0 <= zero_ratio < 1, "zero_ratio must be less than 1"
        less_T = (1 - zero_ratio) * T
        # 현재 주기 내에서의 시간 위치 계산
        t_mod = np.mod(t + phase, T)
        # 활성 구간 마스크 생성
        active_mask = t_mod < less_T
        wave = np.zeros_like(t)
        # 활성 구간에서의 시간 정규화
        t_active = t_mod[active_mask] / less_T
        # 사인파 생성 (0에서 amplitude로 상승했다가 다시 0으로 하강)
        wave[active_mask] = amplitude * np.sin(np.pi * t_active)
        return wave
    
    def sine_wave(t, T=1.0, amplitude=1, phase=0.0, zero_ratio=None):
        if zero_ratio is None:
            zero_ratio = 0.5
        assert 0 <= zero_ratio < 1, "zero_ratio는 0 이상 1 미만이어야 합니다."
        less_T = (1 - zero_ratio) * T  # 활성 구간의 길이
        total_T = T  # 전체 주기

        # 현재 주기 내에서의 시간 위치 계산
        t_mod = np.mod(t + phase, total_T)
        wave = np.zeros_like(t)

        # 활성 구간 마스크 생성
        active_mask = t_mod < less_T
        t_active = t_mod[active_mask]

        # 각도 계산: 0에서 π까지 변환
        theta = 2*np.pi * t_active / less_T

        # 코사인 함수를 이용하여 파형 생성
        wave[active_mask] = amplitude * (-0.5 * np.cos(theta) + 0.5)

        return wave


# sq=SQUARE_WAVE()
# tr=TRIANGULAR_WAVE()
# #시간 벡터 정의
# t1 = np.linspace(0, 6, 1000)  # 0부터 6까지 1000개의 샘플

# # 함수 매개변수 정의
# T1 = 2
# zero_ratio = 0.3  # 주기의 절반

# # 각 함수의 출력값 계산
# trapezoidal = sq.trapezoidal_wave(t1, T=T1, zero_ratio=zero_ratio)
# half_trapezoidal = sq.half_trapezoidal_wave(t1, T=T1, zero_ratio=zero_ratio)
# two_slope = tr.two_slope_wave(t1, T=T1, zero_ratio=zero_ratio)
# triangular = tr.triangular_wave(t1, T=T1, zero_ratio=zero_ratio)
# square = sq.square_wave(t1, T=T1, zero_ratio=zero_ratio)
# sawtooth = tr.sawtooth_wave(t1, T=T1, zero_ratio=zero_ratio)

# 플롯 생성
# plt.figure(figsize=(12, 10))

# plt.subplot(3, 2, 1)
# plt.plot(t1, trapezoidal)
# plt.title('Trapezoidal Wave')
# plt.grid(True)

# plt.subplot(3, 2, 2)
# plt.plot(t1, half_trapezoidal)
# plt.title('Half Trapezoidal Wave')
# plt.grid(True)

# plt.subplot(3, 2, 3)
# plt.plot(t1, two_slope)
# plt.title('Two Slope Wave')
# plt.grid(True)

# plt.subplot(3, 2, 4)
# plt.plot(t1, triangular)
# plt.title('Triangular Wave')
# plt.grid(True)

# plt.subplot(3, 2, 5)
# plt.plot(t1, square)
# plt.title('Square Wave')
# plt.grid(True)

# plt.subplot(3, 2, 6)
# plt.plot(t1, sawtooth)
# plt.title('Sawtooth Wave')
# plt.grid(True)

# plt.tight_layout()
# plt.show()
