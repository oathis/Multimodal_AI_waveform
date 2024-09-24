
from waveform_shape import TRIANGULAR_WAVE as tr, SQUARE_WAVE as sq
import numpy as np
import pylab as plt
import openpyxl
import pandas as pd

def read_selected_columns(file_name, columns, num_rows):
    # 엑셀 파일을 엽니다.
    workbook = openpyxl.load_workbook(file_name)
    
    # 첫 번째 시트를 선택합니다.
    sheet = workbook.active
    
    # 행렬을 저장할 리스트를 초기화합니다.
    matrix = []
    
    # 각 열에 대해 데이터 추출
    for col in columns:
        column_data = []
        for row in range(1, num_rows + 1):
            cell_value = sheet.cell(row=row, column=col).value
            column_data.append(cell_value)
        matrix.append(column_data)
    
    # 리스트를 NumPy 배열로 변환하여 25x360 행렬을 생성합니다.
    matrix_np = np.array(matrix)
    
    return matrix_np


def save_matrix_to_csv(matrix, file_name):
    # NumPy 배열을 pandas DataFrame으로 변환합니다.
    df = pd.DataFrame(matrix)
    # DataFrame을 CSV 파일로 저장합니다.
    df.to_csv(file_name, index=False, header=False)


rad = np.linspace(0,360,360)

sinu_waves = []  
tri_waves =[]
sqr_waves =[]
swt_waves = []

for Ta in range(3):
    T = 45 / (Ta + 1)
    for Am in range(3):
        amplitude = 1 / (Am + 1)
        for ze in range(3):
            zero_ratio = 0.2 * ze
            sinu_wave = tr.sine_wave(rad, T, amplitude, phase=0.0, zero_ratio=zero_ratio)
            sinu_waves.append(sinu_wave)  # 각 배열을 리스트에 추가합니다.

for Ta in range(3):
    T = 45 / (Ta + 1)
    for Am in range(3):
        amplitude = 1 / (Am + 1)
        for ze in range(3):
            zero_ratio = 0.2 * ze
            tri_wave = tr.triangular_wave(rad, T, amplitude, phase=0.0, zero_ratio=zero_ratio)
            tri_waves.append(tri_wave)  # 각 배열을 리스트에 추가합니다.
            

for Ta in range(3):
    T = 45 / (Ta + 1)
    for Am in range(3):
        amplitude = 1 / (Am + 1)
        for ze in range(3):
            zero_ratio = 0.2 * ze
            swt = tr.sawtooth_wave(rad, T, zero_ratio=zero_ratio)+1
            swt_wave = swt/np.max(swt)*amplitude
            swt_waves.append(swt_wave)  # 각 배열을 리스트에 추가합니다.
            
for Ta in range(3):
    T = 45 / (Ta + 1)
    for Am in range(3):
        amplitude = 1 / (Am + 1)
        for ze in range(3):
            zero_ratio = 0.2 * ze
            sqr_wave = amplitude*sq.square_wave(rad, T, duty_cycle=0.5, zero_ratio=zero_ratio)
            sqr_waves.append(sqr_wave)  # 각 배열을 리스트에 추가합니다.
            
tri_waves = np.transpose(tri_waves)
sqr_waves= np.transpose(sqr_waves)
sinu_waves= np.transpose(sinu_waves)
swt_waves= np.transpose(swt_waves)

# tr.sine_wave(rad, T, amplitude, phase=0.0, zero_ratio)
# tr.triangular_wave(rad, T, amplitude, phase=0.0, zero_ratio)
# tr.sawtooth_wave(rad, T, amplitude, zero_ratio)
# sq.square_wave(rad, T, duty_cycle=0.5, zero_ratio)

sinu_name = 'sine_matrix.csv'
tri_name = 'tri_matrix.csv'
sqr_name = 'square_matrix.csv'
swt_name = 'sawtooth_matrix.csv'

save_matrix_to_csv(sinu_waves, sinu_name)
print(f"Matrix saved to {sinu_name}")

save_matrix_to_csv(tri_waves, tri_name)
print(f"Matrix saved to {tri_name}")

save_matrix_to_csv(sqr_waves, sqr_name)
print(f"Matrix saved to {sqr_name}")

save_matrix_to_csv(swt_waves, swt_name)
print(f"Matrix saved to {swt_name}")


