# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl

# parameters setting
B = 135e6  # Sweep Bandwidth
T = 36.5e-6  # Sweep Time
N = 64  # Sample Length
L = 512  # Chirp Total
c = 3e8  # Speed of Light
f0 = 77.2e9  # Start Frequency
NumRangeFFT = 64  # Range FFT Length
NumDopplerFFT = 512  # Doppler FFT Length


def generate_speed_and_distance_img_3d(frame_chunk, frame_index, fiel_name):
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    X = []  # 距离
    Y = []  # 速度
    Z = []  # 亮度，代表点
    for row_index in range(0, len(frame_chunk)):
        for col_index in range(0, len(frame_chunk[row_index])):
            X.append(col_index - 256)
            Y.append(row_index)
            Z.append(frame_chunk[row_index][col_index])
    fig = plt.figure()
    # plt.clf()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
    plt.yticks(ticks=[i for i in range(0, 64, 8)], label=['-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1'])
    ax.set_xlabel('速度维')
    ax.set_ylabel('距离维')
    ax.set_zlabel('信号强度')
    plt.savefig("C:\\Users\\zt\\Desktop\\2d_FFT\\" + fiel_name + "_" + str(frame_index) + ".pdf")
    # plt.show()


def re_list_data(frame_chunk):
    positive_frame = frame_chunk[0:256]
    negative_frame = frame_chunk[256:]
    new_frame_chunk = []
    # positive_frame[1] = [i - i for i in range(64)]
    # negative_frame[256] = [i - i for i in range(64)]

    for d in negative_frame:
        new_frame_chunk.append(d)

    for d in positive_frame:
        new_frame_chunk.append(d)

    return new_frame_chunk


rangeRes = c / 2 / B  # Range Resolution
velRes = c / 2 / f0 / T / NumDopplerFFT  # Velocity Resolution
maxRange = rangeRes * NumRangeFFT / 2  # Max Range
maxVel = velRes * NumDopplerFFT / 2  # Max Velocity
tarR = [10, 60]  # Target Range
tarV = [0, 10]  # Target Velocity
#  generate receive signal
S1 = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    for n in range(0, N):
        S1[l][n] = np.exp(np.complex(0, 1) * 2 * np.pi * (
                ((2 * B * (tarR[0] + tarV[0] * T * l)) / (c * T) + (2 * f0 * tarV[0]) / c) * (T / N) * n + (
                2 * f0 * (tarR[0] + tarV[0] * T * l)) / c))

S2 = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    for n in range(0, N):
        S2[l][n] = np.exp(np.complex(0, 1) * 2 * np.pi * (
                ((2 * B * (tarR[1] + tarV[1] * T * l)) / (c * T) + (2 * f0 * tarV[1]) / c) * (T / N) * n + (
                2 * f0 * (tarR[1] + tarV[1] * T * l)) / c))
        sigReceive = S1 + S2

# 对原始信号进行成像
sig_receive = abs(sigReceive)
sig_receive = np.transpose(sig_receive).tolist()
generate_speed_and_distance_img_3d(sig_receive, 1, "none_fft")
# range win processing
sigRangeWin = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    sigRangeWin[l] = sp.multiply(sigReceive[l], sp.hamming(N).T)

# range fft processing
sigRangeFFT = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    sigRangeFFT[l] = np.fft.fft(sigRangeWin[l], NumRangeFFT)

first_fft = abs(sigRangeFFT)
first_fft = np.transpose(first_fft).tolist()
first_fft = re_list_data(first_fft)
generate_speed_and_distance_img_3d(first_fft, 2, "one_fft")

# doppler win processing 加窗
sigDopplerWin = np.zeros((L, N), dtype=complex)
for n in range(0, N):
    sigDopplerWin[:, n] = sp.multiply(sigRangeFFT[:, n], sp.hamming(L).T)

# generate_speed_and_distance_img_3d(abs(sigDopplerWin), 2)

# doppler fft processing
sigDopplerFFT = np.zeros((L, N), dtype=complex)
for n in range(0, N):
    sigDopplerFFT[:, n] = np.fft.fft(sigDopplerWin[:, n], NumDopplerFFT)

# print('sigDopplerFFT: ', sigDopplerFFT)
# sigDopplerFFT = abs(sigDopplerFFT)
second_fft = abs(sigDopplerFFT)
zero = np.zeros(64)
# second_fft[511] = zero


second_fft = re_list_data(second_fft)
second_fft = np.transpose(second_fft).tolist()
generate_speed_and_distance_img_3d(second_fft, 3, "two_fft")
