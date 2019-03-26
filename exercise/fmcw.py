# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D

# parameters setting
B = 135e6  # Sweep
BandwidthT = 36.5e-6  # Sweep Time
N = 512  # Sample Length
L = 128  # Chirp Total
c = 3e8  # Speed of Light
f0 = 76.5e9  # Start Frequency
NumRangeFFT = 512  # Range FFT Length
NumDopplerFFT = 128  # Doppler FFT Length
T = 50e-4


def generate_speed_and_distance_img_3d(frame_chunk, frame_index):
    X = []  # 距离
    Y = []  # 速度
    Z = []  # 亮度，代表点
    for row_index in range(0, len(frame_chunk)):
        for col_index in range(0, len(frame_chunk[row_index])):
            X.append(col_index)
            Y.append(row_index)
            Z.append(frame_chunk[row_index][col_index] / 1000)
    fig = plt.figure()
    # plt.clf()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
    ax.set_xlabel('Speed')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Strength')
    # plt.savefig(IMG_SAVE_PATH + "speed_dis_" + str(frame_index) + ".png")
    plt.show()


rangeRes = c / 2 / B  # Range Resolution
velRes = c / 2 / f0 / T / NumDopplerFFT  # Velocity Resolution
maxRange = rangeRes * NumRangeFFT / 2  # Max Range
maxVel = velRes * NumDopplerFFT / 2  # Max Velocity
tarR = [50, 90]  # Target Range
tarV = [3, 20]  # Target Velocity
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

# range win processing
sigRangeWin = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    sigRangeWin[l] = sp.multiply(sigReceive[l], sp.hamming(N).T)

generate_speed_and_distance_img_3d(abs(sigRangeWin), 1)

# range fft processing
sigRangeFFT = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    sigRangeFFT[l] = np.fft.fft(sigRangeWin[l], NumRangeFFT)

generate_speed_and_distance_img_3d(abs(sigRangeFFT), 1)

# doppler win processing
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
generate_speed_and_distance_img_3d(abs(sigDopplerFFT), 3)
