import numpy as np
import librosa
import scipy.io.wavfile as wav

sr = 16000

ir, _ = librosa.load("1.wav", sr=sr)    #读取音频 返回值为数组和采样率 x
voice, _ = librosa.load("rir.wav", sr=sr) # r

#voice = voice[:32000]
conv = np.convolve(ir, voice, "full")    #计算卷积   y

n = voice.shape[0] + ir.shape[0] - 1    #卷积长度
conv_fft = np.fft.fft(conv, n=n)    #DFT Y
voice_fft = np.fft.fft(voice, n=n)  #DFT R

ir_fft = conv_fft / voice_fft       #DFT X
ir_fft_ifft = np.fft.ifft(ir_fft)   #x

# 找到中心索引
center_index = n // 2

# 裁剪到与直接卷积相同的长度
ir_fft_ifft = ir_fft_ifft[center_index - len(conv)//2 : center_index + len(conv)//2 + len(conv)%2]

ir_fft_ifft = ir_fft_ifft[:len(ir)]

ir_fft_ifft = np.real(ir_fft_ifft)
ir_fft_ifft = ir_fft_ifft / np.max(np.abs(ir_fft_ifft))
sampling_rate = 16000
wav.write('2.wav', sampling_rate, (ir_fft_ifft * 32767).astype(np.int16))

pass


