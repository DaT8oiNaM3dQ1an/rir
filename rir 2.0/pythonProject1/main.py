import numpy as np
import librosa
import soundfile

sr = 16000

rir, _ = librosa.load("E:/rir 2.0/true_rir_0.wav", sr=sr)            # 真实rir
voice, _ = librosa.load("E:/rir 2.0/1.wav", sr=sr)                   # 音频信号
rir_pre, _ = librosa.load("E:/rir 2.0/predicted_rir_0.wav", sr=sr)   # 预测rir

rir_length = len(rir)  #rir长度

voices=[]
shu=0
u = int(rir_length/2)
for i in range(0, len(voice), u):      #将音频信号进行切片
    v = voice[i:i + u]
    if len(v) < rir_length:
        v = librosa.util.pad_center(v, size=rir_length)
    voices.append(v)

voices_pre_part=[]
voices_rir_part=[]

n = max(rir_length + u  - 1, len(rir_pre)) #确定长度

for vp in voices:                    #反卷积求出预测音频信号
    conv = np.convolve(vp, rir, "same")    #卷积出混响声音
    voices_rir_part.append(conv)    

    conv_fft = np.fft.fft(conv, n)          #卷积结果 fft
    RIR_fft = np.fft.fft(rir_pre, n)        #预测rir fft
    vp_fft = conv_fft / (RIR_fft+ 1e-6)
    vp_pre = np.real(np.fft.ifft(vp_fft))   #ifft得出预测语音
    vp_pre = vp_pre / np.max(np.abs(vp_pre))
    voices_pre_part.append(vp_pre)

voice_rir = np.concatenate(voices_rir_part)
soundfile.write("E:/rir 2.0/voice_rir.wav", voice_rir, sr)
voice_pre = np.concatenate(voices_pre_part)
soundfile.write("E:/rir 2.0/voice_pre.wav", voice_pre, sr)


pass