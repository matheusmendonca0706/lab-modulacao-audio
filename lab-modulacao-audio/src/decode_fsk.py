import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
BIT_DURATION = 1.0
THRESHOLD = 660

def detect_frequency(seg):
    fft = np.fft.fft(seg * np.hanning(len(seg)))
    freqs = np.fft.fftfreq(len(fft), 1/SAMPLE_RATE)
    return abs(freqs[np.argmax(np.abs(fft))])

def decode_fsk(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    samples_per_bit = int(sr * BIT_DURATION)
    bits=[]
    for i in range(len(data)//samples_per_bit):
        seg = data[i*samples_per_bit:(i+1)*samples_per_bit]
        mid = samples_per_bit//2
        first = seg[samples_per_bit//8: mid - samples_per_bit//8]
        second = seg[mid + samples_per_bit//8: samples_per_bit - samples_per_bit//8]
        f1, f2 = detect_frequency(first), detect_frequency(second)
        b1 = '1' if f1>THRESHOLD else '0'
        b2 = '1' if f2>THRESHOLD else '0'
        bits.append(b1 if b1==b2 else b1)
    return ''.join(bits)

if __name__=='__main__':
    print(decode_fsk('dados/lab01_modulacao_dados_codificados_dados_121210603_44100hz.wav'))
