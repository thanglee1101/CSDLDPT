import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Đọc file âm thanh và lấy tín hiệu và tần số lấy mẫu
y, sr = librosa.load(f'.\dataset\\tiền\\tiền 3.WAV',sr=16000)

# Tính toán phổ tần số của tín hiệu âm thanh bằng phép biến đổi Fourier nhanh (STFT)
D = np.abs(librosa.stft(y))

# Hiển thị biểu đồ phổ tần số
plt.figure(figsize=(12, 8))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
