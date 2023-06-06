import pandas as pd
import os
import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile
import librosa





# Đường dẫn tới thư mục chứa các file wav
data_path = './dataset'

# Khởi tạo dataframe
df = pd.DataFrame(columns=['Label', 'Pitch','ZCR','Average Energy', "Spectral Centroid"])

# Duyệt qua từng file trong thư mục
for label in os.listdir(data_path):
    if not label.startswith('.'):
        for file in os.listdir(os.path.join(data_path, label)):
            if file.endswith('.WAV'):
                # Đọc file wav
                # _,signal = wavfile.read(os.path.join(data_path, label, file))
                y, sr = librosa.load(os.path.join(data_path, label, file),sr=16000)
               # Extract pitch feature
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                is_nan = np.isnan(f0)
                f0[is_nan] =0
                # Thêm dòng mới vào dataframe
                # print(label)
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                frame_length = 2048  # độ dài khung
                hop_length = 512  # độ dài bước nhảy
                energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
             
                df = df.append({'Label': label, 'Pitch': f0,'ZCR':zcr,'Average Energy':energy,"Spectral Centroid":spec_centroid}, ignore_index=True)

# Lưu dataframe vào file csv
print(df)
df.to_csv('features.csv', index=False)
