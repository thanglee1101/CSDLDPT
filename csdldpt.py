import csv
import pandas as pd
import numpy as np
import librosa

features_newfile=[]
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    is_nan = np.isnan(f0)
    f0[is_nan] = 0
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    frame_length = 2048  # độ dài khung
    hop_length = 512  # độ dài bước nhảy
    energy = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length)[0]
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features_newfile.append(f0)
    features_newfile.append(zcr)
    features_newfile.append(energy)
    features_newfile.append(spec_centroid)
def euclidean(vec1,vec2):
    maxres=1
    for i in range(32):
        for j in range(32):
            same=0
            same=same+float(abs(vec1[0][i]-vec2[0][j])/max(vec1[0][i],vec2[0][j])*0.3)
            same=same+float(abs(vec1[1][i]-vec2[1][j])/max(vec1[1][i],vec2[1][j])*0.3)
            same=same+float(abs(vec1[2][i]-vec2[2][j])/max(vec1[2][i],vec2[2][j])*0.2)
            same=same+float(abs(vec1[3][i]-vec2[3][j])/max(vec1[3][i],vec2[3][j])*0.2)
            if maxres>same:
                maxres=same
    return maxres
# Đường dẫn đến file CSV
csv_file_path = 'features.csv'

words = ['biên', 'huy', 'Khánh', 'Phòng', 'Phú',
         'Súng', 'thắng', 'thể', 'Tiền', 'việt']

stored_data = {'biên': [], 'huy': [], 'Khánh': [], 'Phòng': [], 'Phú': [
], 'Súng': [], 'thắng': [], 'thể': [], 'Tiền': [], 'việt': []}
compare_arr = {'biên':None , 'huy':None , 'Khánh' :None , 'Phòng' :None , 'Phú':None  
, 'Súng' :None , 'thắng':None  , 'thể':None  , 'Tiền':None  , 'việt':None  }

# Mở file CSV và đọc dữ liệu
with open(csv_file_path, 'r', encoding="utf8") as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == "Label":
            continue
        label = row[0]
        vector = []
        for i in range(1, 5):
            string_nums = row[i]
            string_nums = string_nums.replace('\n', ' ')
            string_nums = string_nums.replace(']', ' ')
            string_nums = string_nums.replace('[', ' ')
            num_list = string_nums.split()
            arr = np.array([float(num) for num in num_list])
            vector.append(arr)
        stored_data[label].append(vector)
while True:
    features_newfile=[]
    for key in compare_arr.keys():
        compare_arr[key]=None
    file_path = input("Nhap duong dan file :")
    if file_path == "c":
        break
    extract_features(file_path)
    for word in words:
        check_arr=[]
        for i in range(len(stored_data[word])):
            distance = euclidean(features_newfile,stored_data[word][i])
            check_arr.append(distance)
        min_dis=min(check_arr)
        compare_arr[word]=min_dis
    min_distance =min(compare_arr.values())
    min_keys = [key for key, value in compare_arr.items() if value == min_distance]
    print(min_keys[0])