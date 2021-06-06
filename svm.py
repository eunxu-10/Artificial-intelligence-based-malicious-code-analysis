import glob
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# get dataset; 0 is normality, 1 is malicious code
files0 = glob.glob("./API/0/*.txt")
files1 = glob.glob("./API/1/*.txt")

software_sentences = []
for fi in files0:
    with open(fi, 'r') as f:
        lines = f.readlines()
        sentences = [line.rstrip('\n') for line in lines]
    software_sentences.append(sentences)

malware_sentences = []
for fi in files1:
    with open(fi,'r') as f:
        lines = f.readlines()
        sentences = [line.rstrip('\n') for line in lines]
    malware_sentences.append(sentences)

#학습에 사용할 doc2vec모델 load 시킴
doc2vec_model_name = "Doc2vec_model_vector30_window15_dm0"
model = Doc2Vec.load(doc2vec_model_name)

#위에서 가져온 문자열들을 doc2vec model에 넣고 벡터로 변환
#오래걸림.
software_vector = [model.infer_vector(sentence,alpha=0.025,min_alpha=0.025, epochs=30)
                 for sentence in software_sentences]
malware_vector = [model.infer_vector(sentence,alpha=0.025,min_alpha=0.025, epochs=30)
                 for sentence in malware_sentences]

# xgboost의 학습을 위한 전처리 과정으로 vecotr list을 numpy의 array로 변환
software_arrays = np.array(software_vector)
malware_arrays = np.array(malware_vector)

# software 개수만큼 0으로 초기화된 label 생성
software_labels = np.zeros(len(software_vector))
# print(software_labels) [0. 0. 0. ... 0. 0. 0.]

# malware 개수만큼 1으로 초기화된 label 생성
malware_labels = np.ones(len(malware_vector))
# print(malware_labels) [1. 1. 1. ... 1. 1. 1.]

#데이터를 하나로 합치기.
arrays = np.vstack((software_arrays,malware_arrays))
labels = np.hstack((software_labels,malware_labels))

kf_test = KFold(n_splits = 5, shuffle = True)#kfold를 사용하여 데이터셋 분리

for train_index, test_index in kf_test.split(arrays):# 1/5 로 데이터를 나눔. 원래 5번 교차검증을 해야하지만 여기선 생략.
    # split train/validation
    train_data, test_data  = arrays[train_index], arrays[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

clf = SVC(kernel='linear', C=1000)
clf.fit(train_data, train_labels)
y_pred = clf.predict(test_data)

print(accuracy_score(test_labels, y_pred)) #0.8658718330849479 (총 학습시간 약 1시간 20분)

import matplotlib.pyplot as plt
plt.title('SVM-test data Classification')
y_pred = y_pred.astype(float)
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        plt.scatter(test_data[:, 0][i],test_data[:,1][i], color='blue')
    if y_pred[i]==1:
        plt.scatter(test_data[:, 0][i], test_data[:,1][i], color='red')