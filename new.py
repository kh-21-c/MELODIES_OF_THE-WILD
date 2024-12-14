from scipy.io import wavfile
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import sklearn
from sklearn.neural_network import MLPClassifier

#Data Training

#Training Cat voice
cat1, sr= librosa.load('C:\Myanimalrecognition\Data Train\cat\cat1.wav', duration=2.0) #load audio
cat1_train = librosa.feature.rms(y=cat1) #feature extraction

cat2, sr= librosa.load('C:\Myanimalrecognition\Data Train\cat\cat2.wav', duration=2.0) #load audio
cat2_train = librosa.feature.rms(y=cat2)  #feature extraction

#Training Dog Voice

dog1, sr= librosa.load('C:\Myanimalrecognition\Data Train\dog\dog1.wav', duration=2.0) #load audio
dog1_train = librosa.feature.rms(y=dog1) #feature extraction

dog2, sr= librosa.load('C:\Myanimalrecognition\Data Train\dog\dog2.wav', duration=2.0) #load audio
dog2_train = librosa.feature.rms(y=dog2) #feature extraction

dog3, sr= librosa.load('C:\Myanimalrecognition\Data Train\dog\dog3.wav', duration=2.0) #load audio
dog3_train = librosa.feature.rms(y=dog3) #feature extraction

dog4, sr= librosa.load('C:\Myanimalrecognition\Data Train\dog\dog4.wav', duration=2.0) #load audio
dog4_train = librosa.feature.rms(y=dog4) #feature extraction

dog5, sr= librosa.load('C:\Myanimalrecognition\Data Train\dog\dog5.wav', duration=2.0) #load audio
dog5_train = librosa.feature.rms(y=dog5) #feature extraction

dog6, sr= librosa.load('C:\Myanimalrecognition\Data Train\dog\dog6.wav', duration=2.0) #load audio
dog6_train = librosa.feature.rms(y=dog6) #feature extraction

dog7, sr= librosa.load('C:\Myanimalrecognition\Data Train\dog\dog7.wav', duration=2.0) #load audio
dog7_train = librosa.feature.rms(y=dog7) #feature extraction


#Training Duck Voice

duck1, sr= librosa.load('C:\Myanimalrecognition\Data Train\duck\duck1.wav', duration=2.0) #load audio
duck1_train = librosa.feature.rms(y=duck1) #feature extraction

duck2, sr= librosa.load('C:\Myanimalrecognition\Data Train\duck\duck2.wav', duration=2.0) #load audio
duck2_train = librosa.feature.rms(y=duck2) #feature extraction

duck3, sr= librosa.load('C:\Myanimalrecognition\Data Train\duck\duck3.wav', duration=2.0) #load audio
duck3_train = librosa.feature.rms(y=duck3) #feature extraction

#Training Horse Voice

horse1, sr= librosa.load('C:\Myanimalrecognition\Data Train\horse\horse1.wav', duration=2.0) #load audio
horse1_train = librosa.feature.rms(y=horse1) #feature extraction

horse2, sr= librosa.load('C:\Myanimalrecognition\Data Train\horse\horse2.wav', duration=2.0) #load audio
horse2_train = librosa.feature.rms(y=horse2) #feature extraction

horse3, sr= librosa.load('C:\Myanimalrecognition\Data Train\horse\horse3.wav', duration=2.0) #load audio
horse3_train = librosa.feature.rms(y=horse3) #feature extraction

horse4, sr= librosa.load('C:\Myanimalrecognition\Data Train\horse\horse4.wav', duration=2.0) #load audio
horse4_train = librosa.feature.rms(y=horse4) #feature extraction


#Training Lion Voice

lion1, sr= librosa.load('C:\Myanimalrecognition\Data Train\lion\lion1.wav', duration=2.0) #load audio
lion1_train = librosa.feature.rms(y=lion1) #feature extraction

lion2, sr= librosa.load('C:\Myanimalrecognition\Data Train\lion\lion2.wav', duration=2.0) #load audio
lion2_train = librosa.feature.rms(y=lion2) #feature extraction

lion4, sr= librosa.load('C:\Myanimalrecognition\Data Train\lion\lion4.wav', duration=2.0) #load audio
lion4_train = librosa.feature.rms(y=lion4) #feature extraction



#Training mosquito Voice

mosquito1, sr= librosa.load('C:\Myanimalrecognition\Data Train\mosquito\mosquito1.wav', duration=2.0) #load audio
mosquito1_train = librosa.feature.rms(y=mosquito1) #feature extraction

mosquito2, sr= librosa.load('C:\Myanimalrecognition\Data Train\mosquito\mosquito2.wav', duration=2.0) #load audio
mosquito2_train = librosa.feature.rms(y=mosquito2) #feature extraction

mosquito3, sr= librosa.load('C:\Myanimalrecognition\Data Train\mosquito\mosquito3.wav', duration=2.0) #load audio
mosquito3_train = librosa.feature.rms(y=mosquito3) #feature extraction


#Training elephant Voice

elephant1, sr= librosa.load('C:\Myanimalrecognition\Data Train\elephant\elephant1.wav', duration=2.0) #load audio
elephant1_train = librosa.feature.rms(y=elephant1) #feature extraction

elephant2, sr= librosa.load('C:\Myanimalrecognition\Data Train\elephant\elephant2.wav', duration=2.0) #load audio
elephant2_train = librosa.feature.rms(y=elephant2) #feature extraction

#Training crocodile Voice

crocodile1, sr= librosa.load('C:\Myanimalrecognition\Data Train\crocodile\crocodile1.wav', duration=2.0) #load audio
crocodile1_train = librosa.feature.rms(y=crocodile1) #feature extraction

#Training bear Voice

#bear1, sr= librosa.load('C:\Myanimalrecognition\Data Train\bear\bear1.wav', duration=2.0) #load audio
#bear1_train = librosa.feature.rms(y=bear1) #feature extraction

#bear2, sr= librosa.load('C:\Myanimalrecognition\Data Train\bear\bear2.wav', duration=2.0) #load audio
#bear2_train = librosa.feature.rms(y=bear2) #feature extraction

#bear3, sr= librosa.load('C:\Myanimalrecognition\Data Train\bear\bear3.wav', duration=2.0) #load audio
#bear3_train = librosa.feature.rms(y=bear3) #feature extraction

#Training the Model
#combine data into one numpy array data train
train_x = [cat1_train.ravel(), cat2_train.ravel(), dog1_train.ravel(), dog2_train.ravel(), dog3_train.ravel(),dog4_train.ravel(),dog5_train.ravel(),dog6_train.ravel(),dog7_train.ravel(), duck1_train.ravel(), duck2_train.ravel(), duck3_train.ravel(),horse1_train.ravel(), horse2_train.ravel(), horse3_train.ravel(), horse4_train.ravel(), lion1_train.ravel(), lion2_train.ravel(), lion4_train.ravel(),mosquito1_train.ravel(),mosquito2_train.ravel(),mosquito3_train.ravel(),elephant1_train.ravel(),elephant2_train.ravel(),crocodile1_train.ravel()]
#labeling data features
train_y = np.array(['cat','cat','dog','dog','dog','dog','dog','dog','dog','duck','duck','duck','horse','horse','horse','horse','lion','lion','lion','mosquito','mosquito','mosquito','elephant','elephant','crocodile'], dtype=object)

mlp = MLPClassifier()
mlp.fit(train_x,train_y)

#Preparing Testing Data

#Voice 1
cat1t, sr= librosa.load('C:\Myanimalrecognition\DataTest\cat1.wav', duration=2.0) #load audio
cat1_test = librosa.feature.rms(y=cat1t) #feature extraction

#Voice 2
duck1t, sr= librosa.load('C:\Myanimalrecognition\DataTest\duck1.wav', duration=2.0) #load audio
duck1_test = librosa.feature.rms(y=duck1t) #feature extraction

#Voice 3
dog2t, sr= librosa.load('C:\Myanimalrecognition\DataTest\dog2.wav', duration=2.0) #load audio
dog2_test = librosa.feature.rms(y=dog2t) #feature extraction

#Voice 4
horse3t, sr= librosa.load('C:\Myanimalrecognition\DataTest\horse3.wav', duration=2.0) #load audio
horse3_test = librosa.feature.rms(y=horse3t) #feature extraction
 
#Voice 5
lion4t, sr= librosa.load('C:\Myanimalrecognition\DataTest\lion4.wav', duration=2.0) #load audio
lion4_test = librosa.feature.rms(y=lion4t) #feature extraction

#Voice 6
#bear3t, sr= librosa.load('C:\Myanimalrecognition\DataTest\bear3.wav', duration=2.0) #load audio
#bear3_test = librosa.feature.rms(y=bear3t) #feature extraction

#Voice 7
elephant1t, sr= librosa.load('C:\Myanimalrecognition\DataTest\elephant1.wav', duration=2.0) #load audio
elephant1_test = librosa.feature.rms(y=elephant1t) #feature extraction

#Voice 9
mosquito2t, sr= librosa.load('C:\Myanimalrecognition\DataTest\mosquito2.wav', duration=2.0) #load audio
mosquito2_test = librosa.feature.rms(y=mosquito2) #feature extraction

data_test = [cat1_test.ravel(), duck1_test.ravel(), dog2_test.ravel(), horse3_test.ravel(), lion4_test.ravel(),elephant1_test.ravel(),mosquito2_test.ravel()]
#d = [cat1_test.ravel()]

# print(data_test)
pred = mlp.predict(data_test)
print(pred)