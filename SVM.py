import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir = 'data'

catagories = ['bacterial leaf blight','brownspot', 'rice blast', 'sheathblight']

data = []

for catagory in catagories:
    path  = os.path.join(dir,catagory)
    label=catagories.index(catagory)

    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        petimage= cv2.imread(imgpath,0)
        try:
            petimage=cv2.resize(petimage,(50,50))
            image = np.array(petimage).flatten()

            data.append([image,label])
        except Exception as e:
            pass

print(len(data))

pick_in = open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()

pick_out = open('data1.pickle','rb')
data_img = pickle.load(pick_out)
pick_in.close()

random.shuffle(data_img)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(labels)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.50)

model = SVC(C=1,kernel = 'poly', gamma = 'auto')
model.fit(xtrain,ytrain)
print(model)

prediction = model.predict(xtest,)
accuracy = model.score(xtest, ytest)

print('Accuricy: ', accuracy)

print('Prediction is : ', catagories[prediction[0]])

mytest = xtest[0].reshape(50,50)
plt.show(mytest, cmap='gray')
plt.show()

pick = open('model.sav', 'wb')
pickle.dump(model,pick)
pick.close()
