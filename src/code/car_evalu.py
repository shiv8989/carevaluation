#CNN for Car evaluation problem
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
data = np.genfromtxt("C:\\Python27\\TrainSet80%csv.csv",dtype=None, delimiter=";")
# split into input (X) and output (Y) variables
X = data[:,0:21]
Y = data[:,21:25]
z=np.genfromtxt("C:\\Python27\\TestSet20%csv.csv",dtype=None, delimiter=";")
z1=z[:,0:21]
z2=z[:,21:25]
#print X
#print Y
# create model
model = Sequential()
model.add(Dense(14,activation="sigmoid",input_shape=(21,), kernel_initializer="uniform"))
model.add(Dense(4, activation="sigmoid", kernel_initializer="uniform"))
# Compile model
sgd = optimizers.SGD(lr=0.1, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['acc'])
# Fit the model
history=model.fit(X, Y,validation_data=(z1,z2), nb_epoch=500, batch_size=1377)

predict=model.evaluate(z1,z2,verbose=0)
print("mean squared error :",predict[0])
#print("TEST ACC  :",predict[1])
rt=model.predict(z1)
print("PREDICTED",rt)
print("ORIGINAL",z2) 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
