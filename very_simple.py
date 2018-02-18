# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("stupid-simple.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:2]
Y = dataset[:,2]

# verify
verify = numpy.loadtxt("stupid-simple-verify.csv", delimiter=",")
ZX = dataset[:,0:2]
ZY = dataset[:,2]

# create model
model = Sequential()
model.add(Dense(15, input_dim=2, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

print("\n")

# evaluate the model
print('on training data')
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print("\n")

print('on independant test data')
scores = model.evaluate(ZX, ZY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
