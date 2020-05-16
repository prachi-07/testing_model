import pandas as pd
df = pd.read_csv('wines.csv')
df.info()
y = df['Class']
y.value_counts()
y_cat = pd.get_dummies(y)
y
df.columns
X = df.drop('Class' , axis=1)
X.info()
import seaborn as sns
sns.scatterplot(x='Alcohol' , y=y , data=df)
from keras.models import Sequential
model  =  Sequential()
X.info()
X.shape
y_cat.shape
from keras.layers import Dense
model.add(Dense(units=5 , input_shape=(13,),activation='relu',kernel_initializer='he_normal' ))
model.summary()
model.add(Dense(units=8 ,activation='relu',kernel_initializer='he_normal' ))
model.summary()
model.add(Dense(units=2,activation='relu',kernel_initializer='he_normal' ))
model.summary()
model.add(Dense(units=3, activation='softmax'))
model.summary()
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(learning_rate=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
model.layers[0].input
model.layers[3].output
model.layers[2].output
model.fit(X,y_cat, epochs=100)
model.get_weights()
model.save('modelsave.h5')