"""
Model definition
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, LSTM
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

class CNN_TF:
    def __init__(self,**args):
        self.filters = args['filters'] # default 16
        self.kernel_size = args['kernel_size'] # default 2
        self.input_shape = args['input_shape'] # (height, width, channels)
        self.pool_size = args['pool_size'] # default 2
        self.dropout = args['dropout']
        self.activation = args['activation']
        self.num_labels = args['num_labels']
        self.loss = args['loss']
        self.optimizer = args['optimizer']
        self.metrics = args['metrics']
    def build(self):
        model = Sequential()
        model.add(Conv2D(filters=self.filters, kernel_size=self.kernel_size, input_shape=self.input_shape, activation=self.activation))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(filters=self.filters*2, kernel_size=self.kernel_size, activation=self.activation))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(self.dropout))
    
        model.add(Conv2D(filters=self.filters*4, kernel_size=self.kernel_size, activation=self.activation))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(self.dropout))
    
        model.add(Flatten())
        model.add(Dense(self.num_labels, activation='softmax'))
        
        model.compile(loss=self.loss, 
                           optimizer=self.optimizer, 
                           metrics=[self.metrics]) 
        num_params = model.count_params()
        print('# network parameters: ' + str(num_params))
        model.summary()
        self.model = model
    def fit(self,x,y,batch_size=None,epochs=1,verbose=1,callbacks=None,validation_data=None):
        self.History = self.model.fit(x,y,batch_size=batch_size,epochs=epochs,verbose=verbose,
                                 validation_data=validation_data, callbacks=callbacks)
        return self.History
    
    def evaluate(self,x,y,batch_size=None,verbose=1):
        return self.model.evaluate(x,y,batch_size=batch_size,verbose=verbose)
    
    def predict(self,x,batch_size=None,verbose=0):
        return self.model.predict(x,batch_size=batch_size,verbose=verbose)

class LSTM_TF:
    def __init__(self,**args):
        self.units = args['units'] # default 16
        self.input_shape = args['input_shape'] # (height, width, channels)
        self.dropout = args['dropout']
        self.num_labels = args['num_labels']
        self.loss = args['loss']
        self.optimizer = args['optimizer']
        self.metrics = args['metrics']
    def build(self):
        model = Sequential()
        model.add(LSTM(units=self.units,input_shape=self.input_shape,dropout=self.dropout))
        model.add(Dense(self.num_labels, activation='softmax'))
        
        model.compile(loss=self.loss, 
                           optimizer=self.optimizer, 
                           metrics=[self.metrics]) 
        num_params = model.count_params()
        print('# network parameters: ' + str(num_params))
        model.summary()
        self.model = model
    def fit(self,x,y,batch_size=None,epochs=1,verbose=1,callbacks=None,validation_data=None):
        self.History = self.model.fit(x,y,batch_size=batch_size,epochs=epochs,verbose=verbose,
                                 validation_data=validation_data, callbacks=callbacks)
        return self.History
    
    def evaluate(self,x,y,batch_size=None,verbose=1):
        return self.model.evaluate(x,y,batch_size=batch_size,verbose=verbose)
    
    def predict(self,x,batch_size=None,verbose=0):
        return self.model.predict(x,batch_size=batch_size,verbose=verbose)