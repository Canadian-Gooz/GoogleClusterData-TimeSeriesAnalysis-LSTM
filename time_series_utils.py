import matplotlib.pyplot as plt
import numpy as np
from tensorflow import stack
from tensorflow.keras.utils import timeseries_dataset_from_array
import pandas as pd


class WindowGen:
    def __init__(self, input_size, offset=0, label_size=1):
        self.input_size = input_size
        self.offset = offset
        self.label_size = label_size

        self.total_size = self.input_size+self.offset+self.label_size
        self.label_start = self.input_size+self.offset

    def __call__(self, data, column_names=None,label_names=None):
        '''
        '''
        self.indices = {name: i for i,name in enumerate(data.columns)}
       
        ds = timeseries_dataset_from_array(data=data,targets=None,sequence_length=self.total_size,
            sequence_stride=1,batch_size=len(data))
        for batch in ds:
            ds = batch
        inputs = ds[:,:self.input_size,:]
        labels = ds[:,self.label_start:self.label_start+self.label_size,:]

        
        if column_names is not None:
            self.column_indices = {name: i for i, name in enumerate(column_names)}
            inputs = stack([inputs[:,:,self.indices[name]] for name in column_names],-1)
        if label_names is not None:
            self.label_indices = {name: i for i,name in enumerate(label_names)}
            labels = stack([labels[:,:,self.indices[name]] for name in label_names],-1)
        self.plt_data = (inputs,labels)
        return inputs,labels

    def plot(self, model=None, plt_col=None, max_subplots=3):

        if self.plt_data is None:
            print('Please provide a dataset by calling the instance')
            return
        elif plt_col is None:
            if len(self.column_indices)>1:
                print('Please provide a specific column to plot')
                return
            else:
                plt_col = next(iter(self.column_indices))
        
        inputs, labels = self.plt_data
        plt.figure(figsize=(12, 8))
        plt_col_index = self.indices[plt_col]
        

        max_n = min(max_subplots, len(inputs))
        rand_n = np.random.randint(0,high=inputs.shape[0],size=(max_n,))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plt_col}')
            if self.column_indices:
                plt_col_index = self.column_indices[plt_col]
            plt.plot(np.arange(0,self.input_size), inputs[rand_n[n], :, plt_col_index],
                        label='Input', marker='.', zorder=-10)
            if self.label_indices:
                label_col_index = self.label_indices.get(plt_col, None)
            else:
                label_col_index = plt_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_start, labels[rand_n[n], :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_start, predictions[rand_n[n], :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Timeslots')
