import tensorflow as tf
import math
import numpy as np


"""class ParallelDense(tf.keras.layers.Layer):
    
    def __init__(self,
                 dim_reduction,
                 kernel_regularizer,
                 _dense_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_reduction = dim_reduction
        self.kernel_regularizer = kernel_regularizer
        self._dense_kwargs = _dense_kwargs
    
    def build(self, input_shape):
        self.num_layers, self.num_features = input_shape[1:]
        if self._dense_kwargs is None:
            self._dense_kwargs = {'units': math.ceil(self.num_features**self.dim_reduction),
                                 'activation': 'relu',
                                 'kernel_regularizer': self.kernel_regularizer}
        self.list_dense_layers = [tf.keras.layers.Dense(**self._dense_kwargs) for a_layer_id in range(self.num_layers)]
        super().build(input_shape)
        
    def call(self, inputs):
        parallel_outputs = [self.list_dense_layers[a_layer_id](inputs[:, a_layer_id]) for a_layer_id in range(self.num_layers)]
        return tf.keras.layers.Concatenate()(parallel_outputs)"""

class ParallelDense(tf.keras.layers.Layer):
    
    def __init__(self,
                 dim_reduction,
                 kernel_regularizer,
                 number_groups=-1,
                 _dense_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_reduction = dim_reduction
        self.kernel_regularizer = kernel_regularizer
        self.number_groups = number_groups
        self._dense_kwargs = _dense_kwargs
    
    def build(self, input_shape):
        self.number_masks, self.num_features = input_shape[1:]
        
        if self._dense_kwargs is None:
            self._dense_kwargs = {'units': math.ceil(self.num_features**self.dim_reduction),
                                 'activation': 'relu',
                                 'kernel_regularizer': self.kernel_regularizer}
        
        if self.number_groups == -1:
            self.list_dense_layers = [tf.keras.layers.Dense(**self._dense_kwargs) for a_layer_id in range(self.number_masks)]
        else:
            self.number_per_group = np.linspace(0,
                                                self.number_masks,
                                                num=self.number_groups+1,
                                                dtype=int)
            self.number_per_group = self.number_per_group[1:] - self.number_per_group[:-1]
            self.list_dense_layers = []
            for a_group_size in self.number_per_group:
                layer = tf.keras.layers.Dense(**self._dense_kwargs)
                for _ in range(a_group_size):
                    self.list_dense_layers.append(layer)
        super().build(input_shape)
        
    def call(self, inputs):
        parallel_outputs = [self.list_dense_layers[a_layer_id](inputs[:, a_layer_id]) for a_layer_id in range(self.number_masks)]
        return tf.keras.layers.Concatenate()(parallel_outputs)