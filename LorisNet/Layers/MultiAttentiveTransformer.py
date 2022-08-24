import tensorflow as tf
from tensorflow_addons.activations import sparsemax
from abc import ABCMeta, abstractmethod
import math
import numpy as np
from LorisNet.Layers.MLPWithSkipConnections import MLPWithSkipConnections


class AbstractMultiAttentiveTransformer(tf.keras.layers.Layer, metaclass=ABCMeta):
    """"""
    
    def __init__(self,
                 number_masks,
                 masks_activation=sparsemax,
                 **kwargs):
        super().__init__(**kwargs)
        self.number_masks = number_masks
        self.masks_activation = tf.keras.activations.get(masks_activation)
        
    @abstractmethod
    def processing(self):
        pass
    
    @abstractmethod
    def call(self):
        pass
        
class AbstractMultiAttentiveTransformerNoFeedback(AbstractMultiAttentiveTransformer):
    """"""
    
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        processed_inputs = self.processing(inputs)
        masks = self.masks_activation(processed_inputs)
        return masks
        
class AbstractMultiAttentiveTransformerWithFeedback(AbstractMultiAttentiveTransformer):
    """"""
    
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def gammas_computing(self):
        pass
                       
    @abstractmethod
    def prior_masks_scales_computing(self):
        pass
    
    def call(self, inputs):
        inputs_tensor, list_prior_info_tensors, list_prior_masks_tensors = inputs
        processed_inputs_and_prior_info = self.processing(inputs_tensor, list_prior_info_tensors)
        prior_masks_scales = self.prior_masks_scales_computing(list_prior_masks_tensors)
        masks = self.masks_activation(processed_inputs_and_prior_info * prior_masks_scales)
        return masks
    
class MultiAttentiveTransformerNoFeedback(AbstractMultiAttentiveTransformerNoFeedback):
    """"""
    
    def __init__(self,
                 dropout_rate,
                 dim_reduction,
                 activity_regularizer,
                 kernel_regularizer,
                 MLP_with_skip_connections_depth,
                 _first_layer_kwargs=None,
                 _MLP_with_skip_connections_kwargs=None,
                 _reduction_layer_kwargs=None,
                 _premask_layer_kwargs=None,
                 **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.dropout_rate = dropout_rate
        self.dim_reduction = dim_reduction
        self.kernel_regularizer = kernel_regularizer
        self.MLP_with_skip_connections_depth = MLP_with_skip_connections_depth
        self._first_layer_kwargs = _first_layer_kwargs
        self._MLP_with_skip_connections_kwargs = _MLP_with_skip_connections_kwargs
        self._reduction_layer_kwargs = _reduction_layer_kwargs
        self._premask_layer_kwargs = _premask_layer_kwargs
    
    def build(self, input_shape):
        self.an_input_tensor_dim = input_shape[1]
        
        if self._first_layer_kwargs is None:
            self._first_layer_kwargs = {
                'units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        if self._MLP_with_skip_connections_kwargs is None:
            self._MLP_with_skip_connections_kwargs = {
                'depth': self.MLP_with_skip_connections_depth,
                'width': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                'dropout_rate': self.dropout_rate,
                '_dense_kwargs': {
                    'activation': 'relu',
                    'kernel_regularizer': self.kernel_regularizer
                }
            }
        if self._reduction_layer_kwargs is None:
            self._reduction_layer_kwargs = {
                'units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        if self._premask_layer_kwargs is None:
            self._premask_layer_kwargs = {
                'units': self.an_input_tensor_dim*self.number_masks,
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        else:
            if self._premask_layer_kwargs['units'] != self.an_input_tensor_dim*self.number_masks:
                raise ValueError("premask_layer should output a tensor of size {}(input tensor size * number of masks)".format(self.an_input_tensor_dim*self.number_masks))
        
        self.first_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.first_layer = tf.keras.layers.Dense(**self._first_layer_kwargs)
        self.MLP_with_skip_connections = MLPWithSkipConnections(**self._MLP_with_skip_connections_kwargs)
        self.last_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.reduction_layer = tf.keras.layers.Dense(**self._reduction_layer_kwargs)
        self.premask_layer = tf.keras.layers.Dense(**self._premask_layer_kwargs)
        super().build(input_shape)
        
    def processing(self, inputs):
        x = self.first_dropout(inputs)
        x = self.first_layer(x)
        x = self.MLP_with_skip_connections(x)
        x = self.last_dropout(x)
        x = self.reduction_layer(x)
        x = self.premask_layer(x)
        premasks = tf.reshape(x, [-1, self.number_masks, self.an_input_tensor_dim])
        return premasks

class MultiAttentiveTransformerWithFeedback(AbstractMultiAttentiveTransformerWithFeedback):
    """"""
    
    def __init__(self,
                 dropout_rate,
                 dim_reduction,
                 activity_regularizer,
                 kernel_regularizer,
                 MLP_with_skip_connections_depth,
                 gammas_computing_kwargs=None,
                 _prior_info_layer_kwargs=None,
                 _first_layer_kwargs=None,
                 _MLP_with_skip_connections_kwargs=None,
                 _reduction_layer_kwargs=None,
                 _premask_layer_kwargs=None,
                 **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.dropout_rate = dropout_rate
        self.dim_reduction = dim_reduction
        self.kernel_regularizer = kernel_regularizer
        self.MLP_with_skip_connections_depth = MLP_with_skip_connections_depth
        if gammas_computing_kwargs is None:
            self.gammas_computing_kwargs = {}
        else:
            self.gammas_computing_kwargs = gammas_computing_kwargs
        self.gammas_list = self.gammas_computing(self.number_masks,
                                                 **self.gammas_computing_kwargs)
        self._prior_info_layer_kwargs = _prior_info_layer_kwargs
        self._first_layer_kwargs = _first_layer_kwargs
        self._MLP_with_skip_connections_kwargs = _MLP_with_skip_connections_kwargs
        self._reduction_layer_kwargs = _reduction_layer_kwargs
        self._premask_layer_kwargs = _premask_layer_kwargs
    
    def build(self, input_shape):
        self.inputs_tensor_dim, self.list_prior_info_tensors_dim, self.list_prior_masks_tensors_dim = input_shape
        self.an_input_tensor_dim = self.inputs_tensor_dim[1]
        
        if self._prior_info_layer_kwargs is None:
            self._prior_info_layer_kwargs = {
                'units': math.ceil(np.array(self.list_prior_info_tensors_dim)[:, 1].sum()**self.dim_reduction),
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        if self._first_layer_kwargs is None:
            self._first_layer_kwargs = {
                'units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        if self._MLP_with_skip_connections_kwargs is None:
            self._MLP_with_skip_connections_kwargs = {
                'depth': self.MLP_with_skip_connections_depth,
                'width': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                'dropout_rate': self.dropout_rate,
                '_dense_kwargs': {
                    'activation': 'relu',
                    'kernel_regularizer': self.kernel_regularizer
                }
            }
        if self._reduction_layer_kwargs is None:
            self._reduction_layer_kwargs = {
                'units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        if self._premask_layer_kwargs is None:
            self._premask_layer_kwargs = {
                'units': self.an_input_tensor_dim*self.number_masks,
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        else:
            if self._premask_layer_kwargs['units'] != self.an_input_tensor_dim*self.number_masks:
                raise ValueError("premask_layer should output a vector of size {}(input vector size * number of masks)".format(self.an_input_tensor_dim*self.number_masks))
        
        self.prior_info_layer = tf.keras.layers.Dense(**self._prior_info_layer_kwargs)
        self.first_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.first_layer = tf.keras.layers.Dense(**self._first_layer_kwargs)
        self.MLP_with_skip_connections = MLPWithSkipConnections(**self._MLP_with_skip_connections_kwargs)
        self.last_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.reduction_layer = tf.keras.layers.Dense(**self._reduction_layer_kwargs)
        self.premask_layer = tf.keras.layers.Dense(**self._premask_layer_kwargs)
        super().build(input_shape)
        
    def processing(self, inputs_tensor, list_prior_info_tensors):
        prior_info_tensors = tf.keras.layers.Concatenate()(list_prior_info_tensors)
        prior_info_tensors = self.prior_info_layer(prior_info_tensors)
        x = tf.keras.layers.Concatenate()([inputs_tensor, prior_info_tensors])
        x = self.first_dropout(x)
        x = self.first_layer(x)
        x = self.MLP_with_skip_connections(x)
        x = self.last_dropout(x)
        x = self.reduction_layer(x)
        x = self.premask_layer(x)
        premasks = tf.reshape(x, [-1, self.number_masks, self.an_input_tensor_dim])
        return premasks
    
    def gammas_computing(self, number_masks, gammas_range=[.5, 1.5]):
        return np.linspace(start=gammas_range[0], stop=gammas_range[1], num=number_masks, dtype=np.float32)
    
    def prior_masks_scales_computing(self, list_prior_masks_tensors):
        mean = tf.reduce_mean(list_prior_masks_tensors, [0, 1, 2])
        return tf.pow(self.gammas_list.reshape((-1, 1)), mean)