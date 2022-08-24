import tensorflow as tf
from abc import ABCMeta, abstractmethod
import math
import numpy as np
from LorisNet.Layers.MultiAttentiveTransformer import AbstractMultiAttentiveTransformerNoFeedback, AbstractMultiAttentiveTransformerWithFeedback
from LorisNet.Layers.ParallelDense import ParallelDense
from LorisNet.Layers.MLPWithSkipConnections import MLPWithSkipConnections


class AbstractStep(tf.keras.layers.Layer, metaclass=ABCMeta):
    """"""
    
    def __init__(self,
                 output_units,
                 pass_next_step_units,
                 **kwargs):
        super().__init__(**kwargs)
        self.output_units = output_units
        self.pass_next_step_units = pass_next_step_units
    
    @abstractmethod
    def processing(self):
        pass
    
    @abstractmethod
    def call(self):
        pass

class AbstractStepNoFeedback(AbstractStep):
    """"""
    
    def __init__(self,
                 multi_attentive_transformer: AbstractMultiAttentiveTransformerNoFeedback,
                 **kwargs):
        super().__init__(**kwargs)
        self.multi_attentive_transformer = multi_attentive_transformer
        
    def call(self, inputs):
        masks = self.multi_attentive_transformer(inputs)
        selected_features = masks * tf.expand_dims(inputs, 1)
        outputs = self.processing(selected_features)
        return outputs[:, :self.output_units], outputs[:, self.output_units:], masks

class AbstractStepWithFeedback(AbstractStep):
    """"""
    
    def __init__(self,
                 multi_attentive_transformer: AbstractMultiAttentiveTransformerWithFeedback,
                 **kwargs):
        super().__init__(**kwargs)
        self.multi_attentive_transformer = multi_attentive_transformer
        
    def call(self, inputs):
        inputs_tensor, list_prior_info_tensors, list_prior_masks_tensors = inputs
        masks = self.multi_attentive_transformer(inputs)
        selected_features = masks * tf.expand_dims(inputs_tensor, 1)
        outputs = self.processing(selected_features, list_prior_info_tensors)
        return outputs[:, :self.output_units], outputs[:, self.output_units:], masks

class StepNoFeedback(AbstractStepNoFeedback):
    """"""
    
    def __init__(self,
                 dropout_rate,
                 dim_reduction,
                 kernel_regularizer,
                 number_parallel_groups,
                 MLP_with_skip_connections_depth,
                 _parallel_dense_kwargs=None,
                 _reduction_layer_kwargs=None,
                 _MLP_with_skip_connections_kwargs=None,
                 _output_layer_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dim_reduction = dim_reduction
        self.kernel_regularizer = kernel_regularizer
        self.number_parallel_groups = number_parallel_groups
        self.MLP_with_skip_connections_depth= MLP_with_skip_connections_depth
        self._parallel_dense_kwargs = _parallel_dense_kwargs
        self._reduction_layer_kwargs = _reduction_layer_kwargs
        self._MLP_with_skip_connections_kwargs = _MLP_with_skip_connections_kwargs
        self._output_layer_kwargs = _output_layer_kwargs
        
    def build(self, input_shape):
        self.an_input_tensor_dim = input_shape[1]
        
        if self._parallel_dense_kwargs is None:
            self._parallel_dense_kwargs = {
                'dim_reduction': self.dim_reduction,
                'kernel_regularizer': self.kernel_regularizer,
                'number_groups': self.number_parallel_groups
            }
        if self._reduction_layer_kwargs is None:
            self._reduction_layer_kwargs = {
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
        if self._output_layer_kwargs is None:
            self._output_layer_kwargs = {
                'units': self.output_units + self.pass_next_step_units,
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        else:
            if self._output_layer_kwargs['units'] != self.output_units + self.pass_next_step_units:
                raise ValueError("output_layer should output a tensor of size {}(output tensor size + pass next step tensor size)".format(self.output_units + self.pass_next_step_units))
        
        self.parallel_dense = ParallelDense(**self._parallel_dense_kwargs)
        self.reduction_layer = tf.keras.layers.Dense(**self._reduction_layer_kwargs)
        self.MLP_with_skip_connections = MLPWithSkipConnections(**self._MLP_with_skip_connections_kwargs)
        self.output_layer = tf.keras.layers.Dense(**self._output_layer_kwargs)
        super().build(input_shape)
    
    def processing(self, selected_features):
        processed_selected_features = self.parallel_dense(selected_features)
        reduced_tensor = self.reduction_layer(processed_selected_features)
        processed_tensor = self.MLP_with_skip_connections(reduced_tensor)
        output_tensor = self.output_layer(processed_tensor)
        return output_tensor
    
class StepWithFeedback(AbstractStepWithFeedback):
    """"""
    
    def __init__(self,
                 dropout_rate,
                 dim_reduction,
                 kernel_regularizer,
                 number_parallel_groups,
                 MLP_with_skip_connections_depth,
                 _parallel_dense_kwargs=None,
                 _prior_info_layer_kwargs=None,
                 _reduction_layer_kwargs=None,
                 _MLP_with_skip_connections_kwargs=None,
                 _output_layer_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dim_reduction = dim_reduction
        self.kernel_regularizer = kernel_regularizer
        self.number_parallel_groups = number_parallel_groups
        self.MLP_with_skip_connections_depth = MLP_with_skip_connections_depth
        self._parallel_dense_kwargs = _parallel_dense_kwargs
        self._prior_info_layer_kwargs = _prior_info_layer_kwargs
        self._reduction_layer_kwargs = _reduction_layer_kwargs
        self._MLP_with_skip_connections_kwargs = _MLP_with_skip_connections_kwargs
        self._output_layer_kwargs = _output_layer_kwargs
        
    def build(self, input_shape):
        self.inputs_tensor_dim, self.list_prior_info_tensors_dim, self.list_prior_masks_tensors_dim = input_shape
        self.an_input_tensor_dim = self.inputs_tensor_dim[1]
        
        if self._parallel_dense_kwargs is None:
            self._parallel_dense_kwargs = {
                'dim_reduction': self.dim_reduction,
                'kernel_regularizer': self.kernel_regularizer,
                'number_groups': self.number_parallel_groups
            }
        if self._prior_info_layer_kwargs is None:
            self._prior_info_layer_kwargs = {
                'units': math.ceil(np.array(self.list_prior_info_tensors_dim)[:, 1].sum()**self.dim_reduction),
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        if self._reduction_layer_kwargs is None:
            self._reduction_layer_kwargs = {
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
        if self._output_layer_kwargs is None:
            self._output_layer_kwargs = {
                'units': self.output_units + self.pass_next_step_units,
                'activation': 'relu',
                'kernel_regularizer': self.kernel_regularizer
            }
        else:
            if self._output_layer_kwargs['units'] != self.output_units + self.pass_next_step_units:
                raise ValueError("output_layer should output a tensor of size {}(output tensor size + pass next step tensor size)".format(self.output_units + self.pass_next_step_units))
        
        self.parallel_dense = ParallelDense(**self._parallel_dense_kwargs)
        self.prior_info_layer = tf.keras.layers.Dense(**self._prior_info_layer_kwargs)
        self.reduction_layer = tf.keras.layers.Dense(**self._reduction_layer_kwargs)
        self.MLP_with_skip_connections = MLPWithSkipConnections(**self._MLP_with_skip_connections_kwargs)
        self.output_layer = tf.keras.layers.Dense(**self._output_layer_kwargs)
        super().build(input_shape)
    
    def processing(self, selected_features, list_prior_info_tensors):
        processed_selected_features = self.parallel_dense(selected_features)
        processed_prior_info = self.prior_info_layer(tf.keras.layers.Concatenate()(list_prior_info_tensors))
        reduced_tensor = self.reduction_layer(tf.keras.layers.Concatenate()([processed_selected_features,
                                                                             processed_prior_info]))
        processed_tensor = self.MLP_with_skip_connections(reduced_tensor)
        output_tensor = self.output_layer(processed_tensor)
        return output_tensor