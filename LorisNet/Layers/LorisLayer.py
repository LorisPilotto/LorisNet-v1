import tensorflow as tf
import math
from LorisNet.Layers.Step import AbstractStepNoFeedback, AbstractStepWithFeedback, StepNoFeedback, StepWithFeedback
from LorisNet.Layers.LSTMCell import LSTMCell
from LorisNet.Layers.MultiAttentiveTransformer import AbstractMultiAttentiveTransformerNoFeedback, AbstractMultiAttentiveTransformerWithFeedback, MultiAttentiveTransformerNoFeedback, MultiAttentiveTransformerWithFeedback


"""class LorisLayer(tf.keras.layers.Layer):
    
    def __init__(self,
                 units,
                 number_steps=5,
                 number_parallel_AT=200,
                 dim_reduction=.7,
                 dropout_rate=.1,
                 multi_attentive_transformer_activity_regularizer=tf.keras.regularizers.L2(),
                 kernel_regularizer=tf.keras.regularizers.L2(),
                 number_parallel_groups=-1,
                 MLP_with_skip_connections_depth=4,
                 _first_step_kwargs=None,
                 first_step: AbstractStepNoFeedback=StepNoFeedback,
                 _with_feedback_steps_kwargs=None,
                 with_feedback_steps: AbstractStepWithFeedback=StepWithFeedback,
                 _list_steps=None,
                 _sequential_output_processing_kwargs=None,
                 sequential_output_processing=LSTMCell,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.number_steps = number_steps
        self.number_parallel_AT = number_parallel_AT
        self.dim_reduction= dim_reduction
        self.dropout_rate = dropout_rate
        self.multi_attentive_transformer_activity_regularizer = multi_attentive_transformer_activity_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.number_parallel_groups = number_parallel_groups
        self.MLP_with_skip_connections_depth = MLP_with_skip_connections_depth
        self._first_step_kwargs = _first_step_kwargs
        self.first_step = first_step
        self._with_feedback_steps_kwargs = _with_feedback_steps_kwargs
        self.with_feedback_steps = with_feedback_steps
        if _list_steps is not None:
            self.number_steps = len(_list_steps)
            self._list_steps = _list_steps
        else:
            self._list_steps = _list_steps
        self._sequential_output_processing_kwargs = _sequential_output_processing_kwargs
        self.sequential_output_processing = sequential_output_processing
        
    def build(self, input_shape):
        self.an_input_tensor_dim = input_shape[1]
        
        if self._list_steps is None:
            self._list_steps = []
            if self._first_step_kwargs is None:
                self._first_step_kwargs = {
                    'output_units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                    'pass_next_step_units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                    'multi_attentive_transformer': MultiAttentiveTransformerNoFeedback(number_masks=self.number_parallel_AT,
                                                                                       dropout_rate=self.dropout_rate,
                                                                                       dim_reduction=self.dim_reduction,
                                                                                       activity_regularizer=self.multi_attentive_transformer_activity_regularizer,
                                                                                       kernel_regularizer=self.kernel_regularizer,
                                                                                       MLP_with_skip_connections_depth=self.MLP_with_skip_connections_depth),
                    'dropout_rate': self.dropout_rate,
                    'dim_reduction': self.dim_reduction,
                    'kernel_regularizer': self.kernel_regularizer,
                    'number_parallel_groups': self.number_parallel_groups,
                    'MLP_with_skip_connections_depth': self.MLP_with_skip_connections_depth
                }
            self._list_steps.append(self.first_step(**self._first_step_kwargs))
            for s in range(1, self.number_steps):
                if self._with_feedback_steps_kwargs is None:
                    self._with_feedback_steps_kwargs = {
                        'output_units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                        'pass_next_step_units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                        'multi_attentive_transformer': MultiAttentiveTransformerWithFeedback(number_masks=self.number_parallel_AT,
                                                                                             dropout_rate=self.dropout_rate,
                                                                                             dim_reduction=self.dim_reduction,
                                                                                           activity_regularizer=self.multi_attentive_transformer_activity_regularizer,
                                                                                            kernel_regularizer=self.kernel_regularizer,
                                                                                            MLP_with_skip_connections_depth=self.MLP_with_skip_connections_depth),
                        'dropout_rate': self.dropout_rate,
                        'dim_reduction': self.dim_reduction,
                        'kernel_regularizer': self.kernel_regularizer,
                        'number_parallel_groups': self.number_parallel_groups,
                        'MLP_with_skip_connections_depth': self.MLP_with_skip_connections_depth
                    }
                self._list_steps.append(self.with_feedback_steps(**self._with_feedback_steps_kwargs))
            # self._list_steps = [self.first_step(**self._first_step_kwargs),
            #                    *[self.with_feedback_steps(**self._with_feedback_steps_kwargs) for l in range(1, self.number_steps)]]
        
        if self._sequential_output_processing_kwargs is None:
            self._sequential_output_processing_kwargs = {
                'units': self.units
            }
        self.sequential_output_processing_layer = self.sequential_output_processing(**self._sequential_output_processing_kwargs)
        super().build(input_shape)
        
    def forward(self, inputs):
        sequential_output_processing_layer_states = [tf.zeros((tf.shape(inputs)[0], self.units)),
                                                     tf.zeros((tf.shape(inputs)[0], self.units))]
        
        output_step, pass_next_step, masks = self._list_steps[0](inputs)
        list_prior_info_tensors = [pass_next_step]
        list_prior_masks_tensors = [masks]
        output, sequential_output_processing_layer_states = self.sequential_output_processing_layer(output_step,
                                                                                                    sequential_output_processing_layer_states)
        
        for s in range(1, self.number_steps):
            output_step, pass_next_step, masks = self._list_steps[s]([inputs,
                                                                      list_prior_info_tensors,
                                                                      list_prior_masks_tensors])
            list_prior_info_tensors.append(pass_next_step)
            list_prior_masks_tensors.append(masks)
            output, sequential_output_processing_layer_states = self.sequential_output_processing_layer(output_step,
                                                                                                        sequential_output_processing_layer_states)
            
        return output, list_prior_info_tensors, list_prior_masks_tensors, sequential_output_processing_layer_states
    
    def call(self, inputs):
        return self.forward(inputs)[0]
    
    def get_list_prior_info_tensors(self, inputs):
        return self.forward(inputs)[1]
    
    def get_list_prior_masks_tensors(self, inputs):
        return self.forward(inputs)[2]
    
    def get_sequential_output_processing_layer_states(self, inputs):
        return self.forward(inputs)[3]"""
            
            
        
class LorisLayer(tf.keras.layers.Layer):
    
    def __init__(self,
                 units,
                 number_steps=5,
                 number_parallel_AT=200,
                 dim_reduction=.7,
                 dropout_rate=.1,
                 multi_attentive_transformer_activity_regularizer=tf.keras.regularizers.L2(),
                 kernel_regularizer=tf.keras.regularizers.L2(),
                 number_parallel_groups=-1,
                 MLP_with_skip_connections_depth=4,
                 _first_step_kwargs=None,
                 _first_step: AbstractStepNoFeedback=StepNoFeedback,
                 _first_multi_attentive_transformer_kwargs=None,
                 _first_multi_attentive_transformer: AbstractMultiAttentiveTransformerNoFeedback=MultiAttentiveTransformerNoFeedback,
                 _with_feedback_steps_kwargs=None,
                 _with_feedback_steps: AbstractStepWithFeedback=StepWithFeedback,
                 _with_feedback_multi_attentive_transformer_kwargs=None,
                 _with_feedback_multi_attentive_transformer: AbstractMultiAttentiveTransformerWithFeedback=MultiAttentiveTransformerWithFeedback,
                 _list_steps=None,
                 _sequential_output_processing_kwargs=None,
                 sequential_output_processing=LSTMCell,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.number_steps = number_steps
        self.number_parallel_AT = number_parallel_AT
        self.dim_reduction= dim_reduction
        self.dropout_rate = dropout_rate
        self.multi_attentive_transformer_activity_regularizer = multi_attentive_transformer_activity_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.number_parallel_groups = number_parallel_groups
        self.MLP_with_skip_connections_depth = MLP_with_skip_connections_depth
        self._first_step_kwargs = _first_step_kwargs
        self._first_step = _first_step
        self._first_multi_attentive_transformer_kwargs = _first_multi_attentive_transformer_kwargs
        self._first_multi_attentive_transformer = _first_multi_attentive_transformer
        self._with_feedback_steps_kwargs = _with_feedback_steps_kwargs
        self._with_feedback_steps = _with_feedback_steps
        self._with_feedback_multi_attentive_transformer_kwargs = _with_feedback_multi_attentive_transformer_kwargs
        self._with_feedback_multi_attentive_transformer = _with_feedback_multi_attentive_transformer
        if _list_steps is not None:
            self.number_steps = len(_list_steps)
            self._list_steps = _list_steps
        else:
            self._list_steps = _list_steps
        self._sequential_output_processing_kwargs = _sequential_output_processing_kwargs
        self.sequential_output_processing = sequential_output_processing
        
    def build(self, input_shape):
        self.an_input_tensor_dim = input_shape[1]
        
        if self._list_steps is None:
            self._list_steps = []
            if self._first_step_kwargs is None:
                self._first_step_kwargs = {
                    'output_units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                    'pass_next_step_units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                    'dropout_rate': self.dropout_rate,
                    'dim_reduction': self.dim_reduction,
                    'kernel_regularizer': self.kernel_regularizer,
                    'number_parallel_groups': self.number_parallel_groups,
                    'MLP_with_skip_connections_depth': self.MLP_with_skip_connections_depth
                }
            if self._first_multi_attentive_transformer_kwargs is None:
                self._first_multi_attentive_transformer_kwargs = {
                    'number_masks': self.number_parallel_AT,
                    'dropout_rate': self.dropout_rate,
                    'dim_reduction': self.dim_reduction,
                    'activity_regularizer': self.multi_attentive_transformer_activity_regularizer,
                    'kernel_regularizer': self.kernel_regularizer,
                    'MLP_with_skip_connections_depth': self.MLP_with_skip_connections_depth
                }
            self._list_steps.append(self._first_step(multi_attentive_transformer=self._first_multi_attentive_transformer(**self._first_multi_attentive_transformer_kwargs), **self._first_step_kwargs))
            if self._with_feedback_steps_kwargs is None:
                self._with_feedback_steps_kwargs = {
                    'output_units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                    'pass_next_step_units': math.ceil(self.an_input_tensor_dim**self.dim_reduction),
                    'dropout_rate': self.dropout_rate,
                    'dim_reduction': self.dim_reduction,
                    'kernel_regularizer': self.kernel_regularizer,
                    'number_parallel_groups': self.number_parallel_groups,
                    'MLP_with_skip_connections_depth': self.MLP_with_skip_connections_depth
                }
            if self._with_feedback_multi_attentive_transformer_kwargs is None:
                self._with_feedback_multi_attentive_transformer_kwargs = {
                    'number_masks': self.number_parallel_AT,
                    'dropout_rate': self.dropout_rate,
                    'dim_reduction': self.dim_reduction,
                    'activity_regularizer': self.multi_attentive_transformer_activity_regularizer,
                    'kernel_regularizer': self.kernel_regularizer,
                    'MLP_with_skip_connections_depth': self.MLP_with_skip_connections_depth
                }
            for s in range(1, self.number_steps):
                self._list_steps.append(self._with_feedback_steps(multi_attentive_transformer=self._with_feedback_multi_attentive_transformer(**self._with_feedback_multi_attentive_transformer_kwargs), **self._with_feedback_steps_kwargs))
        if self._sequential_output_processing_kwargs is None:
            self._sequential_output_processing_kwargs = {
                'units': self.units
            }
        self.sequential_output_processing_layer = self.sequential_output_processing(**self._sequential_output_processing_kwargs)
        super().build(input_shape)
        
    def forward(self, inputs):
        sequential_output_processing_layer_states = [tf.zeros((tf.shape(inputs)[0], self.units)),
                                                     tf.zeros((tf.shape(inputs)[0], self.units))]
        
        output_step, pass_next_step, masks = self._list_steps[0](inputs)
        list_prior_info_tensors = [pass_next_step]
        list_prior_masks_tensors = [masks]
        output, sequential_output_processing_layer_states = self.sequential_output_processing_layer(output_step,
                                                                                                    sequential_output_processing_layer_states)
        
        for s in range(1, self.number_steps):
            output_step, pass_next_step, masks = self._list_steps[s]([inputs,
                                                                      list_prior_info_tensors,
                                                                      list_prior_masks_tensors])
            list_prior_info_tensors.append(pass_next_step)
            list_prior_masks_tensors.append(masks)
            output, sequential_output_processing_layer_states = self.sequential_output_processing_layer(output_step,
                                                                                                        sequential_output_processing_layer_states)
            
        return output, list_prior_info_tensors, list_prior_masks_tensors, sequential_output_processing_layer_states
    
    def call(self, inputs):
        return self.forward(inputs)[0]
    
    def get_list_prior_info_tensors(self, inputs):
        return self.forward(inputs)[1]
    
    def get_list_prior_masks_tensors(self, inputs):
        return self.forward(inputs)[2]
    
    def get_sequential_output_processing_layer_states(self, inputs):
        return self.forward(inputs)[3]