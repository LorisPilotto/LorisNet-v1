import tensorflow as tf


class MLPWithSkipConnections(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 depth,
                 width,
                 dropout_rate,
                 _dense_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.width = width
        self.dropout_rate = dropout_rate
        if _dense_kwargs is None:
            self._dense_kwargs = {}
        else:
            self._dense_kwargs = _dense_kwargs
        
        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate) for layer_nbr in range(self.depth)]
        self.dense_layers = [tf.keras.layers.Dense(self.width, **self._dense_kwargs) for layer_nbr in range(self.depth)]
        
    def call(self, inputs):
        # ret = [inputs]
        for layer_nbr in range(self.depth):
            inputs = self.dropout_layers[layer_nbr](inputs)
            inputs = self.dense_layers[layer_nbr](inputs)
            # ret.append(inputs)
        # return tf.keras.layers.Concatenate()(ret)
        return inputs