from abc import ABCMeta, abstractmethod
import json

import math
from typing import List
import tensorflow as tf


class Node(metaclass=ABCMeta):
    def __init__(self, previous_node, model_blueprint, node_blueprint):
        self.previous_node = previous_node
        self.model_blueprint = model_blueprint
        self.node_blueprint = node_blueprint
        self.name = node_blueprint["name"]
        self.node_type = node_blueprint["node_type"]
        self.input_tensor = None
        self.output_tensor = None

    def build(self):
        self.input_tensor = self._input_tensor()
        processed_input = self._process_input(self.input_tensor)
        self.output_tensor = self._calculate(processed_input)

    @abstractmethod
    def _input_tensor(self):
        pass

    @abstractmethod
    def _process_input(self, input):
        pass

    @abstractmethod
    def _calculate(self, processed_input):
        pass


class InputNode(Node):
    def __init__(self, previous_node, model_blueprint, node_blueprint, input_tensor):
        super().__init__(previous_node, model_blueprint, node_blueprint)
        self.input_tensor = input_tensor

    def _input_tensor(self):
        return self.input_tensor

    def _process_input(self, input):
        return input

    def _calculate(self, processed_input):
        return processed_input


class ActivationNode(Node, metaclass=ABCMeta):
    def __init__(self, previous_node, model_blueprint, node_blueprint):
        super().__init__(previous_node, model_blueprint, node_blueprint)
        self.activate_func = node_blueprint["activate_func"]
        self.weight_dict = node_blueprint["weight"]
        self.bias_dict = node_blueprint["bias"]

    def _input_tensor(self):
        return self.previous_node.output_tensor

    def _process_input(self, input):
        W = self._create_weight(self._w_shape())
        b = self._create_bias(self._b_shape())
        Z = self._calc_Z(input, W, b)
        return Z

    def _calculate(self, processed_input):
        return self._activate(processed_input)


    @abstractmethod
    def _w_shape(self):
        pass

    @abstractmethod
    def _b_shape(self):
        pass

    @abstractmethod
    def _calc_Z(self, input, W, b):
        pass

    @abstractmethod
    def _activate(self, Z):
        if self.activate_func == "relu":
            return tf.nn.relu(Z)
        elif self.activate_func == "softmax":
            return tf.nn.softmax(Z)
        else:
            # TODO
            return tf.nn.relu(Z)

    def _create_weight(self, shape):
        initializer = create_initializer(self.weight_dict["initializer"])
        name = "w_" + self.name
        dtype = get_type(self.weight_dict)
        decay = self.weight_dict["decay"]

        w = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        if isDecay(self.weight_dict):
            weight_decay = tf.mul(tf.nn.l2_loss(w), decay, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)  # TODO: いる？
        return w

    def _create_bias(self, shape):
        initializer = create_initializer(self.bias_dict["initializer"])
        name = "b_" + self.name
        dtype = get_type(self.bias_dict)
        b = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return b


class ConvNode(ActivationNode):
    def __init__(self, previous_node, model_blueprint, node_blueprint):
        super().__init__(previous_node, model_blueprint, node_blueprint)
        self.num_out_channel = node_blueprint["num_out_channel"]

    def _w_shape(self):
        kernel = self.weight_dict["kernel"]
        ch = self.previous_node.output_tensor.get_shape()[3]
        return [kernel[0], kernel[1], ch, self.num_out_channel]

    def _b_shape(self):
        return [self.num_out_channel]

    def _calc_Z(self, input, W, b):
        return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _activate(self, Z):
        return super()._activate(Z)


class FullConnNode(ActivationNode):
    def __init__(self, previous_node, model_blueprint, node_blueprint):
        super().__init__(previous_node, model_blueprint, node_blueprint)
        if self.node_type == "fc_output":
            self.neuron_counts = model_blueprint["num_class"]
        else:
            self.neuron_counts = node_blueprint["neuron_counts"]

    def _input_tensor(self):
        if self.node_type == "fc_input":
            batch_size = self.model_blueprint["batch_size"]
            return tf.reshape(self.previous_node.output_tensor, [batch_size, -1])
        else:
            return super()._input_tensor()

    def _w_shape(self):
        if self.node_type == "fc_input":
            dim = self.input_tensor.get_shape()[1].value
            return [dim, self.neuron_counts]
        else:
            return [self.previous_node.neuron_counts, self.neuron_counts]

    def _b_shape(self):
        return [self.neuron_counts]

    def _calc_Z(self, input, W, b):
        return tf.matmul(input, W) + b

    def _activate(self, Z):
        return super()._activate(Z)


class NormNode(Node):
    def __init__(self, previous_node, model_blueprint, node_blueprint):
        super().__init__(previous_node, model_blueprint, node_blueprint)
        self.norm_type = node_blueprint["norm_type"]
        self.params = node_blueprint["params"]

    def _input_tensor(self):
        return self.previous_node.output_tensor

    def _process_input(self, input):
        return input

    def _calculate(self, processed_input):
        if self.norm_type == "local_response_normalization":
            return tf.nn.lrn(processed_input,
                      self.params["depth_radius"],
                      self.params["bias"],
                      self.params["alpha"],
                      self.params["beta"])
        else:
            # TODO
            return processed_input


class PoolNode(Node):
    def __init__(self, previous_node, model_blueprint, node_blueprint):
        super().__init__(previous_node, model_blueprint, node_blueprint)
        self.pool_type = node_blueprint["pool_type"]
        self.ksize = node_blueprint["ksize"]
        self.strides = node_blueprint["strides"]
        self.padding = node_blueprint["padding"]

    def _input_tensor(self):
        return self.previous_node.output_tensor

    def _process_input(self, input):
        return input

    def _calculate(self, processed_input):
        if self.pool_type == "max":
            return tf.nn.max_pool(processed_input, ksize=self.ksize, strides=self.strides, padding=self.padding)
        else:
            #TODO
            return tf.nn.max_pool(processed_input, ksize=self.ksize, strides=self.strides, padding=self.padding)


def calc_pool_output_size(padding: str, in_height, in_width, filter_height, filter_width, stride_height, stride_width):
    if padding == "SAME":
        out_height = math.ceil(float(in_height) / float(stride_height))
        out_width = math.ceil(float(in_width) / float(stride_width))
    else: #padding == "VALID"
        out_height = math.ceil(float(in_height - filter_height + 1) / float(stride_height))
        out_width = math.ceil(float(in_width - filter_width + 1) / float(stride_width))
    return out_height, out_width


def isDecay(data: dict):
    if "decay" not in data:
        return False
    decay = data["decay"]
    if type(decay) == 'float' or type(decay) == 'int':
        return True
    else:
        return False


def create_initializer(data: dict):
    dtype = get_type(data)
    if data["type"] == "truncated_normal":
        return tf.truncated_normal_initializer(stddev=data["stddev"], dtype=dtype)
    elif data["type"] == "const":
        return tf.constant_initializer(value=data["value"], dtype=dtype)


def get_type(data: dict):
    if "dtype_str" not in data:
        return tf.float32
    else:
        dtype_str = data["dtype_str"]
        if dtype_str == "float32":
            return tf.float32
        else:
            #TODO
            return tf.float32


def create_model(input_tensor, model_blueprint):

    nodes_blueprint = model_blueprint["nodes"]

    nodes = []
    for i, node_blueprint in enumerate(nodes_blueprint):
        if i == 0:
            node = InputNode(previous_node=None,
                             model_blueprint=model_blueprint,
                             node_blueprint=node_blueprint,
                             input_tensor=input_tensor)
        else:
            node_type = node_blueprint["node_type"]
            if node_type == "conv":
                factory = ConvNode
            elif node_type == "pool":
                factory = PoolNode
            elif node_type == "norm":
                factory = NormNode
            else:
                factory = FullConnNode

            node = factory(previous_node=nodes[-1],
                           model_blueprint=model_blueprint,
                           node_blueprint=node_blueprint)

        node.build()
        nodes.append(node)

    return nodes[-1].output_tensor


def inference(images, keep_prob):
    """予測モデルを作成する.
    :param images: input image tensor.
    :param keep_prob:
    :return: logits tensor.
    """
    f = open("model_sample.json")
    data = json.load(f)
    f.close()
    return create_model(images, data)

