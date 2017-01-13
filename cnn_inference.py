from abc import ABCMeta, abstractmethod
import json
import inspect
import sys

import math
from typing import List
import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

Encoders = {}


class Model:
    def __init__(self, num_class, channel, batch_size, image_size, input_tensor):
        self.num_class = num_class
        self.channel = channel
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_tensor = input_tensor

class Node:
    def __init__(self, name, input, op, attr):
        self.name = name
        self.input = input
        self.op = op
        self.attr = attr


class EncodedNode:
    def __init__(self, name, output_tensor):
        self.name = name
        self.output_tensor = output_tensor


def parse_and_create_node(model_blueprint, image_tensor):
    model = Model(model_blueprint["num_class"],
                  model_blueprint["channel"],
                  model_blueprint["batch_size"],
                  model_blueprint["image_size"],
                  image_tensor)

    DAG = nx.DiGraph()
    nodes_blueprint = model_blueprint["nodes"]
    for node_blueprint in nodes_blueprint:
        node = Node(node_blueprint["name"],
                    node_blueprint["input"],
                    node_blueprint["op"],
                    node_blueprint["attr"])
        DAG.add_node(node.name, blueprint=node)
        print(node.attr)
        for in_node in node.input:
            DAG.add_edge(in_node, node.name)

    ns = nx.topological_sort(DAG)
    #print(ns)
    #print(nx.is_directed_acyclic_graph(DAG))

    NS = {}
    for n_name in ns:
        node = DAG.node[n_name]["blueprint"]
        op_name = DAG.node[n_name]["blueprint"].op
        encoder = Encoders[op_name]
        N = encoder.encode(model, node, NS)
        NS[n_name] = N

    #nx.draw(DAG)
    #plt.show()
    last_node = NS[ns[-1]]

    return last_node.output_tensor


class Encoder(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def encode(cls, model: Model, node: Node, NS) -> EncodedNode:
        pass

    @classmethod
    @abstractmethod
    def name(cls):
        pass


class ActivateMixin:
    @classmethod
    def activate(cls, node_name, input,  w_shape, weight_dict, b_shape, b_dict, activate_func):
        W = create_weight(node_name, w_shape, weight_dict)
        b = create_bias(node_name, b_shape, b_dict)
        Z = cls.calc_Z(input, W, b)
        output_tesor = cls._activate(activate_func, Z)
        return output_tesor

    @classmethod
    def calc_Z(cls, input, W, b):
        return tf.matmul(input, W) + b

    @classmethod
    def _activate(cls, activate_func, Z):
        if activate_func == "relu":
            return tf.nn.relu(Z)
        elif activate_func == "softmax":
            return tf.nn.softmax(Z)
        else:
            # TODO
            return tf.nn.relu(Z)


class InputEncoder(Encoder):
    @classmethod
    def encode(cls, model: Model, node: Node, NS):
        return EncodedNode(node.name, model.input_tensor)

    @classmethod
    def name(cls):
        return "input"

class ConvEncoder(Encoder, ActivateMixin):
    @classmethod
    def encode(cls, model, node, NS):
        kernel = node.attr["weight"]["kernel"]
        ch = NS[node.input[0]].output_tensor.get_shape()[3]
        w_shape = [kernel[0], kernel[1], ch, node.attr["num_out_channel"]]
        b_shape = [node.attr["num_out_channel"]]
        input_tensor = NS[node.input[0]].output_tensor
        output_tesor = cls.activate(node.name,input_tensor, w_shape,node.attr["weight"], b_shape, node.attr["bias"], node.attr["activate_func"])
        return EncodedNode(node.name, output_tesor)

    @classmethod
    def calc_Z(cls, input_tensor, W, b):
        return tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding='SAME') + b

    @classmethod
    def name(cls):
        return "conv2d"

def get_input_tensor(NS, node):
    return NS[node.input[0]].output_tensor


class FullConnOutputNode(Encoder, ActivateMixin):
    @classmethod
    def encode(cls, model, node, NS):
        neuron_counts = model.num_class
        input_tensor = get_input_tensor(NS, node)
        ch = input_tensor.get_shape()[1]
        w_shape = [ch, neuron_counts]
        b_shape = [neuron_counts]

        output_tesor = cls.activate(node.name, input_tensor, w_shape,
                                    node.attr["weight"], b_shape,
                                    node.attr["bias"],
                                    node.attr["activate_func"])
        return EncodedNode(node.name, output_tesor)

    @classmethod
    def name(cls):
        return "fc_output"


class FullConnHiddenNode(Encoder, ActivateMixin):
    @classmethod
    def encode(cls, model, node, NS):
        neuron_counts = node.attr["neuron_counts"]
        input_tensor = get_input_tensor(NS, node)
        ch = input_tensor.get_shape()[1]
        w_shape = [ch, neuron_counts]
        b_shape = [neuron_counts]

        output_tesor = cls.activate(node.name, input_tensor, w_shape,
                                    node.attr["weight"], b_shape,
                                    node.attr["bias"],
                                    node.attr["activate_func"])
        return EncodedNode(node.name, output_tesor)


    @classmethod
    def name(cls):
        return "fc_hidden"


class FullConnInputNode(Encoder, ActivateMixin):
    @classmethod
    def encode(cls, model, node, NS):
        neuron_counts = node.attr["neuron_counts"]
        input_tensor = get_input_tensor(NS, node)
        batch_size = model.batch_size
        input_tensor = tf.reshape(input_tensor, [batch_size, -1])
        dim = input_tensor.get_shape()[1].value
        w_shape = [dim, neuron_counts]
        b_shape = [neuron_counts]

        output_tesor = cls.activate(node.name, input_tensor, w_shape,
                                    node.attr["weight"], b_shape,
                                    node.attr["bias"],
                                    node.attr["activate_func"])
        return EncodedNode(node.name, output_tesor)

    @classmethod
    def name(cls):
        return "fc_input"


class NormEncoder(Encoder):
    @classmethod
    def encode(cls, model, node, NS):
        input_tensor = get_input_tensor(NS, node)
        output_tensor = tf.nn.lrn(input_tensor,
                      node.attr["depth_radius"],
                      node.attr["bias"],
                      node.attr["alpha"],
                      node.attr["beta"])
        return EncodedNode(node.name, output_tensor)

    @classmethod
    def name(cls):
        return "local_response_normalization"


class MaxPoolEncoder(Encoder):
    @classmethod
    def encode(cls, model, node, NS):
        ksize = node.attr["ksize"]
        strides = node.attr["strides"]
        padding = node.attr["padding"]
        input_tensor = get_input_tensor(NS, node)
        output_tensor = tf.nn.max_pool(input_tensor, ksize=ksize, strides=strides, padding=padding)
        return EncodedNode(node.name, output_tensor)

    @classmethod
    def name(cls):
        return "max_pool"


def calc_pool_output_size(padding: str, in_height, in_width, filter_height, filter_width, stride_height, stride_width):
    if padding == "SAME":
        out_height = math.ceil(float(in_height) / float(stride_height))
        out_width = math.ceil(float(in_width) / float(stride_width))
    else: #padding == "VALID"
        out_height = math.ceil(float(in_height - filter_height + 1) / float(stride_height))
        out_width = math.ceil(float(in_width - filter_width + 1) / float(stride_width))
    return out_height, out_width



def create_weight(name, shape, weight_dict):
    initializer = create_initializer(weight_dict["initializer"])
    name = "w_" + name
    dtype = get_type(weight_dict)
    decay = weight_dict["decay"]
    w = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    if isDecay(weight_dict):
        weight_decay = tf.mul(tf.nn.l2_loss(w), decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)  # TODO: いる？
    return w


def create_bias(name, shape, bias_dict):
    initializer = create_initializer(bias_dict["initializer"])
    name = "b_" + name
    dtype = get_type(bias_dict)
    b = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return b


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


def inference(images, keep_prob, model_blueprint):
    """予測モデルを作成する.
    :param images: input image tensor.
    :param keep_prob:
    :return: logits tensor.
    """
    tensor = parse_and_create_node(model_blueprint, images)
    return tensor


def regist_encoders():
    import cnn_inference
    cs = inspect.getmembers(cnn_inference, inspect.isclass)
    for c in cs:
        if issubclass(c[1], cnn_inference.Encoder):
            Encoders[c[1].name()] = c[1]


regist_encoders()


