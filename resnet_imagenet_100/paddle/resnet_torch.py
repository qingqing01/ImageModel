# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.trainer_config_helpers import *

"""
paper: https://arxiv.org/abs/1512.03385
"""

args={
    'mean_value': [103.939,116.779,123.68],
    'image_size': 256, 'crop_size': 224, 'color': True}
define_py_data_sources2('data/train.list',
                        'data/test.list',
                        module="provider",
                        obj="processPIL",
                        args=args)

layer_num = 50
num_class = 100
batch_size = 128
learning_rate = 0.1 / batch_size
momentum = 0.9
weight_decay = 0.0001 * batch_size
default_momentum(momentum)
default_decay_rate(weight_decay)

Settings(
    algorithm='sgd',
    batch_size=batch_size,
    learning_rate=learning_rate,

    learning_method='momentum',
    learning_rate_decay_a=0.1,
    learning_rate_decay_b=128660 * 32,
    learning_rate_schedule="discexp",
)


def conv_bn_layer(name, input, filter_size, num_filters,
                  stride, padding, channels=None,
                  active_type=ReluActivation()):
    """
    A wrapper for conv layer with batch normalization layers.
    Note:
    conv layer has no activation.
    """

    tmp = img_conv_layer(name=name + "_conv",
                         input=input,
                         filter_size=filter_size,
                         num_channels=channels,
                         num_filters=num_filters,
                         stride=stride,
                         padding=padding,
                         act=LinearActivation(),
                         bias_attr=False)
    return batch_norm_layer(name=name + "_bn",
                            input=tmp,
                            act=active_type)


def bottleneck_block(name, input, num_filters1, num_filters2):
    """
    A wrapper for bottlenect building block in ResNet.
    Last conv_bn_layer has no activation.
    Addto layer has activation of relu.
    """
    last_name = conv_bn_layer(name=name + '_branch2a',
                              input=input,
                              filter_size=1,
                              num_filters=num_filters1,
                              stride=1,
                              padding=0)
    last_name = conv_bn_layer(name=name + '_branch2b',
                              input=last_name,
                              filter_size=3,
                              num_filters=num_filters1,
                              stride=1,
                              padding=1)
    last_name = conv_bn_layer(name=name + '_branch2c',
                              input=last_name,
                              filter_size=1,
                              num_filters=num_filters2,
                              stride=1,
                              padding=0,
                              active_type=LinearActivation())

    return addto_layer(name=name + "_addto",
                       input=[input, last_name],
                       act=ReluActivation())


def mid_projection(name, input, num_filters1, num_filters2, stride=2):
    """
    A wrapper for middile projection in ResNet.
    projection shortcuts are used for increasing dimensions,
    and other shortcuts are identity
    branch1: projection shortcuts are used for increasing
    dimensions, has no activation.
    branch2x: bottleneck building block, shortcuts are identity.
    """
    # stride = 2
    branch1 = conv_bn_layer(name=name + '_branch1',
                            input=input,
                            filter_size=1,
                            num_filters=num_filters2,
                            stride=stride,
                            padding=0,
                            active_type=LinearActivation())

    last_name = conv_bn_layer(name=name + '_branch2a',
                              input=input,
                              filter_size=1,
                              num_filters=num_filters1,
                              stride=stride,
                              padding=0)
    last_name = conv_bn_layer(name=name + '_branch2b',
                              input=last_name,
                              filter_size=3,
                              num_filters=num_filters1,
                              stride=1,
                              padding=1)

    last_name = conv_bn_layer(name=name + '_branch2c',
                              input=last_name,
                              filter_size=1,
                              num_filters=num_filters2,
                              stride=1,
                              padding=0,
                              active_type=LinearActivation())

    return addto_layer(name=name + "_addto",
                       input=[branch1, last_name],
                       act=ReluActivation())


def mid_projection_k3(name, input, num_filters1, num_filters2, stride=2):
    # stride = 2
    branch1 = conv_bn_layer(name=name + '_branch1',
                            input=input,
                            filter_size=3,
                            num_filters=num_filters2,
                            stride=stride,
                            padding=1,
                            active_type=LinearActivation())

    last_name = conv_bn_layer(name=name + '_branch2a',
                              input=input,
                              filter_size=3,
                              num_filters=num_filters1,
                              stride=stride,
                              padding=1)
    last_name = conv_bn_layer(name=name + '_branch2b',
                              input=last_name,
                              filter_size=3,
                              num_filters=num_filters1,
                              stride=1,
                              padding=1)

    last_name = conv_bn_layer(name=name + '_branch2c',
                              input=last_name,
                              filter_size=1,
                              num_filters=num_filters2,
                              stride=1,
                              padding=0,
                              active_type=LinearActivation())

    return addto_layer(name=name + "_addto",
                       input=[branch1, last_name],
                       act=ReluActivation())



def deep_res_net(res2_num=3, res3_num=4, res4_num=6, res5_num=3):
    """
    A wrapper for 50,101,152 layers of ResNet.
    res2_num: number of blocks stacked in conv2_x
    res3_num: number of blocks stacked in conv3_x
    res4_num: number of blocks stacked in conv4_x
    res5_num: number of blocks stacked in conv5_x
    """
    # For ImageNet
    # conv1: 112x112
    img = data_layer(name='input', size=224 * 224 * 3)
    tmp = conv_bn_layer("conv1", img,
                        filter_size=7,
                        channels=3,
                        num_filters=64,
                        stride=2,
                        padding=3)
    tmp = img_pool_layer(name="pool1", input=tmp, pool_size=3, stride=2)

    # conv2_x: 56x56
    tmp = mid_projection(name="res2_1",
                         input=tmp,
                         num_filters1=64,
                         num_filters2=256,
                         stride=1)
    for i in xrange(2, res2_num + 1, 1):
        tmp = bottleneck_block(name="res2_" + str(i),
                               input=tmp,
                               num_filters1=64,
                               num_filters2=256)

    # conv3_x: 28x28
    tmp = mid_projection_k3(name="res3_1",
                            input=tmp,
                            num_filters1=128,
                            num_filters2=512)
    for i in xrange(2, res3_num + 1, 1):
        tmp = bottleneck_block(name="res3_" + str(i),
                               input=tmp, num_filters1=128,
                               num_filters2=512)

    # conv4_x: 14x14
    tmp = mid_projection_k3(name="res4_1", input=tmp,
                            num_filters1=256, num_filters2=1024)
    for i in xrange(2, res4_num + 1, 1):
        tmp = bottleneck_block(name="res4_" + str(i),
                               input=tmp,
                               num_filters1=256,
                               num_filters2=1024)

    # conv5_x: 7x7
    tmp = mid_projection_k3(name="res5_1", input=tmp,
                            num_filters1=512, num_filters2=2048)
    for i in xrange(2, res5_num + 1, 1):
        tmp = bottleneck_block(name="res5_" + str(i),
                               input=tmp, num_filters1=512,
                               num_filters2=2048)

    tmp = img_pool_layer(name='avgpool',
                         input=tmp,
                         pool_size=7,
                         stride=1,
                         pool_type=AvgPooling())

    output = fc_layer(name='output',
                      input=tmp,
                      size=num_class,
                      param_attr=ParamAttr(initial_std=0.01),
                      bias_attr=ParamAttr(initial_std=0., l2_rate=0.),
                      act=SoftmaxActivation())

    label=data_layer(name='label',size=1)
    tmp=classification_cost(name='cost',input=output, label=label)


def res_net_50():
    deep_res_net(3, 4, 6, 3)


def res_net_101():
    deep_res_net(3, 4, 23, 3)


def res_net_152():
    deep_res_net(3, 8, 36, 3)


Inputs("label", "input")
Outputs("cost")

if layer_num == 50:
    res_net_50()
elif layer_num == 101:
    res_net_101()
elif layer_num == 152:
    res_net_152()
else:
    print("Wrong layer number.")
