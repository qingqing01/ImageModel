#!/usr/bin/env python
from paddle.trainer_config_helpers import *

height=224
width=height
num_class = 100

args={'image_size':256, 'crop_size':224, 'color':True, 'num_class':num_class,
      'mean_value':[103.939,116.779,123.68]}
define_py_data_sources2("data/train.list",
                        "data/test.list",
                        module="provider",
                        obj="processPIL",
                        args=args)

batch_size = 128
learning_rate = 0.01 / batch_size
momentum = 0.9
weight_decay = 0.0002 * batch_size
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

def xavier(channels, filter_size):
    init_w = (3.0 / (filter_size ** 2 * channels)) ** 0.5
    return ParamAttr(initial_min=0.0-init_w,
                     initial_max=init_w)

def inception2(name, input, channels, \
    filter1,
    filter3R, filter3,
    filter5R, filter5,
    proj):

    conv1 = name + '_1'
    conv3r = name + '_3r'
    conv3 = name + '_3'
    conv5r = name + '_5r'
    conv5 = name + '_5'
    maxpool = name + '_max'
    convproj = name + '_proj'

    bias_attr = ParamAttr(initial_std=0.,l2_rate=0., learning_rate=2.0,
                          initial_mean=0.2)
    cov1 = img_conv_layer(name=conv1, input=input, filter_size=1,
                          num_channels=channels, num_filters=filter1,
                          stride=1, padding=0, bias_attr=bias_attr,
                          param_attr=xavier(channels, 1))

    cov3r = img_conv_layer(name=conv3r, input=input, filter_size=1,
                           num_channels=channels, num_filters=filter3R,
                           stride=1, padding=0, bias_attr=bias_attr,
                           param_attr=xavier(channels, 1))
    cov3 = img_conv_layer(name=conv3, input=cov3r, filter_size=3,
                          num_filters=filter3, stride=1, padding=1,
                          bias_attr=bias_attr,
                          param_attr=xavier(conv3r.num_filters, 3))

    cov5r = img_conv_layer(name=conv5r, input=input, filter_size=1,
                           num_channels=channels, num_filters=filter5R,
                           stride=1, padding=0, bias_attr=bias_attr,
                           param_attr=xavier(channels, 1))
    cov5 = img_conv_layer(name=conv5, input=cov5r, filter_size=5,
                          num_filters=filter5, stride=1, padding=2,
                          bias_attr=bias_attr,
                          param_attr=xavier(cov5r.num_filters, 5))
    
    pool1 = img_pool_layer(name=maxpool, input=input, pool_size=3,
                           num_channels=channels, stride=1, padding=1)
    covprj = img_conv_layer(name=convproj, input=pool1, filter_size=1,
                            num_filters=proj, stride=1, padding=0,
                            param_attr=xavier(channels, 1))

    cat = concat_layer(name=name, input=[cov1, cov3, cov5, covprj])
    return cat

def inception(name, input, channels, \
    filter1,
    filter3R, filter3,
    filter5R, filter5,
    proj):

    bias_attr = ParamAttr(initial_std=0.,l2_rate=0., learning_rate=2.0,
                          initial_mean=0.2)
    cov1 = conv_projection(input=input, filter_size=1, num_channels=channels,
                           num_filters=filter1, stride=1, padding=0,
                           param_attr=xavier(channels, 1))

    cov3r = img_conv_layer(name=name + '_3r', input=input, filter_size=1,
                           num_channels=channels, num_filters=filter3R,
                           stride=1, padding=0, bias_attr=bias_attr,
                           param_attr=xavier(channels, 1))
    cov3 = conv_projection(input=cov3r, filter_size=3, num_filters=filter3,
                           stride=1, padding=1,
                           param_attr=xavier(cov3r.num_filters, 3))

    cov5r = img_conv_layer(name=name + '_5r', input=input, filter_size=1,
                           num_channels=channels, num_filters=filter5R,
                           stride=1, padding=0, bias_attr=bias_attr,
                           param_attr=xavier(channels, 1))
    cov5 = conv_projection(input=cov5r, filter_size=5, num_filters=filter5,
                           stride=1, padding=2,
                           param_attr=xavier(cov5r.num_filters, 5))
    
    pool1 = img_pool_layer(name=name + '_max', input=input, pool_size=3,
                           num_channels=channels, stride=1, padding=1)
    covprj = conv_projection(input=pool1, filter_size=1, num_filters=proj,
                             stride=1, padding=0,
                             param_attr=xavier(channels, 1))

    cat = concat_layer(name=name, input=[cov1, cov3, cov5, covprj],
                       act=ReluActivation(), bias_attr=bias_attr)
    return cat


lab = data_layer(name="label", size=num_class)
data = data_layer(name="input", size=3 * height * width)

# stage 1
bias_attr = ParamAttr(initial_std=0.,l2_rate=0., learning_rate=2.0, initial_mean=0.2)
conv1 = img_conv_layer(name="conv1", input=data, filter_size=7,
                       num_channels=3, num_filters=64, stride=2,
                       padding=3, bias_attr=bias_attr,
                       param_attr=xavier(3, 7))
pool1 = img_pool_layer(name="pool1", input=conv1, pool_size=3,
                       num_channels=64, stride=2)

# stage 2
conv2_1 = img_conv_layer(name="conv2_1", input=pool1, filter_size=1,
                         num_filters=64, stride=1, padding=0,
                         bias_attr=bias_attr,
                         param_attr=xavier(pool1.num_filters, 1))
conv2_2 = img_conv_layer(name="conv2_2", input=conv2_1, filter_size=3,
                         num_filters=192, stride=1, padding=1,
                         bias_attr=bias_attr,
                         param_attr=xavier(conv2_1.num_filters, 3))
pool2 = img_pool_layer(name="pool2", input=conv2_2, pool_size=3,
                       num_channels=192, stride=2)

# stage 3
ince3a = inception("ince3a", pool2,  192,  64, 96, 128, 16, 32, 32) 
ince3b = inception("ince3b", ince3a, 256, 128, 128,192, 32, 96, 64) 
pool3 = img_pool_layer(name="pool3", input=ince3b, num_channels=480, pool_size=3, stride=2)

# stage 4
ince4a = inception("ince4a", pool3,  480, 192, 96,  208, 16, 48, 64)  
ince4b = inception("ince4b", ince4a, 512, 160, 112, 224, 24, 64, 64) 
ince4c = inception("ince4c", ince4b, 512, 128, 128, 256, 24, 64, 64)
ince4d = inception("ince4d", ince4c, 512, 112, 144, 288, 32, 64, 64)  
ince4e = inception("ince4e", ince4d, 528, 256, 160, 320, 32, 128, 128) 
pool4 = img_pool_layer(name="pool4", input=ince4e, num_channels=832, pool_size=3, stride=2)

# stage 5
ince5a = inception("ince5a", pool4,  832, 256, 160, 320, 32, 128, 128)
ince5b = inception("ince5b", ince5a, 832, 384, 192, 384, 48, 128, 128)
pool5 = img_pool_layer(name="pool5", input=ince5b, num_channels=1024, pool_size=7, stride=7, pool_type=AvgPooling())

# output 1
pool_o1 = img_pool_layer(name="pool_o1", input=ince4a, num_channels=512, pool_size=5, stride=3, pool_type=AvgPooling())
conv_o1 = img_conv_layer(name="conv_o1", input=pool_o1, filter_size=1, num_filters=128, stride=1, padding=0,
                         bias_attr=bias_attr, param_attr=xavier(pool_o1.num_filters, 1))
fc_o1 = fc_layer(name="fc_o1", input=conv_o1, size=1024, layer_attr=ExtraAttr(drop_rate=0.7), act=ReluActivation(),
                 bias_attr=bias_attr, param_attr=xavier(2048, 1))
out1 = fc_layer(name="output1", input=fc_o1,  size=num_class, act=SoftmaxActivation(),
                bias_attr=bias_attr, param_attr=xavier(1024, 1))
loss1 = cross_entropy(name='loss1', input=out1, label=lab, coeff=0.3) 

# output 2
pool_o2 = img_pool_layer(name="pool_o2", input=ince4d, num_channels=528, pool_size=5, stride=3, pool_type=AvgPooling())
conv_o2 = img_conv_layer(name="conv_o2", input=pool_o2, filter_size=1, num_filters=128, stride=1, padding=0,
                         bias_attr=bias_attr, param_attr=xavier(pool_o2.num_filters, 1))
fc_o2 = fc_layer(name="fc_o2", input=conv_o2, size=1024, layer_attr=ExtraAttr(drop_rate=0.7), act=ReluActivation(),
                 bias_attr=bias_attr, param_attr=xavier(2048, 1))
out2 = fc_layer(name="output2", input=fc_o2, size=num_class, act=SoftmaxActivation(),
                bias_attr=bias_attr, param_attr=xavier(1024, 1))
loss2 = cross_entropy(name='loss2', input=out2, label=lab, coeff=0.3) 

# output 3
dropout = dropout_layer(name="dropout", input=pool5, dropout_rate=0.4)
out3 = fc_layer(name="output3", input=dropout, size=num_class, act=SoftmaxActivation(),
                bias_attr=bias_attr, param_attr=xavier(1024, 1))
loss3 = cross_entropy(name='loss3', input=out3, label=lab) 

outputs(loss3)
