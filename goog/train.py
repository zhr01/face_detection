import tensorflow as tf
import numpy as np
from numpy import random as rnd
import os, shutil
import train_utils
import time
from string import join as sj
from os import listdir as ls
import json

H = {
    'base_net': 'googlenet',
    'weight_init_size': 1e-3,
    'use_loss_matrix': False,
    'optimizer': 'Adam',
    #'learning_rate': 1e-3,
    'learning_rate': 1e-4,
    'epsilon': 1.0,
    'weight_decay': 1e-4,
    'use_dropout': True,
    'batch_size': 20,
    'min_skin_prob': 0.4,
    'min_tax_score': 0.8
}

gpu_id = 0
save_dir = 'default'
halfdata = False
num_threads = 6

if H['base_net'] == 'googlenet':
    head_weights = [0.2, 0.2, 0.6]
    features_dim = 1024
    image_width = 640
    image_height = 480
    input_mean = 117.
    input_std  = 1.
    input_layer = 'input'
    features_layers = ['nn0/reshape', 'nn1/reshape', 'avgpool0/reshape']

opt = tf.train.AdamOptimizer(learning_rate=H['learning_rate'], epsilon=H['epsilon'])

if not os.path.exists(save_dir): os.makedirs(save_dir)

ckpt_file = save_dir + '/save.ckpt'
deploy_graph_file = save_dir + '/graph_{}.pb'
graph_def_orig_file = 'graphs/{}.pb'.format(H['base_net'])

with open(save_dir + '/hypes.json', 'w') as f:
    json.dump(H, f, indent=4)

ckpt_iters = sorted([
    int(i.split('-')[-1]) 
    for i in ls(save_dir) 
    if '.ckpt-' in i
])
if ckpt_iters:
    restore_file = ckpt_file+'-'+str(ckpt_iters[-1])
else:
    restore_file = None

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

data = train_utils.load_data('')

k = len(data['train']['Y'][0][0])

graphs = {name: tf.Graph() for name in ['orig']}

graph_def = tf.GraphDef()
with open(graph_def_orig_file) as f:
    graph_def.MergeFromString(f.read())

with graphs['orig'].as_default():
    tf.import_graph_def(graph_def, name='')

input_op = graphs['orig'].get_operation_by_name(input_layer)

features_ops = [
    graphs['orig'].get_operation_by_name(i)
    for i in features_layers
]

weights_ops = [
    op for op in graphs['orig'].get_operations() 
    if ('params' in op.name or 'batchnorm/beta' in op.name)
    and op.type == 'Const'
]

reuse_ops = [
    op for op in graphs['orig'].get_operations() 
    if op not in weights_ops + [input_op]
    and op.name != 'output'
]

with tf.Session(graph=graphs['orig'], config=config):
    weights_orig = {
        op.name: op.outputs[0].eval()
        for op in weights_ops
    }

weight_init = lambda: H['weight_init_size']*rnd.randn(features_dim, k).astype(np.float32)
bias_init = lambda: H['weight_init_size']*rnd.randn(k).astype(np.float32)

W = [
    tf.Variable(weight_init(), name='softmax/weights_{}'.format(i)) 
    for i in range(len(features_layers))
]

B = [
    tf.Variable(bias_init(), name='softmax/biases_{}'.format(i)) 
    for i in range(len(features_layers))
]

weight_vars = {
    name: tf.Variable(weight, name=name)
    for name, weight in weights_orig.iteritems()
}

weight_tensors = {
    name: tf.convert_to_tensor(weight)
    for name, weight in weight_vars.iteritems()
}

W_norm = [tf.nn.l2_loss(weight) for weight in weight_vars.values() + W]
W_norm = tf.reduce_sum(tf.pack(W_norm), name='weights_norm')
tf.scalar_summary(W_norm.op.name, W_norm)

def model(x, weights_dict):
    def is_early_loss(name):
        early_loss_layers = ['head0', 'nn0', 'softmax0', 'head1', 'nn1', 'softmax1', 'output1']
        return any(name.startswith(prefix) for prefix in early_loss_layers)

    T = weights_dict
    T[input_op.name] = x

    for op in reuse_ops:

        if op.type == 'BatchNormWithGlobalNormalization':

            t, mu, sig, beta, gamma = [T[t.op.name] for t in list(op.inputs)]
            mu, sig = tf.nn.moments(t, [0,1,2])
            T[op.name] = tf.nn.batch_norm_with_global_normalization(
                t=t, m=mu, v=sig, beta=beta, gamma=gamma, 
                variance_epsilon=1e-8, scale_after_normalization=True, 
                name=op.name
            )

        else:

            if is_early_loss(op.name):
                continue
            elif op.name == 'avgpool0':
                pool_op = tf.nn.avg_pool(T['mixed5b'], ksize=[1,15,20,1], strides=[1,1,1,1], padding='VALID', name=op.name)
                T[op.name] = pool_op

            else:
                copied_op = x.graph.create_op(
                    op_type = op.type, 
                    inputs = [T[t.op.name] for t in list(op.inputs)], 
                    dtypes = [o.dtype for o in op.outputs], 
                    name = op.name, 
                    attrs =  op.node_def.attr
                )

                T[op.name] = copied_op.outputs[0]
    

    cnn_feat = T['mixed5b']
    cnn_feat_r = tf.reshape(cnn_feat, [H['batch_size'] * 15 * 20, 1024])

    Z = [cnn_feat_r]

    S = tf.shape(weights_dict['mixed5b'])
    if H['base_net'] == 'inception': Z = [tf.squeeze(z) for z in Z]

    return Z, S

files, scores, loss, accuracy = {},{},{},{}

for phase in ['train', 'test']:

    X = data[phase]['X']
    Y = data[phase]['Y']
    print(Y.shape)

    files[phase], y = tf.train.slice_input_producer(
        tensor_list=[X, Y],
        shuffle=True,
        capacity=H['batch_size']*(num_threads+1)
    )

    x = tf.read_file(files[phase])
    x = tf.image.decode_png(x, channels=3)
    if phase == 'train':
        #x = tf.image.resize_images(x, image_height/5*6, image_width/5*6)
        x = tf.image.resize_images(x, image_height, image_width)
        tf.image_summary('orig_image', tf.expand_dims(x, 0))
        tf.scalar_summary('x.shape[0]', tf.shape(x)[0])
        tf.scalar_summary('x.shape[1]', tf.shape(x)[1])
        tf.scalar_summary('y.shape[0]', tf.shape(y)[0])
        tf.scalar_summary('y.shape[1]', tf.shape(y)[1])
        #x = tf.image.random_crop(x, [image_height, image_width])
        #x = tf.image.random_flip_left_right(x)
        #x = tf.image.random_flip_up_down(x)
        tf.image_summary('train_image', tf.expand_dims(x, 0))
    else:
        x = tf.image.resize_images(x, image_height, image_width)

    x -= input_mean
    x /= input_std

    x, y, files[phase] = tf.train.shuffle_batch(
        tensor_list = [x, y, files[phase]],
        batch_size = H['batch_size'],
        num_threads = num_threads,
        capacity = H['batch_size']*(num_threads+1),
        min_after_dequeue = H['batch_size']
    )

    Z, S = model(x, weight_tensors)

    if H['use_dropout'] and phase == 'train':
        Z = [tf.nn.dropout(z, 0.5) for z in Z]

    Z2 = tf.reshape(tf.nn.xw_plus_b(Z[0],W[0],B[0], name=phase+'/logits_0'), 
          [H['batch_size'], 300, k])
    Z2 = [Z2]
    Z3 = [tf.reshape(Z2[0], [H['batch_size'] * 300, k])]

    if H['use_loss_matrix']:

        loss_weights = tf.nn.embedding_lookup(H['loss_matrix'], y)
        L = [tf.reduce_sum(-tf.log(tf.nn.softmax(z)+1e-8)*loss_weights, 1)
             for z in Z2]
    else:

        y_sparse = y
        y = tf.reshape(y_sparse, [H['batch_size'] * 300, k])

        L = [tf.nn.softmax_cross_entropy_with_logits(Z3[0], y)]


    L = tf.mul(tf.pack(L), np.array(head_weights).reshape([len(features_layers), 1]))
    L = tf.reduce_sum(L, reduction_indices=0)
    tf.histogram_summary(phase+'/loss_histogram', L)
    L = tf.reduce_mean(L, name=phase+'/loss')
    tf.scalar_summary(L.op.name, L)
    loss[phase] = L

    z = Z3[-1]
    scores[phase] = z

    a = tf.equal(tf.argmax(y, 1), tf.argmax(z, 1))
    a = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')
    tf.scalar_summary(a.op.name, a)
    accuracy[phase] = a

    if phase == 'test':
        moving_avg = tf.train.ExponentialMovingAverage(0.95)
        smooth_op = moving_avg.apply([a])
        tf.scalar_summary(a.op.name + '/smooth', moving_avg.average(a))

    if phase == 'train':
        grads = opt.compute_gradients(L + H['weight_decay']*W_norm)
        global_step = tf.Variable(0, trainable=False)
        train_op = opt.apply_gradients(grads, global_step=global_step)

summary_op = tf.merge_all_summaries()

coord = tf.train.Coordinator()
saver = tf.train.Saver(max_to_keep=None)
writer = tf.train.SummaryWriter(
    logdir=save_dir, 
    flush_secs=1
)
writer.add_graph(tf.get_default_graph().as_graph_def())

print_str = sj([
    'Step: %d',
    'Train Loss: %.2f',
    'Test Accuracy: %.1f%%',
    'W_norm/Loss: %.2f',
    'Time/image (ms): %.1f'
], ', ')

batch_files, batch_scores, batch_loss, batch_accuracy = {},{},{},{}
scores_lookup = {'train': {}, 'test': {}}
labels_lookup = {
    phase: {i: j for i, j in zip(data[phase]['X'], data[phase]['Y'])}
    for phase in ['train', 'test']
}

with tf.Session(config=config) as sess:

    sess.run(tf.initialize_all_variables())
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if restore_file: saver.restore(sess, restore_file)
    #if restore_file: print(restore_file)

    while not coord.should_stop():

        t = time.time()

        batch_loss['train'], batch_accuracy['test'], weights_norm, \
            summary_str, _, _, \
            batch_scores['train'], batch_scores['test'], \
            batch_files['train'], batch_files['test'], \
            shape = \
            sess.run([loss['train'], accuracy['test'], W_norm, 
                summary_op, train_op, smooth_op,
                scores['train'], scores['test'], 
                files['train'], files['test'], S])
        print(shape)

        for phase in ['train', 'test']:
            for filename, feats in zip(batch_files[phase], batch_scores[phase]):
                scores_lookup[phase][filename] = feats

        dt = (time.time()-t)/H['batch_size']

        writer.add_summary(summary_str, global_step=global_step.eval())
        print print_str % (
            global_step.eval(),
            batch_loss['train'],
            batch_accuracy['test']*100,
            H['weight_decay']*weights_norm, dt*1000
        )

        if global_step.eval() % 1000 == 0: 
            saver.save(sess, ckpt_file, global_step=global_step)

coord.join(threads)