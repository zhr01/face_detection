import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from scipy.misc import imread
from train import build_forward
from evaluate import add_rectangles
import cv2


hypes_file = './hypes/overfeat_rezoom.json'
iteration = 150000
with open(hypes_file, 'r') as f:
    H = json.load(f)
true_idl = './data/brainwash/brainwash_val.idl'
pred_idl = './output/%d_val_%s.idl' % (iteration, os.path.basename(hypes_file).replace('.json', ''))


if __name__ == "__main__":
    import sys
    pic_path = sys.argv[1]

    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './data/overfeat_rezoom/save.ckpt-%dv2' % iteration)

        # dataset_path = "/home/zhr/tensorflow/crowd_count/dataset/mall_dataset/frames"
        # files = os.listdir(dataset_path)
        # files.sort()
        files = [pic_path]

        outfile = 'output'

        for i, file in enumerate(files):
            img = imread(file)
            img = cv2.resize(img, (640, 480))
            feed = {x_in: img}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)

            new_img, raw_image, stitch_image = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.3,
                                            show_suppressed=False)

            cv2.imshow("result", cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
            cv2.imshow("raw", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("stitch", cv2.cvtColor(stitch_image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(0) & 0xff == 27:
                continue
    cv2.destroyAllWindows()
            # cv2.imwrite(os.path.join(outfile, file), cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
