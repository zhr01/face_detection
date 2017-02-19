import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from scipy.misc import imread
from train import build_forward
from evaluate import add_rectangles
import cv2
import scipy.io

gt_path = "/home/zhr/tensorflow/crowd_count/dataset/mall_dataset/mall_gt.mat"
data = scipy.io.loadmat(gt_path)["frame"][0]

hypes_file = './hypes/overfeat_rezoom.json'
iteration = 150000
with open(hypes_file, 'r') as f:
    H = json.load(f)
true_idl = './data/brainwash/brainwash_val.idl'
pred_idl = './output/%d_val_%s.idl' % (iteration, os.path.basename(hypes_file).replace('.json', ''))


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
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, './data/overfeat_rezoom/save.ckpt-%dv2' % iteration)

    dataset_path = "/home/zhr/tensorflow/crowd_count/dataset/mall_dataset/frames"
    files = os.listdir(dataset_path)
    files.sort()

    fgbg = cv2.createBackgroundSubtractorMOG2()
    for i in range(10):
        img = imread(os.path.join(dataset_path, files[i]))
        fgbg.apply(img)

    outfile = 'output'
    # bbox = []
    # bconf = []
    for i, file in enumerate(files):
        gt_points = data[i][0][0][0]
        img = imread(os.path.join(dataset_path, file))
        fgbg.apply(img)
        bg = fgbg.getBackgroundImage()
        img = cv2.resize(img, (640, 480))
        feed = {x_in: img}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)

        new_img, picks = add_rectangles(H, [img, bg], np_pred_confidences, np_pred_boxes,
                                        use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.3,
                                        show_suppressed=False)
        cv2.line(new_img,(0, 120), (640,120), (0,255,0),2)
        for pp in gt_points:
            cv2.circle(new_img, (int(pp[0]),int(pp[1])), 2,(0,255,0),1)
        # bbox.append(np_pred_boxes)
        # bconf.append(np_pred_confidences)
        # cv2.imshow('detection', cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
        # ch = 0xFF & cv2.waitKey(10)
        # if ch == 27:
        #     break
        cv2.imwrite(os.path.join(outfile, file), cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    # import pickle
    # pk_file = open("raw_results.pkl", 'wb')
    # pickle.dump(bbox, pk_file, -1)
    # pickle.dump(bconf, pk_file, -1)
    # pk_file.close()



    # video_path = '/home/zhr/tensorflow/opencv/notebook/768x576.avi'
    # cap = cv2.VideoCapture(video_path)
    # if cap.isOpened():
    #     fgbg = cv2.createBackgroundSubtractorMOG2()
    #     for i in range(20):
    #         ret, frame = cap.read()
    #         fgmask = fgbg.apply(frame)
    #     bg = fgbg.getBackgroundImage()
    #     bg = cv2.resize(bg, (640, 480))
    #
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret: break
    #         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         img = cv2.resize(img, (640, 480))
    #         feed = {x_in: img}
    #         (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
    #
    #         new_img, rects = add_rectangles(H, [img, bg], np_pred_confidences, np_pred_boxes,
    #                                         use_stitching=False, rnn_len=H['rnn_len'], min_conf=0.3)
    #         cv2.imshow('detection', cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    #         ch = 0xFF & cv2.waitKey(10)
    #         if ch == 27:
    #             break
    #     cap.release()

    # cv2.destroyAllWindows()
