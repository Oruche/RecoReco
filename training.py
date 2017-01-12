import tensorflow as tf
import numpy as np

import cnn_inference


def loss(logits, labels):
    """
    lossを計算する関数
    :param tensor(float) logits: logitのtensor, [batch_size, NUM_CLASSES]
    :param tensor(int32) labels:  ラベルのtensor, [batch_size, NUM_CLASSES]
    :rtype: tensor(float)
    :return: crossentropy
    """

    # 交差エントロピーの計算
    cross_entropy = - \
        tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
    # TensorBoardで表示するよう指定
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy


def _add_loss_summaries(total_loss):
    """
    Add summaries for losses

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    :param total_loss: Total loss from loss().
    :return: loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.999, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the
        # loss as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def training(loss, learning_rate):
    """
    訓練のopを定義する関数
    :param 損失のtensor loss: loss()の結果
    :param learning_rate: 学習係数
    :return: 訓練のop
    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


def accuracy(logits, labels):
    """
    accuracyを計算する関数
    :param logits: inference()の結果
    :param labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    :rtype: float
    :return: accuracy
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", accuracy)
    return accuracy


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    capacity = min_queue_examples + 3 * batch_size
    num_preprocess_threads = 4
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity)
    return images, label_batch


def execute_train(images, labels, params, loadname="", savename=""):

    image_size = params.image_size
    max_steps = params.max_steps
    batch_size = params.batch_size
    learning_rate = params.learning_rate
    image_pixels = image_size * image_size * 3
    num_class = len(params.labels)

    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2

    # # numpy形式に変換
    train_image = np.asarray(images)
    train_label = np.asarray(labels)

    with tf.Graph().as_default():

        train_image_tensor = tf.convert_to_tensor(
            train_image, dtype=tf.float32)
        train_label_tensor = tf.convert_to_tensor(
            train_label, dtype=tf.float32)

        train_label_tensor = tf.cast(train_label_tensor, tf.int64)
        train_one_hot_label = tf.one_hot(
            train_label_tensor, depth=num_class, on_value=1.0, off_value=0.0, axis=-1)

        img_queue = tf.train.input_producer(train_image_tensor)
        label_queue = tf.train.input_producer(train_one_hot_label)

        img = img_queue.dequeue()
        label = label_queue.dequeue()

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(
            NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        images_tensor, labels_tensor = _generate_image_and_label_batch(
            img, label, min_queue_examples, batch_size=1, shuffle=False)

        # dropout率
        keep_prob = tf.placeholder("float")

        # 入力をIMAGE_SIZExIMAGE_SIZEx3に変形
        x_image = tf.reshape(images_tensor, [-1, image_size, image_size, 3])

        # inference()を呼び出してモデルを作る
        #logits = inference(x_image, keep_prob)
        logits = cnn_inference.inference(x_image, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_tensor)
        # training()を呼び出して訓練
        train_op = training(loss_value, learning_rate)
        # 精度の計算
        acc = accuracy(logits, labels_tensor)

        saver = tf.train.Saver()

        # Sessionの作成
        sess = tf.Session()

        if loadname == "":
            # 変数の初期化
            sess.run(tf.initialize_all_variables())
        else:
            saver.restore(sess, "./model_tmp/downloads/model.ckpt")

        tf.train.start_queue_runners(sess)

        # 訓練の実行
        for step in range(max_steps):
            for i in range(int(len(train_image) / batch_size)):
                # batch_size分の画像に対して訓練の実行
                batch = batch_size * i

                image_ = sess.run(x_image, feed_dict={keep_prob: 0.5})

                sess.run(train_op, feed_dict={keep_prob: 0.5})

            # 1step終わるたびに精度を計算する
            train_accuracy = sess.run(acc, feed_dict={keep_prob: 1.0})
            print("step {}, training accuracy {}".format(step, train_accuracy))

        # 最終的なモデルを保存
        save_path = saver.save(sess, savename)
