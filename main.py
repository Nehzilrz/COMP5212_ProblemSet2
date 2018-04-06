import argparse
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 64
NUM_LABELS = 47
rnd = np.random.RandomState(123)
tf.set_random_seed(123)

def convert_image_data_to_float(image_raw):
    img_float = tf.expand_dims(tf.cast(image_raw, tf.float32) / 255, axis=-1)
    return img_float

def visualize_ae(i, x, features, reconstructed_image):
    '''
    This might be helpful for visualizing your autoencoder outputs
    :param i: index
    :param x: original data
    :param features: feature maps
    :param reconstructed_image: autoencoder output
    :return:
    '''
    plt.figure(0)
    plt.imshow(x[i, :, :], cmap="gray")
    plt.figure(1)
    plt.imshow(reconstructed_image[i, :, :, 0], cmap="gray")
    plt.figure(2)
    plt.imshow(np.reshape(features[i, :, :, :], (7, -1), order="F"), cmap="gray",)

def gradient_decent(loss, params, learning_rate=0.001, momentum=0.9):
    train_op_1 = []
    train_op_2 = []

    for i in range(len(params)):
        delta = tf.Variable(tf.zeros(params[i].shape), dtype=tf.float32)
        gradient = tf.gradients(loss, params[i])[0]
        cu_delta = momentum * delta - learning_rate * gradient
        train_op_1.append(tf.assign_add(params[i], cu_delta))
        train_op_2.append(tf.assign(delta, cu_delta))

    train_op = train_op_1 + train_op_2
    return train_op

def tf_get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    return tf.get_variable(name, shape = shape, initializer = initializer)

def build_cnn_model(placeholder_x, placeholder_y):
    with tf.variable_scope("cnn") as scope:
        img_float = convert_image_data_to_float(placeholder_x)

        conv1_w = tf_get_variable("conv1_weight", shape=(3, 3, 1, 32))
        conv1_b = tf_get_variable("conv1_bias", shape=(32))
        conv1_h = tf.nn.relu(tf.nn.conv2d(img_float, conv1_w, strides=(1, 1, 1, 1), padding='SAME') + conv1_b)

        conv2_w = tf_get_variable("conv2_weight", shape=(5, 5, 32, 32))
        conv2_b = tf_get_variable("conv2_bias", shape=(32))
        conv2_h = tf.nn.relu(tf.nn.conv2d(conv1_h, conv2_w, strides=(1, 2, 2, 1), padding='SAME') + conv2_b)

        conv3_w = tf_get_variable("conv3_weight", shape=(5, 5, 32, 64))
        conv3_b = tf_get_variable("conv3_bias", shape=(64))
        conv3_h = tf.nn.relu(tf.nn.conv2d(conv2_h, conv3_w, strides=(1, 1, 1, 1), padding='SAME') + conv3_b)

        conv4_w = tf_get_variable("conv4_weight", shape=(5, 5, 64, 64))
        conv4_b = tf_get_variable("conv4_bias", shape=(64))
        conv4_h = tf.nn.relu(tf.nn.conv2d(conv3_h, conv4_w, strides=(1, 2, 2, 1), padding='SAME') + conv4_b)

        # This is a simple fully connected network
        img_flattened = tf.reshape(conv4_h, [-1, np.prod(conv4_h.shape[1:])])
        weight = tf_get_variable("fc_weight", shape=(img_flattened.shape[1], NUM_LABELS),
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        logits = tf.matmul(img_flattened, weight)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=placeholder_y, logits=logits)
        # gradient decent algorithm
        params = [conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, conv4_w, conv4_b, weight]
        train_op = gradient_decent(loss, params, learning_rate=0.0010, momentum=0.9)

        y = tf.one_hot(placeholder_y, NUM_LABELS)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return params, train_op, accuracy, loss

# Major interfaces
def train_cnn(x, y, placeholder_x, placeholder_y):
    # Below is just a simple example, replace them with your own code
    num_iterations = 20

    # This is a simple model, write your own
    params, train_op, accuracy, loss = build_cnn_model(placeholder_x, placeholder_y)
    saver = tf.train.Saver(var_list=params)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        n = int(x.shape[0] / BATCH_SIZE)
        permutation = np.random.permutation(y.shape[0])
        x = x[permutation]
        y = y[permutation]

        train_n = int(n * 0.75)
        train_set = range(0, train_n)
        valid_set = range(train_n, n)

        for n_pass in range(num_iterations):
            accs = []
            losses = []
            for i in train_set:
                batch_x = x[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                batch_y = y[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                feed_dict = { placeholder_x: batch_x, placeholder_y: batch_y }

                _, acc, loss_ = sess.run([train_op, accuracy, loss], feed_dict = feed_dict)
                accs.append(acc)
                losses.append(loss_)

            train_acc = np.mean(accs)
            train_loss = np.mean(losses)
            accs = []
            losses = []
            for i in valid_set:
                batch_x = x[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                batch_y = y[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                feed_dict = { placeholder_x: batch_x, placeholder_y: batch_y }

                acc, loss_ = sess.run([accuracy, loss], feed_dict = feed_dict)
                accs.append(acc)
                losses.append(loss_)

            valid_acc = np.mean(accs)
            valid_loss = np.mean(losses)

            print("Epoch #{} , Acc_train={}, Loss_train={}, Acc_valid={}, Loss_valid={}".format(
                n_pass, train_acc, train_loss, valid_acc, valid_loss
            ))

        saver.save(sess=sess, save_path="./cnn_model")


def test_cnn(x, y, placeholder_x, placeholder_y):
    params, train_op, accuracy, loss = build_cnn_model(placeholder_x, placeholder_y)

    permutation = np.random.permutation(x.shape[0])
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    n = int(x.shape[0] / BATCH_SIZE)

    with tf.Session() as sess:    
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(var_list=params)
        saver.restore(sess=sess, save_path="./cnn_model")

        losses = []
        accs = []
        for i in range(n):
            batch_x = shuffled_x[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch_y = shuffled_y[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            feed_dict = { placeholder_x: batch_x, placeholder_y: batch_y }

            _, acc, loss_ = sess.run([train_op, accuracy, loss], feed_dict = feed_dict)
            accs.append(acc)
            losses.append(loss_)

        print("Result , Acc_test={}, Loss_test={}".format(np.mean(accs), np.mean(losses)))

    return np.mean(accs)

def build_ae_model(placeholder_x):
    with tf.variable_scope("ae") as scope:
        img_float = convert_image_data_to_float(placeholder_x)
        batch_size = tf.shape(img_float)[0]

        conv1_w = tf_get_variable("conv1_weight", shape=(5, 5, 1, 32))
        conv1_b = tf_get_variable("conv1_bias", shape=(32))
        conv1_h = tf.nn.relu(tf.nn.conv2d(img_float, conv1_w, strides=(1, 2, 2, 1), padding='SAME') + conv1_b)

        conv2_w = tf_get_variable("conv2_weight", shape=(5, 5, 32, 64))
        conv2_b = tf_get_variable("conv2_bias", shape=(64))
        conv2_h = tf.nn.relu(tf.nn.conv2d(conv1_h, conv2_w, strides=(1, 2, 2, 1), padding='SAME') + conv2_b)

        conv3_w = tf_get_variable("conv3_weight", shape=(3, 3, 64, 2))
        conv3_b = tf_get_variable("conv3_bias", shape=(2))
        conv3_h = tf.nn.conv2d(conv2_h, conv3_w, strides=(1, 1, 1, 1), padding='SAME') + conv3_b

        decon3_w = tf_get_variable("decon3_weight", shape=(3, 3, 64, 2))
        decon3_b = tf_get_variable("decon3_bias", shape=(64))
        decon3_h = tf.nn.relu(tf.nn.conv2d_transpose(
            conv3_h, decon3_w, (batch_size, 7, 7, 64), strides=(1, 1, 1, 1), padding='SAME') + decon3_b)

        decon2_w = tf_get_variable("decon2_weight", shape=(5, 5, 32, 64))
        decon2_b = tf_get_variable("decon2_bias", shape=(32))
        decon2_h = tf.nn.relu(tf.nn.conv2d_transpose(
            decon3_h, decon2_w, (batch_size, 14, 14, 32), strides=(1, 2, 2, 1), padding='SAME') + decon2_b)

        decon1_w = tf_get_variable("decon1_weight", shape=(5, 5, 1, 32))
        decon1_b = tf_get_variable("decon1_bias", shape=(1))
        decon1_h = tf.nn.conv2d_transpose(
            decon2_h, decon1_w, (batch_size, 28, 28, 1), strides=(1, 2, 2, 1), padding='SAME') + decon1_b

        loss = tf.reduce_mean(tf.square(decon1_h - img_float))
        features = conv3_h
        # gradient decent algorithm
        params = [conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b,
                  decon3_w, decon3_b, decon2_w, decon2_b, decon1_w, decon1_b]
        train_op = gradient_decent(loss, params, learning_rate=0.0008, momentum=0.9)
        
        img_recon = decon1_h

    return params, train_op, loss, img_recon, features

def train_ae(x, placeholder_x):
    num_iterations = 20
    params, train_op, loss, img_recon, features = build_ae_model(placeholder_x)

    saver = tf.train.Saver(var_list=params)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        n = int(x.shape[0] / BATCH_SIZE)

        for n_pass in range(num_iterations):
            permutation = np.random.permutation(x.shape[0])
            x = x[permutation]
            losses = []

            for i in range(n - 1):
                batch_x = x[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                feed_dict = { placeholder_x: batch_x }
                _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
                losses.append(loss_)

            print("Epoch #{}, Loss_train={}".format(n_pass, np.mean(losses)))

        saver.save(sess=sess, save_path="./ae_model")


def evaluate_ae(x, placeholder_x):
    params, train_op, loss, x_recon, x_features = build_ae_model(placeholder_x)

    n = int(x.shape[0] / BATCH_SIZE)
    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=params)
        saver.restore(sess, "./ae_model")
        losses = []
        for i in range(n):
            batch_x = x[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            feed_dict = { placeholder_x: batch_x }
            loss_, x_recon_, features_ = sess.run(
                [loss, x_recon, x_features],
                feed_dict = feed_dict
            )
            losses.append(loss_)

    print(np.mean(losses))
    for i in np.random.random_integers(0, len(batch_x), 5):
        plt.imshow(batch_x[i].reshape(28,28))
        plt.gray()
        plt.show()
        plt.imshow(x_recon_[i].reshape(28,28))
        plt.gray()
        plt.show()
        print(features_[i])
    return np.mean(losses)


def main():
    parser = argparse.ArgumentParser(description='COMP5212 Programming Project 2')
    parser.add_argument('--task', default="train", type=str,
                        help='Select the task, train_cnn, test_cnn, '
                             'train_ae, evaluate_ae, ')
    parser.add_argument('--datapath',default="./data",type=str, required=False,
                        help='Select the path to the data directory')
    args = parser.parse_args()
    datapath = args.datapath
    with tf.variable_scope("placeholders"):
        img_var = tf.placeholder(tf.uint8, shape=(None, 28, 28), name="img")
        label_var = tf.placeholder(tf.int32, shape=(None,), name="true_label")

    if args.task == "train_cnn":
        file_train = np.load(datapath+"/data_classifier_train.npz")
        x_train = file_train["x_train"]
        y_train = file_train["y_train"]
        train_cnn(x_train, y_train, img_var, label_var)
    elif args.task == "test_cnn":
        file_test = np.load(datapath+"/data_classifier_test.npz")
        x_test = file_test["x_test"]
        y_test = file_test["y_test"]
        accuracy = test_cnn(x_test, y_test,img_var,label_var)
        print("accuracy = {}\n".format(accuracy))
    elif args.task == "train_ae":
        file_unsupervised = np.load(datapath + "/data_autoencoder_train.npz")
        x_ae_train = file_unsupervised["x_ae_train"]
        train_ae(x_ae_train, img_var)
    elif args.task == "evaluate_ae":
        file_unsupervised = np.load(datapath + "/data_autoencoder_eval.npz")
        x_ae_eval = file_unsupervised["x_ae_eval"]
        evaluate_ae(x_ae_eval, img_var)


if __name__ == "__main__":
    main()
