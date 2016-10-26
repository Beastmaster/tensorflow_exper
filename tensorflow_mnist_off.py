"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md

mnist dataset load module is implemented here
class: 
    tensorflow_mnist_off
    tensorflow_minst_mul_layer
are using data loading module in this class
"""

import tensorflow as tf
import tensorflow_mnist

class tensorflow_mnist_off:
    def load_data(self):
        self.mnist = tensorflow_mnist.Tensorflow_mnist()
        self.mnist.load_mnist_train_off(dbname='mnist_train')
        self.mnist.load_mnist_test_off(dbname='mnist_test')
        
    def train(self):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.Session()
        # Train
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in range(1000):
            batch_xs, batch_ys = self.mnist.next_batch_off(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print (sess.run(b))
    
        
        # save model
        save_path = "checkpoint/temp.ckpt"
        saver = tf.train.Saver()
        save_path = saver.save(sess,save_path)
        print ("Model saved in ",save_path)
        sess.close()

    def evaluate(self):
        print "Evaluating model..."

        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        with tf.Session() as sess:
            init = tf.initialize_all_variables()  
            sess.run(init)

            saver = tf.train.Saver()
            saver.restore(sess,"checkpoint/temp.ckpt")
            print "Model restored"
            ww = sess.run(W)
            print type(ww)
            print ww
            bb = sess.run(b)
            print type(bb)
            print bb
            #print(sess.run(accuracy, feed_dict={x: self.mnist.test_data,y_:self.mnist.test_label}))

if __name__ == '__main__':
    mm = tensorflow_mnist_off()
    #mm.load_data()

    #mm.train()

    mm.evaluate()