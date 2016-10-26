'''
Something has been modified 
The code in the tutorial website may result a Nan error

SO please refer to this one:
https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/mnist/mnist_softmax.py

'''
import struct

import lmdb
import tensorflow as tf
import numpy as np

class Tensorflow_mnist:
    '''
    Basic workflow 
    '''
    def __init__(self):
        pass

    
    def load_mnist_train(self,dbname = 'train'):
        '''
        load from lmdb file
        setup data for trainning
        '''
        print "loading ",dbname
        db = lmdb.open(dbname,readonly=True)
        with db.begin(write=False) as txn:
            cursor = txn.cursor()
            self.height = int(cursor.get('height'))
            self.width = int(cursor.get('width'))
            self.num = int(cursor.get('num'))
            
            self.data = np.fromstring(cursor.get('data'),dtype=np.uint8)
            # attention: if reverse, reconstruct wrong!!!!
            self.data = self.data.reshape([self.num,self.height*self.width])
            self.data = np.transpose(self.data)
            
            self.label = np.fromstring(cursor.get('label'),dtype=np.uint8)
            label_temp = np.zeros([10,self.num])
            for idx,lab in enumerate(self.label):
                label_temp[lab,idx] = 1
            self.label = label_temp        
    def load_mnist_train_off(self,dbname = 'train'):
        '''
        load from lmdb file 
        setup data for trainning, for official version
        '''
        print "Lading training data..."
        db = lmdb.open(dbname,readonly=True)
        with db.begin(write=False) as txn:
            cursor = txn.cursor()
            self.height = int(cursor.get('height'))
            self.width = int(cursor.get('width'))
            self.num = int(cursor.get('num'))
            
            self.data = np.fromstring(cursor.get('data'),dtype=np.uint8)
            # attention: if reverse, reconstruct wrong!!!!
            self.data = self.data.reshape([self.num,self.height*self.width])
            
            self.label = np.fromstring(cursor.get('label'),dtype=np.uint8)
            label_temp = np.zeros([self.num,10])
            for idx,lab in enumerate(self.label):
                label_temp[idx,lab] = 1
            self.label = label_temp        

    def next_batch(self,num):
        batch_xs = np.zeros([self.height*self.width,num])
        batch_ys = np.zeros([10,num])
        for i in range(num):
            ran = np.random.randint(0,self.num)
            batch_xs[:,i] = self.data[:,ran]
            batch_ys[:,i] = self.label[:,ran]
        return batch_xs,batch_ys

    def next_batch_off(self,num):
        batch_xs = np.zeros([num,self.height*self.width])
        batch_ys = np.zeros([num,10])
        for i in range(num):
            ran = np.random.randint(0,self.num)
            batch_xs[i,:] = self.data[ran,:]
            batch_ys[i,:] = self.label[ran,:]
        return batch_xs,batch_ys


    
    def load_mnist_test(self,dbname = 'test'):
        print 'loading ', dbname 
        db = lmdb.open(dbname,readonly=True)
        with db.begin(write=False) as txn:
            cursor = txn.cursor()
            self.test_height = int(cursor.get('height'))
            self.test_width = int(cursor.get('width'))
            self.test_num = int(cursor.get('num'))
            
            self.test_data = np.fromstring(cursor.get('data'),dtype=np.uint8)
            #self.data = self.data.reshape((self.height,self.width,self.num))
            self.test_data = self.test_data.reshape([self.test_num,self.test_height*self.test_width])
            self.test_data = np.transpose(self.test_data)
            
            self.test_label = np.fromstring(cursor.get('label'),dtype=np.uint8)
            label_temp = np.zeros([10,self.test_num])
            for idx,lab in enumerate(self.test_label):
                label_temp[lab,idx] = 1
            self.test_label = label_temp
    def load_mnist_test_off(self,dbname = 'test'):
        print 'loading ', dbname 
        db = lmdb.open(dbname,readonly=True)
        with db.begin(write=False) as txn:
            cursor = txn.cursor()
            self.test_height = int(cursor.get('height'))
            self.test_width = int(cursor.get('width'))
            self.test_num = int(cursor.get('num'))
            
            self.test_data = np.fromstring(cursor.get('data'),dtype=np.uint8)
            self.test_data = self.test_data.reshape([self.test_num,self.test_height*self.test_width])
            
            self.test_label = np.fromstring(cursor.get('label'),dtype=np.uint8)
            label_temp = np.zeros([self.test_num,10])
            for idx,lab in enumerate(self.test_label):
                label_temp[idx,lab] = 1
            self.test_label = label_temp

    def train_model(self):
        '''
        !! Numpy array: zeros([a,b]),a:number of rows, b:number of colums !!
        '''
        # placeholder
        x = tf.placeholder(tf.float32,[self.height*self.width,None])
        W = tf.Variable(tf.zeros([10,self.height*self.width]))
        b = tf.Variable(tf.zeros([10,1]))
        y = tf.matmul(W,x) + b
        yT = tf.transpose(y)
        #y = tf.nn.softmax(tf.matmul(W,x) + b)
        y_ = tf.placeholder(tf.float32, [10,None]) 
        y_T = tf.transpose(y_)

        # setup grap here
        #with tf.device('/gpu:0'):   
            #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yT, y_T))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        
        init = tf.initialize_all_variables()
        sess = tf.Session()
        print "Session: Running init..."
        sess.run(init)

        for i in range(500):
            if i%100 == 0:
                print "Running ",i+1," th iteration..."
            batch_xs, batch_ys = self.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # print b
        print sess.run(b)

        correct_prediction = tf.equal(tf.argmax(yT,1), tf.argmax(y_T,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print sess.run(accuracy,feed_dict={x:self.test_data, y_: self.test_label })

        # save model
        save_path = "checkpoint/temp.ckpt"
        saver = tf.train.Saver()
        save_path = saver.save(sess,save_path)
        print "Model saved in ",save_path
        sess.close()




    def evaluate_model(self):
        print 'Evaluating model...'

        W = tf.Variable(tf.zeros([10,self.test_height*self.test_width]))
        b = tf.Variable(tf.zeros([10,1]))

        x = tf.placeholder(tf.float32,[self.test_height*self.test_width,None])
        y_ = tf.placeholder(tf.float32, [10,None])
        y_T = tf.transpose(y_)
        #y = tf.nn.softmax(tf.matmul(W,x) + b)
        y = tf.matmul(W,x) + b
        yT = tf.transpose(y)
        correct_prediction = tf.equal(tf.argmax(yT,1), tf.argmax(y_T,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess,'checkpoint/temp.ckpt')
            print "Model restored"
            print sess.run(b)
            print sess.run(accuracy, feed_dict={x:self.test_data, y_: self.test_label })

        print 'Evaluation done!'


if __name__ == '__main__':
    mnist = Tensorflow_mnist()
    
    mnist.load_mnist_train(dbname='mnist_train')
    mnist.load_mnist_test(dbname='mnist_test')
    
    train = 1
    if train ==0:
        # train
        mnist.train_model()
    else:
        # test
        mnist.evaluate_model()




