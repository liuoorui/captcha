# captcha_recognization.py
# Date: 2018/11/1

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import random
import time

from captcha.image import ImageCaptcha


numbers = ['0','1','2','3','4','5','6','7','8','9']
alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
                'n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABETS = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
CAPTCHA_SIZE = 4
CHAR_CANDIDATES = numbers
CHAR_CANDIDATES_SIZE = len( CHAR_CANDIDATES )

model_path = "../captcha_recognization_saved_model/captcha_recognization.model"

batch_size = 500
traing_size =1000
test_size = 500

# create a random captcha text
def random_captcha_text( char_set=numbers+alphabets+ALPHABETS, captcha_size=4 ):
    captcha_text = []
    for _ in range( captcha_size ):
        captcha_text.append( random.choice(char_set) )
    
    return captcha_text

# creat a random text and corresponding image
def get_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text( CHAR_CANDIDATES )
    captcha_text = "".join( captcha_text )

    captcha = image.generate( captcha_text )

    captcha_image = Image.open( captcha )
    captcha_image = np.array( captcha_image )

    return captcha_text, captcha_image

def data2vector( data ):
    ret = []
    for d in data:
        v = np.zeros( (CHAR_CANDIDATES_SIZE, ), dtype=np.float32 )
        
        p = CHAR_CANDIDATES.index( d )
        v[p] = 1
        ret.append( v )

    return np.hstack( ret )

# convert a rgb image to gray
def convert2gray( img ):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = gray.reshape( 60, 160, 1 )
        return gray
    else:
        return img

def init_weights( shape ):
    return tf.Variable( tf.random_normal(shape, stddev=0.01) )

def model( X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden ):
    l1a = tf.nn.relu( tf.nn.conv2d( X, w, 
                            strides=[1, 1, 1, 1], padding='SAME')) 
    l1 = tf.nn.max_pool( l1a, ksize=[1, 2, 2, 1], 
                            strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout( l1, p_keep_conv )

    l2a = tf.nn.relu( tf.nn.conv2d( l1, w2, 
                            strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool( l2a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout( l2, p_keep_conv )

    l3a = tf.nn.relu( tf.nn.conv2d( l2, w3, 
                            strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool( l3a, ksize=[1, 2, 2, 1], 
                            strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape( l3, [-1, w4.get_shape().as_list()[0]] )
    l3 = tf.nn.dropout( l3, p_keep_conv )

    l4 = tf.nn.relu( tf.matmul(l3, w4) ) 
    l4 = tf.nn.dropout( l4, p_keep_hidden )

    pyx = tf.matmul( l4, w_o )
    return pyx

# generating training data and test data
trX, trY = [], []
teX, teY = [], []

for _ in range( test_size ):
    y, x = get_captcha_text_and_image() # 60*160*3
    x = convert2gray( x ) # 60*160*1
    teX.append( x ) 
    teY.append( data2vector(y) )
teX = np.array( teX ) # ?*60*160*3
teY = np.array( teY ) # ? * (CAPTCHA_SIZE*CHAR_CANDIDATES_SIZE)

for _ in range( traing_size ):
    y, x = get_captcha_text_and_image() # 60*160*3
    x = convert2gray( x ) # 60*160*1
    trX.append( x )
    trY.append( data2vector(y) )
trX = np.array( trX ) 
trY = np.array( trY )

X = tf.placeholder( "float", [None, 60, 160, 1] )
Y = tf.placeholder( "float", [None, CAPTCHA_SIZE*CHAR_CANDIDATES_SIZE] )

w = init_weights( [3, 3, 1, 32] )
w2 = init_weights( [3, 3, 32, 64] )
w3 = init_weights( [3, 3, 64, 64] )
w4 = init_weights( [8*32*40, 1024] )
w_o = init_weights( [1024, CAPTCHA_SIZE*CHAR_CANDIDATES_SIZE] )

p_keep_conv = tf.placeholder( "float" )
p_keep_hidden = tf.placeholder( "float" )

def crack_captcha_cnn():
    py_x = model( X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden )

    return py_x


py_x = crack_captcha_cnn()
cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( 
    logits=py_x, labels=Y ) )
train_op = tf.train.AdamOptimizer( learning_rate=0.001 ).minimize( cost )
predict_op = tf.argmax( py_x, 1 )

# Launch the graph in a session
pretime = time.time()

with tf.Session() as session:
    # initialize
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    for i in range( 1000 ):
        training_batch = zip( range(0, len(trX), batch_size), 
                        range(batch_size, len(trX)+1, batch_size) )
        for start, end in training_batch:
            _, cost_ = session.run( [train_op, cost],
                        feed_dict={X: trX[start:end], Y: trY[start:end],
                        p_keep_conv: 0.8, p_keep_hidden: 0.5} )

        test_indices = np.arange( len(teX) )
        np.random.shuffle( test_indices )
        test_indices = test_indices[0:test_size]

        max_idx_p = tf.argmax(tf.reshape(py_x, [-1, CAPTCHA_SIZE, CHAR_CANDIDATES_SIZE]), 2)
        max_idx_l = tf.argmax(tf.reshape(Y, [-1, CAPTCHA_SIZE, CHAR_CANDIDATES_SIZE]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc = session.run( accuracy, feed_dict={X: teX[test_indices],
                                                Y: teY[test_indices],
                                                p_keep_conv: 1.0,
                                                p_keep_hidden: 1.0})

        if acc > 0.9:
            saver.save( session, model_path, global_step=i )
            if acc > 0.95:
                break
        
        nowtime = time.time()
        print( "{:d}, cost= {:.10f}, acc= {:.10f}    time_cost= {:.1f} minutes".format( i, cost_, acc, (nowtime-pretime)/60 ) )