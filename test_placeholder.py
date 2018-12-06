import tensorflow as tf
import mask_unused_gpus

mask_unused_gpus.mask_unused_gpus(2)

W = tf.Variable([2.5, 4.0], tf.float32, name='var_W')
#here W is a Variable
x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([5.0, 10.0], tf.float32, name='var_b')
#b is also a variable with initial value 5 and 10
y = W * x + b

xs = tf.placeholder(tf.string)
ys = 'franz' + xs

fp = tf.placeholder(tf.string)
# fps = tf.strings.substr(fp, 2, 5)
fps = tf.string.regex_replace(fp, '(d).png', "XXX")
decoded_image = tf.to_float(tf.image.decode_jpeg(tf.read_file(fps), channels=3,
                                                         try_recover_truncated=True))


#initialize all variables defined
init = tf.global_variables_initializer()
#global_variable_initializer() will declare all the variable we have initilized
# use with statement to instantiate and assign a session
with tf.Session() as sess:
    sess.run(init)
    #this computation is required to initialize the variable
    print("Final result: Wx + b = ", sess.run(y, feed_dict={x: [10, 100]}))
    print("ys=", sess.run(ys, feed_dict={xs: 'binarization/'}))

    test = 'binarization/training_ms_tex/single_channel_small/images/z30.png'

    decoded_image_real = sess.run(decoded_image, feed_dict={fp: test})
    # decoded_image = sess.run(tf.to_float(tf.image.decode_jpeg(tf.read_file(test), channels=3,
                                                        #  try_recover_truncated=True)))
    print(decoded_image_real)

# changing values
number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()
result = number.assign(tf.multiply(number, multiplier))
