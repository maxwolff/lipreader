import tensorflow as tf
import cnn2d_build
import numpy as np

checkpoint_dir = ''
test_dir = ''


batch_size, images, labels, _, numPhonemes = cnn2d_build.inputs(test_dir)
logits = cnn2d_build.inference(images, batch_size, numPhonemes)
predictions = tf.nn.in_top_k(logits, labels, 1)
saver = tf.train.Saver()
with tf.Session() as sess:
	checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
	saver.restore(sess, checkpoint.model_checkpoint_path)
	accuracy = sess.run(predictions)
	print 'test accuracy = ' + str(np.sum(accuracy)) + '/' + str(batch_size)
