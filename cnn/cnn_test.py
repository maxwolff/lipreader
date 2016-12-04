import tensorflow as tf
import cnn2layer_build
import numpy as np

checkpoint_dir = ''
test_dir = ''


batch_size, videos, labels, _, numPhonemes = cnn2layer_build.inputs(test_dir)
logits = cnn2layer_build.inference(videos, batch_size, numPhonemes)
predictions = tf.nn.in_top_k(logits, tf.cast(labels, tf.int32), 1)
saver = tf.train.Saver()
with tf.Session() as sess:
	checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
	saver.restore(sess, checkpoint.model_checkpoint_path)
	accuracy = sess.run(predictions)
	print 'test accuracy = ' + str(np.sum(accuracy)) + '/' + str(batch_size)
