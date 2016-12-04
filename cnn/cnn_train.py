import tensorflow as tf
import cnn_build
import numpy as np


save_frequency = 2
save_path = './model.ckpt'

#hyperparamater
max_steps = 10000

def train(train_dir, max_steps):
	global_step = tf.Variable(0, trainable=False)
	batch_size, videos, labels, assignOp, numPhonemes = cnn_build.inputs(train_dir)
	print('input')
	logits = cnn_build.inference(videos, batch_size, numPhonemes)
	print('logits')
	loss, classifications = cnn_build.loss(logits, labels, batch_size)
	print('loss')
	train_op = cnn_build.train(loss, global_step, batch_size)
	print('train')
	init_op = tf.initialize_all_variables()
	print('init')
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init_op)
		sess.run(assignOp)
		print(videos.eval())
		for step in xrange(max_steps):
			_, loss_value, accuracy = sess.run([train_op, loss, classifications])
			print 'At step ' + str(step) + ', loss = ' +  str(loss_value) + ' and accuracy = ' + str(np.sum(accuracy)) + '/' + str(batch_size)
			if step%save_frequency == 0:
				saver.save(sess, save_path)
train_dir = './test/'
train(train_dir, max_steps)
