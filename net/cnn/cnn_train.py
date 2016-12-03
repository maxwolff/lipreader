import tensorflow as tf
import cnn_build


#hyperparamater
max_steps = 10000

def train(train_dir, max_steps):
	global_step = tf.Variable(0, trainable=False)
	batch_size, videos, labels, assignOp = cnn_build.inputs(train_dir)
	print('input')
	logits = cnn_build.inference(videos, batch_size)
	print('logits')
	loss, accuracy = cnn_build.loss(logits, labels)
	print('loss')
	train_op = cnn_build.train(loss, global_step, batch_size)
	print('train')
	init_op = tf.initialize_all_variables()
	print('init')
	with tf.Session() as sess:
		sess.run(init_op)
		sess.run(assignOp)
		print(videos.eval())
		for step in xrange(max_steps):
			_, _, accuracy = sess.run([train_op, loss, accuracy])
			print 'At step', step 'accuracy =', accuracy
train_dir = '/Users/wamsood/Documents/Classes 16-17/221/Project/test/'
train(train_dir, max_steps)