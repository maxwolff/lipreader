import tensorflow as tf
import cnn_build

#hyperparamater
max_steps = 1000000

def train(train_dir, max_steps):
	global_step = tf.Variable(0, trainable=False)
	batch_size, videos, labels = cnn_build.inputs(train_dir)
	logits = cnn_build.inference(videos, batch_size)
	loss = cnn_build.loss(logits, labels)
	train_op = cnn_build.train(loss, global_step, batch_size)
	init_op = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init_op)
		for step in xrange(max_steps):
			_, loss_value = sess.run([train_op, loss])
			if step%1000 == 0:
				print("At step %d loss = %d", step, loss_value)
train_dir = #####
train(train_dir, max_steps)