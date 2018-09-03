# Ideas:
# * Combine left and right into a single input
# * Remove X, Y and Up, I never press those
# 

import tensorflow as tf
import socket
import sys
import traceback
import math
import random
import time
import datetime
import Config as config
import os

ValidationPeriod = config.get_validation_period()
def do_validation(session, state):
	global model, data, best_cost
	#global global_step
	global step
	
	feed_dict = {
		data.input: data.data[-1],
		data.output: data.labels[-1],
		data.cost_weight: data.cost[-1],
	}
	for i, (c, h) in enumerate(model.initial_state):
		feed_dict[c] = state[i].c
		feed_dict[h] = state[i].h
		
	#(cost, step) = session.run([model.cost, global_step], feed_dict)
	cost = session.run(model.cost, feed_dict)
	
	print("Batches: %05d Cost: %.4f (%s)" % (step, cost, str(datetime.datetime.now())))
	
	if best_cost == 0 or cost < best_cost:
		best_cost = cost
		print("Saving model.")
		global saver
		saver.save(session, config.get_checkpoint_dir() + '\mario', global_step=step, write_meta_graph=False)

data = config.get_data(training=True)
print("Batches: %d Batch Size: %d Sequence Length: %d" % (data.num_batches, data.batch_size, data.num_steps))

model = config.get_model(data, training=True)

init = tf.global_variables_initializer()

#global_step = tf.contrib.framework.get_or_create_global_step()
saver = tf.train.Saver()
#sv = tf.train.Supervisor(logdir=config.get_checkpoint_dir())

best_cost = 0
step = 0

#with sv.managed_session(config=tf.ConfigProto(log_device_placement=True)) as session:
#with sv.managed_session() as session:
with tf.Session() as session:

	last_checkpoint = tf.train.latest_checkpoint(config.get_checkpoint_dir() + '\\')
	if last_checkpoint != None:
		print("Restoring session from " + last_checkpoint)
		saver.restore(session, last_checkpoint)
		step = int(last_checkpoint.split("-")[-1])+1
	else:
		session.run(init)

	state = session.run(model.initial_state)
	validation_state = state

	fetches = {
		"train": model.train_op,
		"final_state": model.final_state
	}
	
	start_time = time.time()
	last_validation = start_time - ValidationPeriod

	while True:
		try:
			state = session.run(model.initial_state)
			
			#data.random_reorder()
			for b in range(data.num_batches-1):
				#if sv.should_stop():
				#	break
				
				if time.time() - last_validation > ValidationPeriod:
					last_validation += ValidationPeriod
					do_validation(session, validation_state)
				
				feed_dict = {
					data.input: data.data[b],
					data.output: data.labels[b],
					data.cost_weight: data.cost[b],
				}

				for i, (c, h) in enumerate(model.initial_state):
					feed_dict[c] = state[i].c
					feed_dict[h] = state[i].h
				
					
				vals = session.run(fetches, feed_dict)
				state = vals["final_state"]
				
				step = step + 1
			
			validation_state = state
			
		except KeyboardInterrupt:
			#step = session.run(global_step)
			print("Manually saving model.")
			saver.save(session, config.get_checkpoint_dir() + '\mario', global_step=step, write_meta_graph=False)