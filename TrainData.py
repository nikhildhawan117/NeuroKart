import tensorflow as tf
import random
import math

def contains_negative(buttons):
	for b in buttons:
		if b < 0:
			return True
	
	return False
	
def unmix(buttons):
	contains_positives = False
	for b in buttons:
		if b >= 0:
			contains_positives = True
	
	if not contains_positives:
		return buttons
		
	unmixed = []
		
	for b in buttons:
		if b < 0:
			unmixed.append(0)
		else:
			unmixed.append(b)
			
	return unmixed

class DataSet(object):
	def __init__(self, filenames, sequence_len, batch_size, num_passes, train, recur_buttons):
		self.batch_size = batch_size
		self.num_steps = sequence_len
		self.recur_buttons = recur_buttons
		self.get_data(filenames, train, num_passes)
		
		self.input = tf.placeholder(shape=[sequence_len, batch_size, self.input_size], dtype = self.dtype)
		self.output = tf.placeholder(shape=[sequence_len, batch_size, self.output_size], dtype = self.dtype)
		self.cost_weight = tf.placeholder(shape=[sequence_len, batch_size, self.output_size], dtype = self.dtype)
	
	def get_data(self, filenames, train, num_passes):
		self.dtype = tf.float32
		self.data = data = []
		self.labels = labels = []
		self.cost = cost = []

		frames = []
		print("Data files:")
		for filename in filenames:
			print(filename)
			sessions = self.get_sessions(filename.strip(), train)
			if sessions != None:
				frames += sessions
				print("{0} frames".format(len(sessions)))
		
		if self.recur_buttons:
			self.input_size += self.output_size
		
		if not train:
			return None
		
		whole_sequence_length = math.floor(len(frames) / self.batch_size)
		whole_sequences = [frames[seq*whole_sequence_length:(seq+1)*whole_sequence_length] for seq in range(self.batch_size)]
		if self.recur_buttons:
			for s in whole_sequences:
				prev = [-1] * self.output_size
				for f in range(len(s)):
					inputs, outputs = s[f]
					inputs += prev
					prev = unmix(outputs)
					s[f] = inputs, outputs
				
		batches_per_pass = math.floor(whole_sequence_length / self.num_steps)-2
		
		for pass_num in range(num_passes):
			if pass_num == num_passes-1:
				batches_per_pass = math.floor(whole_sequence_length / self.num_steps)
					
			for batch in range(batches_per_pass):
				data.append([])
				labels.append([])
				cost.append([])
				for step in range(self.num_steps):
					data[-1].append([])
					labels[-1].append([])
					cost[-1].append([])

			for seq in range(self.batch_size):
				start = random.randint(0, self.num_steps)
				if pass_num == num_passes-1:
					start = 0
				
				for batch in range(batches_per_pass):
					batch_index = len(data) - batches_per_pass + batch
					for step in range(self.num_steps):
						tiles, buttons = whole_sequences[seq][start + batch*self.num_steps + step]
						data[batch_index][step].append(tiles)
						labels[batch_index][step].append(buttons)
						if contains_negative(buttons):
							cost[batch_index][step].append([0]*self.output_size)
						else:
							cost[batch_index][step].append([1]*self.output_size)
							
		self.num_batches = len(data)
		
	def get_sessions(self, filename, train):
		try:
			with open(filename) as f:
				line = f.readline().strip()
				if line == None:
					raise Exception("No data in data file {0}".format(filename))
				
				params = line.split(" ")
				if len(params) < 4:
					raise Exception("Not enough parameters in the first line of the data file.")
					
				self.header = params
				self.input_width = int(params[0])
				self.input_height = int(params[1])
				self.extra_inputs = int(params[2])
				self.input_size = self.input_width*self.input_height + self.extra_inputs
				self.output_size = int(params[3])
				
				if not train:
					return None
					
				lines = f.readlines()
				
			lines = [line.strip() for line in lines]
			
			sessions = []
			
			session = []
			linenum = 0
			while linenum < len(lines):
				if lines[linenum][:7] == "Session":
					if len(session) > 0:
						#sessions.append(session)
						sessions = sessions + session
						session = []
					linenum += 1
					continue
				
				screen = []
				while len(screen) < self.input_size:
					screen += [float(v) for v in lines[linenum].split(" ")]
					linenum += 1
						
				assert(len(screen) == self.input_size)
				
				buttons = [float(button) for button in lines[linenum].split(" ")]
				assert(len(buttons) == self.output_size)
				linenum += 1
				
				session.append((screen, buttons))			
			
			#sessions.append(session)
			sessions = sessions + session
		except:
			raise Exception("Data file parse error on line " + str(linenum) + ":\n" + lines[linenum])
		
		return sessions