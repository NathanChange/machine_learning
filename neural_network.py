import numpy
import scipy.special
import matplotlib.pyplot as plt
class neuralNetwork:
	#初始化函数
	def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
		#各层节点数量
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		#链接权重矩阵
		self.wih = (numpy.random.rand(self.hnodes,self.inodes)-0.5)
		self.who = (numpy.random.rand(self.onodes,self.hnodes)-0.5)
		#激活函数
		self.activation_function = lambda x: scipy.special.expit(x)
		#学习率
		self.lr = learningrate
		pass
	#训练
	def train(self,inputs_list,targets_list):
		inputs=numpy.array(inputs_list,ndmin=2).T
		targets=numpy.array(targets_list,ndmin=2).T
		
		hidden_inputs=numpy.dot(self.wih,inputs)
		hidden_outputs=self.activation_function(hidden_inputs)
		
		final_inputs=numpy.dot(self.who,hidden_outputs)
		final_outputs=self.activation_function(final_inputs)
		
		output_errors=targets-final_outputs
		
		hidden_errors=numpy.dot(self.who.T,output_errors)
		self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
		self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
		pass
	#查询	
	def query(self,inputs_list):
		inputs=numpy.array(inputs_list,ndmin=2).T
		
		hidden_inputs=numpy.dot(self.wih,inputs)
		hidden_outputs=self.activation_function(hidden_inputs)
		
		final_inputs=numpy.dot(self.who,hidden_outputs)
		final_outputs=self.activation_function(final_inputs)
		return final_outputs
		pass


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

train_data_file=open('mnist_train.csv','r')
train_data_list=train_data_file.readlines()
train_data_file.close()
for record in train_data_list:
	a=train_data_list.index(record)/len(train_data_list)
	if train_data_list.index(record)%100==0:
		print("已进行",a*100,"%")
	all_values=record.split(',')
	inputs=numpy.asfarray(all_values[1:])/255.0*0.99+0.01
	targets=numpy.zeros(output_nodes)+0.01
	targets[int(all_values[0])]=0.99
	n.train(inputs,targets)
test_data_file=open('mnist_test.csv','r')
test_data_list=test_data_file.readlines()
test_data_file.close()
all_values=test_data_list[0].split(',')
print(all_values[0])
image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array)
plt.show()
print(n.query(numpy.asfarray(all_values[1:])/255.0*0.99+0.01))
