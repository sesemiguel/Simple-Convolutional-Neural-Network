##################################################
#		      IMPORTS    		 #
##################################################
import numpy as np
from scipy import signal
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

##################################################
#		     FUNCTIONS    		 #
##################################################

# Generating new random values for the weights
def new_weights_random(x_size, y_size, flag):
	if flag == 0:
		return np.random.rand(x_size, y_size)#np.zeros((x_size,y_size))np.zeros((x_size,y_size))
	elif flag == 1:
		return fc_weight

# Softmax activation implementation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Sigmoid activation implementation
def sigmoid_function(x):
	return 1/(1+np.exp(-x))

# Error calculation
def error_calculation(target_array, output_array):
	return output_array - target_array

# ReLU activation implementation
def relu_activation(data_array):
    return np.maximum(data_array, 0)

# Getting the max filter weights for 3D
def get_max_filter_weights(max_filter_weights, input_filter):
	for i in range(0,3):
		for j in range(0,3):
			for k in range(0,3):
				if(max_filter_weights[i][j][k] == 0):
						max_filter_weights[i][j][k] = input_filter[i][j][k]
				else:
						if(input_filter[i][j][k] > max_filter_weights[i][j][k]):
							max_filter_weights[i][j][k] = input_filter[i][j][k]

# Getting the min filter weights for 3D 
def get_min_filter_weights(min_filter_weights, input_filter):
	for i in range(0,3):
		for j in range(0,3):
			for k in range(0,3):
				if(min_filter_weights[i][j][k] == 0):
						min_filter_weights[i][j][k] = input_filter[i][j][k]
				else:
						if(input_filter[i][j][k] < min_filter_weights[i][j][k]):
							min_filter_weights[i][j][k] = input_filter[i][j][k]

# Getting the max FC weights for 2D
def get_max_FC_weights(max_filter_weights, input_filter):
	for i in range(0,12):
		for j in range(0,4):
				if(max_filter_weights[i][j] == 0):
						max_filter_weights[i][j] = input_filter[i][j]
				else:
						if(input_filter[i][j] > max_filter_weights[i][j]):
							max_filter_weights[i][j] = input_filter[i][j]

# Getting the min FC weights for 2D
def get_min_FC_weights(max_filter_weights, input_filter):
	for i in range(0,12):
		for j in range(0,4):
				if(max_filter_weights[i][j] == 0):
						max_filter_weights[i][j] = input_filter[i][j]
				else:
						if(input_filter[i][j] < max_filter_weights[i][j]):
							max_filter_weights[i][j] = input_filter[i][j]

# Writing 2d arrays to a CSV file
def pd_2darray_to_csv(input_array, filename_output):
	np.savetxt(filename_output, input_array, delimiter=",")

# Writing 2d arrays to a CSV file
def pd_3darray_to_csv(input_array, filename_output):
	stacked = pd.Panel(input_array.swapaxes(1,2)).to_frame().stack().reset_index()
	stacked.columns = ['x', 'y', 'z', 'value']
	stacked.to_csv(filename_output, index=False)

##################################################
#		   INITIALIZATION   		 #
##################################################

# Number of decimal places
np.set_printoptions(precision=4)

# Target array
target_array = np.array([1,0,0,0])

# Random initial weights for convolution layer to output layer
convolved_nodes_to_output_nodes = new_weights_random(12, 4, 0)
print("INITIAL WEIGHTS")
print(convolved_nodes_to_output_nodes)

# Save initial weights to CSV file
pd_2darray_to_csv(convolved_nodes_to_output_nodes, "Initial_weights_small_scale.csv")

# Initialize convolution layer
convolved_nodes = np.zeros([3,2,2])

# Initialize maximum and minimum filter weights
max_filter_weights = np.zeros((3,3,3))
min_filter_weights = np.zeros((3,3,3))

# Initialize maximum and minimum FC weights
max_FC_weights = np.zeros((12,4))
min_FC_weights = np.zeros((12,4))

# Input for convolution array
input_img = np.invert(np.array(Image.open("sabog.png").convert("L")))/255
print("INPUT BINARY ARRAY")
print(input_img)

# Show input image
plt.imshow(input_img)
plt.axis('off')
plt.title("Input Image")
plt.show()

# Input filter
input_filter = np.array([[[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]],[[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]],[[0.1,0.1,.1],[0.1,.1,0.1],[.1,0.1,0.1]]])#np.random.rand(3, 3, 3)#np.array([[[0,1,0],[0,1,0],[0,1,0]],[[0,0,0],[1,1,1],[0,0,0]],[[0,0,1],[0,1,0],[1,0,0]]])#new_weights_random(3,3,0)#np.array([[0,0,0,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,0]])
print("INPUT FILTERS")
print(input_filter)

# Number of iterations
iteration = 1000

# Temporary value for incrementing convolution nodes
temp = 0

# Iteration only for same input image only
for x in range(0,iteration):

	# Print iteration number
	print("ITERATION: ", x)

####################################################
#                   FORWARD PASS         	   #
####################################################

	# Convolution
	for x in range(0, input_filter.shape[0]):
		convolved_nodes[x] = signal.convolve(input_img, input_filter[x], mode="valid")

	# Sigmoid activation of convolution node
	convolved_nodes_sigmoid = sigmoid_function(convolved_nodes)

	# Flattening of sigmoid activated convolution layer
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,12)

	# Fully connected layer
	output_nodes_flat = np.matmul(convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes)

	# Softmax activation of output node
	output_nodes_flat_column = np.transpose(output_nodes_flat)#.reshape(10,1)#for column comparison
	softmax_output = softmax(output_nodes_flat_column)

####################################################
#                  BACKPROPAGATION       	   #
####################################################

	# Error calculation
	softmax_output_row = np.transpose(softmax_output)#.reshape(1,10)
	error_array = error_calculation(target_array,softmax_output_row)

	# Error * Sigmoid backpropagation formula
	A = error_array * softmax_output_row * (1-softmax_output_row)

	# Transpose result A
	A = np.transpose(A)

	# SOP of A and Nodes in flat form
	B = np.matmul(A,convolved_nodes_sigmoid_flat)

	# Transpose result B
	B = np.transpose(B)

	# Updating of FC weights
	for i in range(0,12):
		for j in range(0,4):
			convolved_nodes_to_output_nodes[i][j] = convolved_nodes_to_output_nodes[i][j] - .5 * B[i][j]

	# Updating of Convolution layer nodes
	for i in range(0,12):
		for j in range(0,4):
			convolved_nodes_sigmoid_flat[0][i] = (softmax_output_row[0][j] * convolved_nodes_to_output_nodes[i][j]) + temp 
			temp = convolved_nodes_sigmoid_flat[0][i]
		temp = 0

	# Reshaping of flat convolution nodes back to 2x2 with a depth of 3
	convolved_nodes = convolved_nodes_sigmoid_flat.reshape(3,2,2)

	# Get max/min filter weights
	get_max_filter_weights(max_filter_weights, input_filter)
	get_min_filter_weights(min_filter_weights, input_filter)

	# Get max/min FC weights
	get_max_FC_weights(max_FC_weights, convolved_nodes_to_output_nodes)
	get_min_FC_weights(min_FC_weights, convolved_nodes_to_output_nodes)

	# Updating the filters
	for x in range(0, input_filter.shape[0]):
		input_filter[x] = signal.convolve(input_img, convolved_nodes[x], mode="valid")

####################################################
#                      OUTPUTS       		   #
####################################################

#  Printing and writing to CSV of the final/maximum/minimum filter weights and FC weights respectively
print("FINAL CONV FILTERS")
print(input_filter)
pd_3darray_to_csv(input_filter, "Final_filters_small_scale.csv")
print("MAX CONV FILTERS")
print(max_filter_weights)
pd_3darray_to_csv(max_filter_weights, "Max_final_filters_small_scale.csv")
print("MIN CONV FILTERS")
print(min_filter_weights )
pd_3darray_to_csv(min_filter_weights, "Min_final_filters_small_scale.csv")
print("FINAL FULLY CONNECTED WEIGHTS")
print(convolved_nodes_to_output_nodes)
pd_2darray_to_csv(convolved_nodes_to_output_nodes, "Final_Weights_small_scale.csv")
print("MAX FULLY CONNECTED WEIGHTS")
print(max_FC_weights)
pd_2darray_to_csv(max_FC_weights, "Max_Final_Weights_small_scale.csv")
print("MIN FULLY CONNECTED WEIGHTS")
print(min_FC_weights)
pd_2darray_to_csv(min_FC_weights, "Min_Final_Weights_small_scale.csv")

# Plotting final filter weights
plt.subplot(331).set_title("Final Filter 1")
plt.imshow(input_filter[0])
plt.subplot(332).set_title("Final Filter 2")
plt.imshow(input_filter[1])
plt.subplot(333).set_title("Final Filter 3")
plt.imshow(input_filter[2])
plt.show()
