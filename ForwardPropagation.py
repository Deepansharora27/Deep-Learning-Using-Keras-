# I have Coded the Forward Propagation Algorithm Below :


# Importing the Necessary Files :
import numpy as np

input_data = np.array([2, 3]) #We will take cross-product of input_data and weights :

weights = {
    'node_0': np.array([1, 1]),
    'node_1': np.array([-1, 1]),
    'output': np.array([2, -1])
}

node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])

# Printing the Values in the Nodes of the Hidden Layer :
print(f'Values present in Hidden Layer Nodes  are:', hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()

# Printing the Final Output :
print(f'The Final Output is :', output)

'''
Sample Test Case (Forward Propagation Algorithm) ==>
Input:    Hidden Layer :      Output Layer(Will be a Single layer)
2            {5}                        9  


3            {1}                SAME OUTPUT         

'''
