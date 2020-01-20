#Code Describing the Usage of Activation Function .


#Here , I am using the tan h activation function ==>

#Step 1 : Importing the Necessary Packages :
import numpy as np

input_data = np.array([-1,2])

#Defining a Weights Dictionary :

weights ={
    'node_0' :np.array([3,3]) ,
    'node_1' :np.array([1,5]),
    'output' :np.array([2,-1])
}

node_0_input = (input_data * weights['node_0']).sum()

#Taking Out the Output for Node 0 Separately this time :
node_0_output = np.tanh(node_0_input) #Currently Present ==> 2

node_1_input = (input_data * weights['node_1']).sum() #CP ==> 9

#Taking the Output for Node 1 Separately this time :
node_1_output = np.tanh(node_1_input)

hidden_layer_outputs = np.array([node_0_output,node_1_output])

output = (hidden_layer_outputs * weights['output']).sum()


#Printing the Final Output :
print(output)


