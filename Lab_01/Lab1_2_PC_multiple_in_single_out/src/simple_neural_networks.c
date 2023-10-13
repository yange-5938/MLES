#include "simple_neural_networks.h"

double single_in_single_out_nn(double  input, double weight) {
	// TODO: Return the result of multiplication of input and its weight.
   	return input * weight;
}


double weighted_sum(double * input, double * weight, uint32_t INPUT_LEN){
	double output = 0;
	// TODO: Use for loop to multiply all inputs with their weights
	for(int i =0; i< INPUT_LEN;i++){
		output += input[i]*weight[i];
	}
	
 return output;
}


double multiple_inputs_single_output_nn(double * input, double *weight, uint32_t INPUT_LEN){
	double predicted_value = 0;
	// TODO: Use weighted_sum function to calculate the output
	
	predicted_value =  weighted_sum(input, weight,INPUT_LEN);
	return predicted_value;
}
