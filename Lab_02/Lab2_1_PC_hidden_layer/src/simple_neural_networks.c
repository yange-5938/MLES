#include "simple_neural_networks.h"

double single_in_single_out_nn(double  input, double weight) {
	// TODO: Return the result of multiplication of input and its weight.
   	return input* weight;
}


double weighted_sum(double * input, double * weight, uint32_t INPUT_LEN) {
	double output = 0;
	// TODO: Use for loop to multiply all inputs with their weights
	for( int i = 0; i<INPUT_LEN;i++ ){
		output += input[i] * weight[i]; 
	}

 return output;
}


double multiple_inputs_single_output_nn(double * input, double *weight, uint32_t INPUT_LEN) {
	double predicted_value = 0;
	// TODO: Use weighted_sum function to calculate the output
	predicted_value = weighted_sum(input,weight,INPUT_LEN);
	
	return predicted_value;
}


void elementwise_multiple( double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN) {
	// TODO: Use for loop to calculate output_vector

	for (int i =0; i < VECTOR_LEN;i++){
		output_vector[i] = input_scalar * weight_vector[i];
	}
	
}

void single_input_multiple_output_nn(double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN){
  elementwise_multiple(input_scalar, weight_vector,output_vector,VECTOR_LEN);
}


void matrix_vector_multiplication(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {
	// TODO: Use two for loops to calculate output vector based on the input vector and weights matrix

	for (int i = 0; i < OUTPUT_LEN; i++){
		for (int j = 0; j < OUTPUT_LEN; j++){
			output_vector[i] += input_vector[j]*weights_matrix[i][j];
		}
	}

}


void multiple_inputs_multiple_outputs_nn(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {
	matrix_vector_multiplication(input_vector,INPUT_LEN,output_vector,OUTPUT_LEN,weights_matrix);
}


void hidden_nn( double *input_vector, uint32_t INPUT_LEN,
				uint32_t HIDDEN_LEN, double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
				uint32_t OUTPUT_LEN, double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN], double *output_vector) {
	/* TODO: Use matrix_vector_multiplication to calculate values for hidden_layer. Make sure that when you initialize
	   hidden_pred_vector variable then zero its value with for loop */
	double hidden_pred_vector[HIDDEN_LEN];
	for(int i = 0;i < HIDDEN_LEN;i++){
		hidden_pred_vector[i]=0;
	}

	matrix_vector_multiplication(input_vector,INPUT_LEN,hidden_pred_vector,  HIDDEN_LEN, in_to_hid_weights);
	// TODO: Use matrix_vector_multiplication to calculate output layer values from hidden layer

	matrix_vector_multiplication(hidden_pred_vector,HIDDEN_LEN,output_vector, OUTPUT_LEN,  hid_to_out_weights);

}
