#include "simple_neural_networks.h"
#include <math.h>
#include <stdio.h>

double single_in_single_out_nn(double  input, double weight) {
	// TODO: Return the result of multiplication of input and its weight.
   	return 0;
}


double weighted_sum(double * input, double * weight, uint32_t INPUT_LEN) {
	double output = 0;
	// TODO: Use for loop to multiply all inputs with their weights
 return output;
}


double multiple_inputs_single_output_nn(double * input, double *weight, uint32_t INPUT_LEN) {
	double predicted_value = 0;
	// TODO: Use weighted_sum function to calculate the output
	return predicted_value;
}


void elementwise_multiple( double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN) {
	// TODO: Use for loop to calculate output_vector
}


void single_input_multiple_output_nn(double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN){
  elementwise_multiple(input_scalar, weight_vector,output_vector,VECTOR_LEN);
}


void matrix_vector_multiplication(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {
	// TODO: Use two for loops to calculate output vector based on the input vector and weights matrix
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

	// TODO: Use matrix_vector_multiplication to calculate output layer values from hidden layer

}


double find_error(double yhat, double y) {
	// TODO: Use math.h functions to calculate the error with double precision
	return 0;
}



void brute_force_learning( double input, double weight, double expected_value, double step_amount, uint32_t itr) {
   double prediction,error;
   double up_prediction, down_prediction, up_error, down_error;
   int i;
	 for(i=0;i<itr;i++){

		 prediction  = input * weight;
		 // TODO: Calculate the error
		 error = 0;

		 printf("Step: %d   Error: %f    Prediction: %f    Weight: %f\n", i, error, prediction, weight);

		 up_prediction =  input * (weight + step_amount);
		 up_error      =   powf((up_prediction - expected_value),2);
		 //up_error      =  (up_prediction - expected_value)*(up_prediction - expected_value);	// use, if issues with powf()

		 // TODO: Calculate down_prediction and down_error on the same way as up_prediction and up_error
		 down_prediction =  0;
		 down_error      =  0;

		 if(down_error <  up_error)
			 // TODO: Change weight value accordingly if down_error is smaller than up_error
			 weight  = 0;
		 if(down_error >  up_error)
			 // TODO: Change weight value accordingly if down_error is larger than up_error
			 weight = 0;
		 if(down_error == up_error)
			 weight = 0;
	 }
}
