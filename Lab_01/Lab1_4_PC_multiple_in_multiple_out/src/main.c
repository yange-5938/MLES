/*
 ============================================================================
 Name        : Lab2_multiple_in_multiple_out.c
 Author      : Mairo Leier
 Version     :
 Copyright   : TalTech
 Description : Multiple in, multiple out in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"

#define  SAD_PREDICTION_IDX		0
#define  SICK_PREDCITION_IDX	1
#define  ACTIVE_PREDICTION_IDX	2

#define OUT_LEN     3
#define IN_LEN		3

double predicted_output[OUT_LEN];
								   //temp, hum,  air_q
double weights[OUT_LEN][IN_LEN] ={  {-2.0, 9.5, 2.0 },        // sad?
									{-0.8, 7.2, 6.3 },        // sick?
									{-0.5, 0.4, 0.9 }  };     // active ?

int main(void) {
	double input_vector[IN_LEN] = {30.0, 87.0, 110.0};	// temp, hum, air_q input values

	multiple_inputs_multiple_outputs_nn(input_vector,IN_LEN,predicted_output,OUT_LEN,weights);

	printf("Sad prediction  : %f\n", predicted_output[SAD_PREDICTION_IDX]);
	printf("Sick prediction  : %f\n", predicted_output[SICK_PREDCITION_IDX]);
	printf("Active prediction  : %f\n", predicted_output[ACTIVE_PREDICTION_IDX]);
	return 0;
}


