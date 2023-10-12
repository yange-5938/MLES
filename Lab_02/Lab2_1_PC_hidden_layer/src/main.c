/*
 ============================================================================
 Name        : Lab2_hidden_layer.c
 Author      : Mairo Leier
 Version     :
 Copyright   : TalTech
 Description : Hidden layer in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"

#define  SAD_PREDICTION_IDX		0
#define  SICK_PREDCITION_IDX	1
#define  ACTIVE_PREDICTION_IDX	2

// Size of the layers
#define OUT_LEN     3
#define IN_LEN		3
#define HID_LEN   	3

double predicted_output[OUT_LEN];
								   	   	   	   	   //temp, hum,  air_q
double input_to_hidden_weights[HID_LEN][IN_LEN] ={  {-2.0, 9.5, 2.0},   	//hid[0]
													{-0.8, 7.2, 6.3},      //hid[1]
													{-0.5, 0.4, 0.9}};     //hid[2]

												   //hid[0] hid[1] hid[2]
double hidden_to_output_weights[OUT_LEN][HID_LEN] ={{-1.0,  1.15,  0.11},   //sad?
													{-0.18, 0.15, -0.01},   //sick?
													{0.25, -0.25, -0.1 }};  //active?

int main(void) {
	double input_vector[IN_LEN] = {30.0, 87.0, 110.0};	// temp, hum, air_q input values

	hidden_nn(input_vector,IN_LEN,HID_LEN,input_to_hidden_weights,OUT_LEN,hidden_to_output_weights,predicted_output);

	printf("Sad prediction: %f\n", predicted_output[SAD_PREDICTION_IDX]);
	printf("Sick prediction: %f\n", predicted_output[SICK_PREDCITION_IDX]);
	printf("Active prediction: %f\n", predicted_output[ACTIVE_PREDICTION_IDX]);
	return 0;
}


