/*
 ============================================================================
 Name        : HomeAssignment 1
 Author      : Yange Zheng
 Version     :
 Copyright   : TalTech
============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"


// Size of the layers
#define NUM_OF_FEATURES   	3  	// input values
#define NUM_OF_HID1_NODES	5
#define NUM_OF_HID2_NODES	4
#define NUM_OF_OUT_NODES	1	// output classes

#define DATA_LENGTH 		2

double learning_rate=0.01;

/*Input layer to hidden1 layer*/
double a1[1][NUM_OF_HID1_NODES];	// activation function
double b1[NUM_OF_HID1_NODES][1];		// bias
double z1[1][NUM_OF_HID1_NODES];	// output vector

// Input layer to hidden layer weight matrix
double w1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] =    {{0.25, 0.5,   0.05},   	 //hid[0]
													{0.8,  0.82,  0.3 },     //hid[1]
													{0.5,  0.45,  0.19},
													{0.5,  0.54,  0.39},
													{0.3,  0.45,  0.45}};   //hid[2]


/*hidden1 layer to hidden2 layer*/
double a12[1][NUM_OF_HID2_NODES];	// activation function
double b12[NUM_OF_HID2_NODES][1];		// bias
double z12[1][NUM_OF_HID2_NODES];	// output vector

// hidden layer1 to hidden layer2 weight matrix
double w12[NUM_OF_HID2_NODES][NUM_OF_HID1_NODES] =    {{0.35, 0.2,   0.15, 0.59, 0.45},   	 //hid[0]
													   {0.56, 0.1,   0.3 , 0.49, 0.76},     //hid[1]
													   {0.23, 0.35,  0.2,  0.52, 0.54},
													   {0.6,  0.74,  0.39, 0.19, 0.42}};   //hid[2]


/*Hidden layer2 to output layer*/
double b2[NUM_OF_OUT_NODES][1];
double z2[1][NUM_OF_OUT_NODES];	// Predicted output vector

// Hidden layer to output layer weight matrix
double w2[NUM_OF_OUT_NODES][NUM_OF_HID2_NODES] =    {{0.48, 0.73, 0.03, 0.43}};

// Predicted values
double yhat[1][NUM_OF_OUT_NODES];
double yhat_eg[NUM_OF_OUT_NODES];	// Predicted yhat

// Training data
double train_x[1][NUM_OF_FEATURES];				// Training data after normalization
double train_y[1][NUM_OF_OUT_NODES] = {{1}};  	// The expected (training) y values


int main(void) {
	// Raw training data
	double raw_x[DATA_LENGTH][NUM_OF_FEATURES] = {{32.0, 49.0, 120.0}, {10.0, 34.0, 220.0}};	// temp, hum, air_q input values

	normalize_data_2d(DATA_LENGTH,NUM_OF_FEATURES, raw_x, train_x);	// Data normalization

	printf("normalized train_x \n");
	matrix_print(DATA_LENGTH, NUM_OF_FEATURES, train_x);

	weightsB_zero_initialization(*b1, NUM_OF_HID1_NODES);
	weightsB_zero_initialization(*b12, NUM_OF_HID2_NODES);
	weightsB_zero_initialization(*b2, NUM_OF_OUT_NODES);

	double cost;

	int epoch = 1; 
	for (int i =0; i < epoch; i++){

		linear_forward_nn(*train_x, NUM_OF_FEATURES, z1[0], NUM_OF_HID1_NODES, w1, *b1);
		vector_relu(z1[0],a1[0],NUM_OF_HID1_NODES);

		linear_forward_nn(a1[0], NUM_OF_HID1_NODES, z12[0], NUM_OF_HID2_NODES, w12, *b12);
		vector_relu(z12[0],a12[0],NUM_OF_HID2_NODES);

		linear_forward_nn(a12[0], NUM_OF_HID2_NODES, z2[0], NUM_OF_OUT_NODES, w2, *b2);
		vector_sigmoid(z2[0],yhat[0], NUM_OF_OUT_NODES);

		cost = compute_cost(1, yhat, train_y);

		double dA1[1][NUM_OF_HID1_NODES] = {{0, 0, 0, 0, 0}};
		double dZ1[1][NUM_OF_HID1_NODES] = {{0, 0, 0, 0, 0}};
		
		double dA12[1][NUM_OF_HID2_NODES] = {{0, 0, 0, 0}};
		double dZ12[1][NUM_OF_HID2_NODES] = {{0, 0, 0, 0}};
		
		double dZ2[1][1] = {{0}};

		double dW1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
		double dW12[NUM_OF_HID2_NODES][NUM_OF_HID1_NODES] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
		double dW2[NUM_OF_OUT_NODES][NUM_OF_HID2_NODES] = {{0, 0, 0, 0}};

		double db1[1][NUM_OF_HID1_NODES] = {{0, 0, 0, 0, 0}};
		double db12[1][NUM_OF_HID2_NODES] = {{0, 0, 0, 0}};
		double db2[NUM_OF_OUT_NODES][1] = {0};

		
		matrix_matrix_sub(NUM_OF_OUT_NODES,1,yhat,train_y,dZ2);

		linear_backward(NUM_OF_OUT_NODES,NUM_OF_HID2_NODES,1,dZ2,a12,dW2,db2[0]);
		double W12_T[NUM_OF_HID2_NODES][NUM_OF_OUT_NODES] = {{0},{0},{0},{0}};
		matrix_transpose(NUM_OF_HID2_NODES,NUM_OF_OUT_NODES,w12,W12_T);
		matrix_matrix_multiplication(NUM_OF_HID2_NODES,NUM_OF_OUT_NODES,1,W12_T,dZ2,dA12);
		relu_backward(1,NUM_OF_HID2_NODES,dA12,z12,dZ12);

		linear_backward(NUM_OF_HID2_NODES,NUM_OF_HID1_NODES,1,dZ12,a1,dW2,db12[0]);
		double W2_T[NUM_OF_HID1_NODES][NUM_OF_HID2_NODES] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
		matrix_transpose(NUM_OF_HID1_NODES,NUM_OF_HID2_NODES,w2,W2_T);
		matrix_matrix_multiplication(NUM_OF_HID1_NODES,NUM_OF_HID1_NODES,1,W2_T,dZ12,dA1);
		relu_backward(1,NUM_OF_HID2_NODES,dA1,z1,dZ1);

		linear_backward(NUM_OF_HID1_NODES,NUM_OF_FEATURES,1,dZ1,train_x,dW1,db1[0]);

		weights_update(NUM_OF_FEATURES,NUM_OF_HID1_NODES,learning_rate,dW1,w1);
		weights_update(NUM_OF_FEATURES,NUM_OF_HID1_NODES,learning_rate,db1,b1);
		weights_update(NUM_OF_HID1_NODES,NUM_OF_HID2_NODES,learning_rate,dW2,w12);
		weights_update(NUM_OF_HID1_NODES,NUM_OF_HID2_NODES,learning_rate,db2,b12);
		weights_update(NUM_OF_HID2_NODES,NUM_OF_OUT_NODES,learning_rate,dW12,w2);
		weights_update(NUM_OF_HID2_NODES,NUM_OF_OUT_NODES,learning_rate,db12,b2);
	}

	return 0;

}
