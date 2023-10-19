#include <check.h>
#include <stdio.h>
#include <stdbool.h>
#include "../src/simple_neural_networks.h"

// Define a global test fixture to hold shared parameters
typedef struct {
    double learning_rate;
} TestFixture;

TestFixture tf; // Declare a global instance

// Setup function to initialize the test fixture
void setup(void) {
    tf.learning_rate=0.01;
}

void teardown(void) {
    // Cleanup code if needed
}

// Custom function to compare two doubles with the first 5 digits
bool areDoublesEqualWithTolerance(double a, double b, int digits) {
    double tolerance = pow(10, -digits);
    return fabs(a - b) < tolerance;
}

START_TEST(test_single_in_single_out_nn)
{
    // Define input and weight values
    double input = 5.0;
    double weight = 2.0;

    // Call the function to get the result
    double result = single_in_single_out_nn(input, weight);

    // Check if the result is as expected
    ck_assert_double_eq(result, 10.0);
} END_TEST

// Test case for the weighted_sum function
START_TEST(test_multiple_inputs_single_output_nn) {
    double input[] = {1, 2.0};
    double weight[] = {0.5, 0.25};
    uint32_t INPUT_LEN = 2;
    double result = multiple_inputs_single_output_nn(input, weight, INPUT_LEN);
    
    // Check if the result is close to the expected value (considering floating-point precision)
    ck_assert_double_eq(result, 1.0);
} END_TEST

START_TEST(test_relu_backward) {
    // uint32_t m;
    // uint32_t LAYER_LEN;
    // double dA[2][3];  // Adjust the size according to m and LAYER_LEN
    // double Z[2][3];   // Adjust the size according to m and LAYER_LEN
    // double dZ[2][3];  // Adjust the size according to m and LAYER_LEN
    // // Call the function to calculate dZ using the provided dA and Z matrices
    // relu_backward(m, LAYER_LEN, dA, Z,dZ);
    
    // // Add your assertions to check if dZ was calculated correctly
    // ck_assert_double_eq(dZ[0][0], 1.0);  // Adjust the expected values as needed
    // ck_assert_double_eq(dZ[0][1], 1.0);
    // ck_assert_double_eq(dZ[0][2], 1.0);
    // ck_assert_double_eq(dZ[1][0], 1.0);
    // ck_assert_double_eq(dZ[1][1], 1.0);
    // ck_assert_double_eq(dZ[1][2], 1.0);
} END_TEST

// Test case for the linear_backward function
START_TEST(test_linear_backward) {
    uint32_t LAYER_LEN = 1;
    uint32_t PREV_LAYER_LEN = 3;
    uint32_t m = 1;
    // Initialize dZ, A_prev, dW, and db matrices/vectors
    double dZ2[1][1] = {{-0.172060}};       // Adjust the size according to m and LAYER_LEN
    double A_prev[3][1] = {{0.750000},{1.620000},{0.950000}};    // Adjust the size according to m and PREV_LAYER_LEN
    double dW2[1][3] = {{ 0,0,0}};        // Adjust the size according to LAYER_LEN and PREV_LAYER_LEN
    double db2[1] = {0};           // Adjust the size according to LAYER_LEN

    // Call the function to calculate dW, db using the provided dZ, A_prev matrices
    linear_backward(LAYER_LEN, PREV_LAYER_LEN, m, dZ2, A_prev, dW2, db2);
    
    // printf("A_prev \n");
    // matrix_print(1,3,A_prev);
    // printf("dW2 \n");
	// matrix_print(1, 3, dW2);

    // printf("db2 \n");
	// matrix_print(1, 1, db2);

    // Add your assertions to check if dW, db were calculated correctly
    ck_assert(areDoublesEqualWithTolerance(dW2[0][0], -0.129045,6));  // Adjust the expected values as needed
    ck_assert(areDoublesEqualWithTolerance(dW2[0][1], -0.278737,6));
    ck_assert(areDoublesEqualWithTolerance(dW2[0][2], -0.163457,6));

    ck_assert(areDoublesEqualWithTolerance(db2[0], -0.172060,3));
} END_TEST

// Test case for the linear_backward function
START_TEST(test_linear_backward_2) {
    uint32_t LAYER_LEN = 3;
    uint32_t PREV_LAYER_LEN = 3;
    uint32_t m = 1;
    // Initialize dZ, A_prev, dW, and db matrices/vectors
    double dZ1[3][1] = {{-0.082589},{-0.125604},{-0.005162}};       // Adjust the size according to m and LAYER_LEN
    double A_prev[3][1] = {{1},{1},{0}};    // Adjust the size according to m and PREV_LAYER_LEN
    double dW1[3][3] = {{0,0,0},{0,0,0},{0,0,0}};        // Adjust the size according to LAYER_LEN and PREV_LAYER_LEN
    double db1[3] = {0,0,0};           // Adjust the size according to LAYER_LEN


    // Call the function to calculate dW, db using the provided dZ, A_prev matrices
    linear_backward(LAYER_LEN, PREV_LAYER_LEN, m, dZ1, A_prev, dW1, db1);
    
    // Add your assertions to check if dW, db were calculated correctly
    // printf("dW2[0][0] is: %f\n",dW2[0][0]);
    // updated W1  
    //  0.250826 	 0.500826 	 0.050000 	
    //  0.801256 	 0.821256 	 0.300000 	
    //  0.500052 	 0.450052 	 0.190000 	
    // w1
    // {{0.25, 0.5,   0.05},   	 //hid[0]
    // {0.8,  0.82,  0.3 },     //hid[1]
    // {0.5,  0.45,  0.19}}; 

    // updated b1  
    //  0.000826 	
    //  0.001256 	
    //  0.000052 
    ck_assert(areDoublesEqualWithTolerance(dW1[0][0], -0.0826,3));  // Adjust the expected values as needed
    ck_assert(areDoublesEqualWithTolerance(dW1[0][1], -0.0826,3));
    ck_assert(areDoublesEqualWithTolerance(dW1[0][2], -0.0000,3));
    ck_assert(areDoublesEqualWithTolerance(dW1[1][0], -0.1256,3));  // Adjust the expected values as needed
    ck_assert(areDoublesEqualWithTolerance(dW1[1][1], -0.1256,3));
    ck_assert(areDoublesEqualWithTolerance(dW1[1][2], -0.0000,3));
    ck_assert(areDoublesEqualWithTolerance(dW1[2][0], -0.0052,3));  // Adjust the expected values as needed
    ck_assert(areDoublesEqualWithTolerance(dW1[2][1], -0.0052,3));
    ck_assert(areDoublesEqualWithTolerance(dW1[2][2], -0.0000,3));

    ck_assert(areDoublesEqualWithTolerance(db1[0], -0.0826,3));
    ck_assert(areDoublesEqualWithTolerance(db1[1], -0.1256,3));
    ck_assert(areDoublesEqualWithTolerance(db1[2], -0.0052,3));
} END_TEST

int main(void)
{
    Suite *s1 = suite_create("SimpleNeuralNetworks");
    TCase *tc1 = tcase_create("SingleInSingleOutNN");
    TCase *tc2 = tcase_create("multiple_inputs_single_output_nn");
    TCase *tc3 = tcase_create("linear_backward");
    SRunner *sr = srunner_create(s1);

    // add test cases
    tcase_add_test(tc1, test_single_in_single_out_nn);
    tcase_add_test(tc2, test_multiple_inputs_single_output_nn);
    tcase_add_test(tc3, test_linear_backward);
    tcase_add_test(tc3, test_linear_backward_2);


    // Add setup and teardown functions to the TCase
    tcase_add_checked_fixture(tc1, setup, teardown);
    tcase_add_checked_fixture(tc2, setup, teardown);
    tcase_add_checked_fixture(tc3, setup, teardown);
    
    // Add the test case to the test suite
    suite_add_tcase(s1, tc1);
    suite_add_tcase(s1, tc2);
    suite_add_tcase(s1, tc3);

    srunner_run_all(sr, CK_NORMAL);
    int failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (failed == 0) ? 0 : 1;
}
