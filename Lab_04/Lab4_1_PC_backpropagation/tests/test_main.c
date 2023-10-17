#include <check.h>
#include "../src/simple_neural_networks.h"


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
    double input[] = {1.0, 2.0};
    double weight[] = {0.5, 0.25};
    uint32_t INPUT_LEN = 2;
    
    double result = multiple_inputs_single_output_nn(input, weight, INPUT_LEN);
    
    // Check if the result is close to the expected value (considering floating-point precision)
    ck_assert_double_eq(result, 1.0);
} END_TEST

int main(void)
{
    Suite *s1 = suite_create("SimpleNeuralNetworks");
    TCase *tc1 = tcase_create("SingleInSingleOutNN");
    TCase *tc2 = tcase_create("multiple_inputs_single_output_nn");
    SRunner *sr = srunner_create(s1);

    // add test cases
    tcase_add_test(tc1, test_single_in_single_out_nn);
    tcase_add_test(tc2, test_multiple_inputs_single_output_nn);

    // Add the test case to the test suite
    suite_add_tcase(s1, tc1);
    suite_add_tcase(s1, tc2);

    srunner_run_all(sr, CK_NORMAL);
    int failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (failed == 0) ? 0 : 1;
}
