#include <iostream>
#include <vector>
#include <random>
#include <cmath>

std::mt19937 gen(42);

//  initialize parameters randomly
    // std::random_device rd;
    // std::mt19937 gen(rd());
void initialize_parameters(std::vector<std::vector<double>>& weights, std::vector<double>& biases, int input_dim, int output_dim) {
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < output_dim; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            weights[i][j] = dis(gen);
        }
    }

    for (int i = 0; i < output_dim; ++i) {
        biases[i] = dis(gen);
    }
}

// Mathematically, a linear layer can be represented as:

// Y = XW + b

// where:

// X is the input vector of size n x m, where n is the batch size and m is the number of input features.
// W is the weight matrix of size m x p, where p is the number of output features.
// b is the bias vector of size p.
// Y is the output vector of size n x p

void normalize_columns(std::vector<std::vector<double>>& weights, int input_dim, int output_dim) {
    for (int j = 0; j < input_dim; ++j) { 
        double sum_of_squares = 0.0;

        for (int i = 0; i < output_dim; ++i) {
            sum_of_squares += weights[i][j] * weights[i][j];
        }

        double norm_factor = std::sqrt(sum_of_squares);
        if (norm_factor > 0) {
            for (int i = 0; i < output_dim; ++i) {
                weights[i][j] /= norm_factor;
            }
        }
    }
}

std::vector<double> linear_layer(const std::vector<double>& input, const std::vector<std::vector<double>>& weights, const std::vector<double>& biases) {
    int output_dim = weights.size();
    int input_dim = input.size();
    std::vector<double> output(output_dim, 0.0);

    for (int i = 0; i < output_dim; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            output[i] += weights[i][j] * input[j];
        }
        output[i] += biases[i];
    }

    return output;
}

void print_vector(const std::vector<double>& vec) {
    for (double val : vec) {
        std::cout << val << " ";
    }

    std::cout << std::endl;
}

int main() {
    int input_dim = 10; 
    int output_dim = 3; 

    std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}; // 1*5

    std::vector<std::vector<double>> weights(output_dim, std::vector<double>(input_dim));  // m * p
    std::vector<double> biases(output_dim);  // p

    initialize_parameters(weights, biases, input_dim, output_dim);

    normalize_columns(weights, input_dim, output_dim);

    std::vector<double> output = linear_layer(input, weights, biases);

    std::cout << "Output: ";
    print_vector(output);


    // for (int i = 0; i < output_dim; ++i) {
    //     for (int j = 0; j < input_dim; ++j) {
    //         std::cout << weights[i][j] << " ";
    //         }
    // std::cout << std::endl;
    // }

    return 0;
}
