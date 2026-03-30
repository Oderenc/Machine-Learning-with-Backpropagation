#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <numeric>
#include <fstream>

#include <Eigen/Dense>

#include "assert_system.h"


std::random_device rd;
std::mt19937 gen(rd());

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;


struct neuralNetwork {
    std::vector<int> neurons_per_layer;
    std::vector<MatrixXd> weight_matrices;
    std::vector<VectorXd> bias_vectors;
};


struct layerCache {
    VectorXd z_values;
    VectorXd a_values;
};


struct trainingExamples {
    std::vector<double> x_coords;
    std::vector<double> y_coords;
};


struct neuronCache {
    std::vector<double> z_values;
    std::vector<double> a_values;
};


MatrixXd he_init_layer(int fan_in, int fan_out) {
    float stddev = std::sqrt(2.0 / fan_in);
    std::normal_distribution<double> dist(0.0, stddev);

    return MatrixXd::NullaryExpr(
        fan_out,
        fan_in,
        [&]() { return dist(gen); }
    );
}


void he_init_network(neuralNetwork& network) {

    for (int i = 0; i < network.neurons_per_layer.size() - 1; i++) { //-1 bc if we have a network layers = 1, 2, 3, we push_back(1, 2), (2, 3)
        int j = i + 1;

        int fan_in = network.neurons_per_layer[i];
        int fan_out = network.neurons_per_layer[j];

        network.weight_matrices.push_back(he_init_layer(fan_in, fan_out));
        network.bias_vectors.push_back(VectorXd(fan_out).setConstant(.01)); //Set constant to minimize ReLu death. TODO I just set it to zero.
    }

    return;
}


//regression makes it so we dont do a ReLu on the last layer(so we can predict negative values)
std::vector<layerCache> forward_pass(const neuralNetwork& network, VectorXd& inputLayer, bool regression=true) {
    VectorXd current_neurons = inputLayer;
    std::vector<layerCache> layers;
    layerCache current_layer;

    current_layer.a_values = current_neurons; //so we also cache our input a_vals
    layers.push_back(current_layer);

    for (int i=0; i<network.bias_vectors.size(); i++) {
        current_neurons =  network.weight_matrices[i] * current_neurons;

        current_neurons += network.bias_vectors[i];

        current_layer.z_values = current_neurons;
        //Activation
        if (!(regression)) { 
            for (int j=0; j<network.neurons_per_layer[i+1]; j++) {
                if (current_neurons(j) < 0) {
                    current_neurons(j) = 0;
                }
            }
        }
        current_layer.a_values = current_neurons;

        layers.push_back(current_layer);
    }

    return layers;
}


//for now only one weight/neuron
//could possibly optimize by computing dC/da, then multiplying what we need to in the final differentiation(bias, weight, prev. activation).
std::vector<double> find_gradient_weights(
    const std::vector<double>& predicted_values, const std::vector<double>& pre_activation_predictions,
    const std::vector<double>& actual_values, 
    const std::vector<std::vector<double>>& previous_layer_activations, bool regression=true
) {


    std::vector<double> all_weight_gradients; //weight num, data_num. (diff than the others)
    double gradient;

    for (int weight_num=0; weight_num<previous_layer_activations[0].size(); weight_num++) {
        std::vector<double> current_weight_gradients;
        for(int data_num=0; data_num<actual_values.size(); data_num++) {

            double activation_value = predicted_values[data_num];

            //ReLu didnt activate
            if (!regression && pre_activation_predictions[data_num] < 0) {
                gradient = 0;
            }

            //else {
                double dC_over_da = -2*(actual_values[data_num] - activation_value);
                //da_over_dz = 1 if our ReLu activates

                double ds_over_dw = previous_layer_activations[data_num][weight_num];

                gradient = dC_over_da * ds_over_dw;
            //}

            current_weight_gradients.push_back(gradient);
        }

        double sum_gradients = std::accumulate(current_weight_gradients.begin(), current_weight_gradients.end(), 0.0);
        double avg_gradient = sum_gradients/double(current_weight_gradients.size());
        all_weight_gradients.push_back(avg_gradient);
    }

    return all_weight_gradients;
}


std::vector<double> find_gradient_biases(
    const std::vector<double>& predicted_values, const std::vector<double>& pre_activation_predictions,
    const std::vector<double>& actual_values,
    const std::vector<std::vector<double>>& previous_layer_activations, bool regression=true
) {

    std::vector<double> all_bias_gradients; //bias num, data_num. (diff than the others)
    double gradient;

    for (int bias_num=0; bias_num<previous_layer_activations[0].size(); bias_num++) {
        std::vector<double> current_bias_gradients;
        for(int data_num=0; data_num<predicted_values.size(); data_num++) {

            double activation_value = predicted_values[data_num];

            //if ReLu didnt activate
            if (!regression && pre_activation_predictions[data_num] < 0) {
                gradient = 0;
            }

            else {
                double dC_over_da = -2*(actual_values[data_num] - activation_value);
                //da_over_dz = 1 if our ReLu activates
                //dz_over_db = 1 for biases

                gradient = dC_over_da;
            }

            current_bias_gradients.push_back(gradient);
        }

        double sum_gradients = std::accumulate(current_bias_gradients.begin(), current_bias_gradients.end(), 0.0);
        double avg_gradient = sum_gradients/double(current_bias_gradients.size());
        all_bias_gradients.push_back(avg_gradient);
    }

    return all_bias_gradients;
}


neuronCache get_neuron_activations(int current_layer, int current_neuron,  const auto& every_known_layer) {
    std::vector<double> a_values;
    std::vector<double> z_values;
    layerCache layer;

    for (int example_num=0; example_num<every_known_layer.size(); example_num++) {
        layer = every_known_layer[example_num][current_layer];

        a_values.push_back(layer.a_values(current_neuron));
        z_values.push_back(layer.z_values(current_neuron));
    }

    neuronCache activations;
    activations.a_values = a_values;
    activations.z_values = z_values;
    return activations;
}


void run(trainingExamples training_data, int data_length) {
    /*
    //create training data
    auto linear_equation = [](double x) {return (2.0*x/5.0) + 4;};
    data_length = 50
    for (int i=0; i<data_length; i++){
        std::uniform_real_distribution<double> dist_x(-20.0, 20.0);
        std::uniform_real_distribution<double> dist_randomness(-5.0, 5.0);

        double x = dist_x(gen);
        double y = linear_equation(x) + dist_randomness(gen);
        training_data.x_coords.push_back(x);
        training_data.y_coords.push_back(y);
    }
    */

    neuralNetwork network;
    network.neurons_per_layer = {1, 1};

    he_init_network(network);

    std::cout << "Num Matrices: " << network.weight_matrices.size() << "\n";

    int input_size = 1;
    VectorXd input_layer(input_size);

    std::vector<double> predicted_values;
    std::vector<double> pre_activation_predictions;

    double mean_squared_error = 100;
    double initial_lr = 0.0001;
    double learning_rate = 0.0001;
    double gradient_threshold = 0.0000001;

    int num_epochs = 1000000;

    std::vector<layerCache> layers;
    std::vector<std::vector<layerCache>> every_known_layer;

    std::ofstream bias_file("training_output/bias.csv");
    bias_file << "epoch,bias\n";

    std::ofstream weights_file("training_output/weights.csv");
    weights_file << "epoch,weights\n";

    std::ofstream loss_file("training_output/loss.csv");
    loss_file << "epoch,loss\n";  //header

    std::uniform_int_distribution<int> dist(0, data_length-1);

    for (int epoch = 0; epoch < num_epochs; epoch++) {

    //We dont have enough data for batches to be useful
    //Forward Pass
    every_known_layer.clear(); //Otherwise your cache will be stale
    for (int i=0; i<data_length; i++) {
        input_layer(0) = training_data.x_coords[i];
        layers = forward_pass(network, input_layer);
        every_known_layer.push_back(layers);
    }

    //Calc MSE
    double running_sum = 0;
    for(int example_num=0; example_num<data_length; example_num++) {
        for(int j=0; j<network.neurons_per_layer.back(); j++) {
            //TODO, change training_data to vec<vec> ot make more sense on larger models with >1 output.
            layerCache output_layer = every_known_layer[example_num].back();
            double predicted_value = output_layer.a_values(j);
            double actual_value = training_data.y_coords[example_num];

            running_sum += std::pow(actual_value - predicted_value, 2);
        }
    }

    mean_squared_error = running_sum/double(data_length);
    //update files
    loss_file << epoch << "," << mean_squared_error << "\n";
    bias_file << epoch << "," << network.bias_vectors[0](0) << "\n";
    weights_file << epoch << "," << network.weight_matrices[0](0, 0) << "\n";

    //Update weights and biases
    //TODO this only works for one neuron currently
    for (int j=0; j<network.neurons_per_layer[1]; j++) {
        neuronCache activations = get_neuron_activations(1, j, every_known_layer);

        std::vector<std::vector<double>> previous_layer_activations;
        for(int k=0; k<data_length; k++) {
            previous_layer_activations.push_back(std::vector<double> {training_data.x_coords[k]});
        }

        std::vector<double> gradient_weights = find_gradient_weights(
            activations.a_values, 
            activations.z_values, 
            training_data.y_coords,
            previous_layer_activations //TODO replace with a vector of vector<d> thats example_num[activation]
        );

        for(int k=0; k<network.neurons_per_layer[0]; k++){
            network.weight_matrices[0](j, k) = network.weight_matrices[0](j, k) - learning_rate*gradient_weights[k];
        }


        std::vector<double> gradient_biases = find_gradient_biases(
            activations.a_values, 
            activations.z_values, 
            training_data.y_coords,
            previous_layer_activations //TODO replace with a vector of vector<d> thats example_num[activation]
        );

        for(int k=0; k<network.neurons_per_layer[1]; k++) {
            if (epoch % 100 == 0) {
                std::cout << "Bias Vector: " << network.bias_vectors[0] << " Gradient: " << gradient_biases[k] << "\n";
            }
            network.bias_vectors[0](j, k) = network.bias_vectors[0](j, k) - learning_rate*gradient_biases[k];
            if (std::abs(gradient_biases[k]) < gradient_threshold){
                std::cout << "Gradient threshold reached \n";
                goto output;
            }
        }
    }


    
    //show what we have currently
    if (epoch % 100 == 0) {
        std::cout << "MSE: " << mean_squared_error << " Learning rate: " << learning_rate << "\n";
    }

    learning_rate = initial_lr * (1.0 - (double(epoch) / double(num_epochs)));
}

    output:
    //cout our final network.
    for (MatrixXd W:network.weight_matrices) {
        std::cout << "W: " << W << "\n\n";
    }
    for (VectorXd b:network.bias_vectors) {
        std::cout << "b: " << b << "\n\n";
    }

    std::cout << "MSE: " << mean_squared_error << "\n";
}

int main() {
    trainingExamples training_data;
    int data_length = 4;

    //Replace with your own training data to regress to
    training_data.x_coords = {1, 2, 3, 4};
    training_data.y_coords = {2, 5, 6, 8}; //example here is y = 2x
    run(training_data, data_length);

    return 0;
}

/*
for debugging
g++ $(Get-ChildItem -Recurse -Filter *.cpp src | ForEach-Object { $_.FullName }) -I include -I "C:\Users\abuch\Downloads\eigen-5.0.0"  -o ML.exe

with optimizations:
g++ -O3 $(Get-ChildItem -Recurse -Filter *.cpp src | ForEach-Object { $_.FullName }) -I include -I "C:\Users\abuch\Downloads\eigen-5.0.0"  -o ML.exe
*/
