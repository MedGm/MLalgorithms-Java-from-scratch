package com.medgm.LinearRegression;

/* To start with , this will be the simplest ML algorithm to implement using Java.
   It is a univariate linear regression model, which means it has one input feature 
   and one output variable. The goal is to find the best-fitting line through the data 
   points using gradient descent.

   The dataset is simply a list of pairs of input (x) and output (y) values. The model will 
   learn the parameters (weights like w0 and w1) that minimize the cost function, which is the mean 
   squared error between the predicted and actual output values.

   Note: This is just a basic implementation for educational purposes.
 */



public class LinearRegressionUniVar {

    // This is our Step 1: Defining the dataset , which is a simple linear relationship , y = 10x + 2
    private double[] x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    private double[] y = {12, 22, 32, 42, 52, 62, 72, 82, 92, 102};

    // Initializing weights (w0 and w1) to zero
    private double w0 = 0.0;
    private double w1 = 0.0;

    // This is Step 2 : Predicting the output using the linear equation y = w0 + w1 * x 
    //(it will be used in training and cost computation)
    public double predict(double X) {
        return w0 + w1 * X;
    }

    // Step 3 : Computing Cost function (Mean Squared Error, which measures how well the model fits the data)
    public double computeCost() {
        double cost = 0.0;
        int samples = x.length;

        for (int i = 0; i < samples; i++) {
            double prediction = predict(x[i]);
            cost += Math.pow(prediction - y[i], 2);
        }

        return cost / (2 * samples);
    }

    // Step 4 : Training the model using Gradient Descent 
    // For each iteration, compute the partial derivatives of the cost function:
    // dJ/dw0 and dJ/dw1
    // Update the weights using the learning rate and the partial derivatives

    // This will be the hardest part to explain, but it is the core of the algorithm.
    public void train(double learningRate, int numIterations) {
        // first we need to get the number of samples
        int m = x.length;

        // Now we will iterate for a number of iterations to update the weights
        for (int iter = 0; iter < numIterations; iter++) {
            double dJ_dw0 = 0.0;
            double dJ_dw1 = 0.0;

            for (int i = 0; i < m; i++) {
                double prediction = predict(x[i]); // prediction = w0 + w1 * x[i]
                double error = prediction - y[i]; // error = h(x[i]) - y[i]

                // Compute the gradients
                // dJ/dw0 = Σ(error)
                // dJ/dw1 = Σ(error * x[i])
                dJ_dw0 += error;
                dJ_dw1 += error * x[i];
            }
            
            // Average the gradients
            dJ_dw0 /= m;
            dJ_dw1 /= m;

            // Update the weights using the gradients and learning rate
            w0 -= learningRate * dJ_dw0;
            w1 -= learningRate * dJ_dw1;

            if (iter % 100 == 0) {
                System.out.printf("Iteration %d, Cost: %.6f%n", iter, computeCost());
                // Each 100 iterations, print the cost to monitor progress
            }
        }
    }

    // Finally Step 5 : Evaluating the model
    // This will print the final model parameters and the cost
    public void evaluate() {
        System.out.printf("Final model: y = %.4f + %.4fx%n", w0, w1);
        System.out.printf("Final cost: %.6f%n", computeCost());
    }

    // we will test the model with 0.01 learning rate and 1000 iterations
    public static void main(String[] args) {
        LinearRegressionUniVar model = new LinearRegressionUniVar();

        double learningRate = 0.01;
        int numIterations = 1000;

        model.train(learningRate, numIterations);
        model.evaluate();

        // Now we can make a prediction with the trained model
        // For example, predict the output for x = 11.0
        double newX = 11.0;
        double predictedY = model.predict(newX);
        System.out.printf("Prediction for x = %.1f → y = %.4f%n", newX, predictedY);
    }
}