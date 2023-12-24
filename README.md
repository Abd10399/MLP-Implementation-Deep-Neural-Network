# MLP-Implementation-Deep-Neural-Network


## Basic Outline
This project was focused on designing an MLP and a CNN and training them on 2 different image datasets, namely
Fashion MNIST and CIFAR-10. Experiments were ran on both different models for tuning the hyper parameters
which led to us using a learning rate of 0.001 for the MLP and a learning rate of 0.01 for the CNN to optimize
computational time and see good convergence. Mini batch sizes of 256 were used throughout for all the experiments
which is a normal batch size and widely used. Gradients were optimized using Stochastic Gradient Descent, and
viable convergence was shown. Overall, the best performing MLP was the one having weights initialized through
the ”Kaiming” distribution, having 2 hidden layers (128 neurons in each layer), ReLu as an activation function,
weak regularization and a batch size of 256. Each MLP model was trained for 300 epochs which was deemed
sufficient due to the time constraints and computational resources at hand. Regardless, however, CNNs would
continuously outperform the MLPs for both datasets, specifically CIFAR 10, and therefore could be considered
the optimal way for training a model for these datasets. 

## Results and Conclusion
Two primary datasets, Fashion-MNIST and CIFAR10, were explored with vectorization and normalization processes
for neural network applications. During weight initialization experiments, the Kaiming distribution emerged
as the most effective, producing the best accuracy. In-depth model assessments showed that networks with more
hidden layers learned faster and exhibited less overfitting, while the ReLU activation function outperformed the
Sigmoid. Regularization techniques revealed that L1 needed smaller lambda values for convergence, whereas L2
required more epochs. Convolutional Neural Networks (CNNs) significantly surpassed Multi-Layer Perceptrons
(MLPs) in accuracy on both datasets, and experiments with various optimizers highlighted the efficacy of momentum
in Stochastic Gradient Descent.