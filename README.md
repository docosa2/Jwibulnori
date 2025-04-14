# Jwibulnori
C# AI Framework

Introduction
Jwibulnori(쥐볼놀이) is a C#-based AI framework with support for tensor operations, neural network modules, and CUDA integration. It is designed for researchers, developers, and students alike to make it easy to build and train deep learning models in the C# environment. With its intuitive API and GPU-accelerated high-performance computation, Jwibulnori enables users to conduct deep learning experiments efficiently.
Features
Multi-dimensional tensor operations with autograd – Support for N-dimensional tensors and automatic differentiation for gradient computation.
Modular neural network components – Layers, activation functions, and other building blocks provided as modules for easy model construction.
Built-in optimizers – Includes various optimization algorithms like SGD, Adam for training models.
GPU acceleration via CUDA – Ability to run tensor computations on the GPU for improved performance.
Simple and intuitive API – Designed with a clean API for ease of use by beginners and experts alike.
Installation
Jwibulnori is distributed as a NuGet package. Install it via the NuGet Package Manager Console or the .NET CLI:
powershell
복사
편집
PM> Install-Package Jwibulnori
or
bash
복사
편집
$ dotnet add package Jwibulnori
Usage Example
Below is a simple example that adds two 2x2 tensors:

using Jwibulnori;

Tensor A = new Tensor(new float[,] { {1, 2}, {3, 4} });
Tensor B = Tensor.Ones(2, 2);
Tensor C = A + B;
Console.WriteLine(C);  // Output: [[2, 3], [4, 5]]
License
This project is licensed under the MIT License.
Framework Architecture


The diagram above illustrates the layered architecture of the Jwibulnori framework. The framework is organized in a hierarchical design where each layer is responsible for a specific function. At the base is the Tensor Core, which handles tensor operations and underpins the rest of the system. Above it are the Neural Modules (e.g. layers and activation functions) and the Optimizer layer, which together enable the construction and training of neural network models. Finally, the Interop Manager interfaces with the Tensor Core to manage external libraries like CUDA, thereby supporting GPU-accelerated computations.
Tensor Core – Core module for multi-dimensional tensor data structures and operations, handling all numerical computation and automatic differentiation of gradients.
Neural Modules – Modular neural network components (layers, activation functions, etc.) provided on top of the Tensor Core to build models.
Optimizer – Component that performs parameter updates during training using built-in algorithms such as SGD and Adam.
Interop Manager – Manages integration with low-level libraries (e.g. CUDA) so that Tensor Core operations can execute on the GPU.
NuGet Package Description
Jwibulnori is an open-source deep learning framework for .NET/C# developers. It provides tensor-based numerical computation, modular neural network building blocks, and GPU acceleration to make it simple to construct and train deep learning models. Features:
Simple and intuitive API
N-dimensional tensor operations with autograd support
Variety of built-in neural network layers and activation functions
Optimizers such as SGD and Adam included
GPU acceleration via CUDA
Example Usage:

using Jwibulnori;

Tensor A = new Tensor(new float[,] { {1, 2}, {3, 4} });
Tensor B = Tensor.Ones(2, 2);
Tensor C = A + B;
Console.WriteLine(C);
User Documentation and Examples
Tensor Creation and Operations
The Tensor class in Jwibulnori represents an N-dimensional array and supports a variety of numerical operations. The example below creates a 2x3 tensor and demonstrates basic arithmetic operations (addition and multiplication) on it.

Tensor t1 = Tensor.Arange(1, 7).Reshape(2, 3); // 2x3 tensor with values 1 through 6
Tensor t2 = Tensor.Ones(2, 3) * 2;             // 2x3 tensor filled with the value 2
Tensor sum = t1 + t2;
Tensor product = t1 * t2;
Console.WriteLine(sum);     // [[3, 4, 5], [6, 7, 8]]
Console.WriteLine(product); // [[2, 4, 6], [8, 10, 12]]
In the code above, t1 is a 2x3 tensor containing values 1 to 6, and t2 is a 2x3 tensor where every element is 2. The resulting sum adds 2 to each element of t1, and product multiplies each element of t1 by 2.
Simple Model Definition & Training Loop
This example defines a simple neural network model and trains it on dummy data. We construct a small multi-layer perceptron (MLP) with an input dimension of 784, a hidden layer of 128 units, and an output dimension of 10. The model is then trained for 5 epochs using an SGD optimizer and Mean Squared Error (MSE) loss. (Random tensors are used for the inputs and targets to demonstrate the training loop.)

int inputDim = 784, hiddenDim = 128, outputDim = 10;
var model = new Sequential(
    new Dense(inputDim, hiddenDim),
    new ReLU(),
    new Dense(hiddenDim, outputDim)
);
var optimizer = new SGD(model.Parameters, lr: 0.01f);
for (int epoch = 1; epoch <= 5; epoch++) {
    // create a random batch of 64 samples
    Tensor x = Tensor.RandomNormal(64, inputDim);
    Tensor target = Tensor.RandomNormal(64, outputDim);
    Tensor output = model.Forward(x);
    Tensor loss = Loss.MSE(output, target);
    optimizer.ZeroGrad();
    loss.Backward();
    optimizer.Step();
    float lossVal = loss.ToScalar();
    Console.WriteLine($"Epoch {epoch}, Loss: {lossVal:F4}");
}
In this code, we defined a model with an inputDim × hiddenDim × outputDim architecture and then trained it using random data (x and target). The loss value is printed at each epoch to show the training progress.
MNIST Classifier Example
For the final example, we illustrate training a model to classify the MNIST handwritten digit dataset using Jwibulnori. First, it loads the MNIST training and test data (using a hypothetical Dataset.LoadMNIST() function). Then, it defines a neural network model similar to the previous example and trains it over multiple epochs using an Adam optimizer and cross-entropy loss. After each epoch, the code computes the model’s accuracy on the test set and prints it. (In a real implementation, additional steps such as data preprocessing and a more thorough evaluation would be needed, but they are omitted here for brevity.)

var (trainData, testData) = Dataset.LoadMNIST();
var model = new Sequential(
    new Dense(784, 128), new ReLU(),
    new Dense(128, 10)
);
var optimizer = new Adam(model.Parameters, lr: 0.001f);
for (int epoch = 1; epoch <= 5; epoch++) {
    foreach ((Tensor x, Tensor label) in trainData) {
        Tensor pred = model.Forward(x);
        Tensor loss = Loss.CrossEntropy(pred, label);
        optimizer.ZeroGrad();
        loss.Backward();
        optimizer.Step();
    }
    float testAcc = EvaluateAccuracy(model, testData);  // user-defined evaluation of accuracy
    Console.WriteLine($"Epoch {epoch}: Test Accuracy = {testAcc * 100:F2}%");
}
The code above demonstrates the full training loop from data loading to model training and evaluation on the MNIST dataset. After each epoch, it prints the test accuracy, allowing you to monitor the model’s performance over time. (Details for data loading and the EvaluateAccuracy implementation are simplified for this example.)

What is Jwibulnori?

Jwibulnori (쥐불놀이) is a traditional Korean folk game celebrated during the first full moon of the lunar calendar (Jeongwol Daeboreum). Participants spin burning cans filled with embers, creating beautiful spirals of fire in the night sky. Historically, this ritual symbolizes the burning away of harmful insects and weeds, wishing for prosperity and good harvests in the coming year.

Inspired by this tradition, the Jwibulnori framework metaphorically represents the illumination and deep exploration of neural network internals through low-level GPU acceleration and intuitive visualization tools. Just like spinning fire reveals mesmerizing patterns, Jwibulnori reveals the intricate details and hidden potentials within deep learning models.
