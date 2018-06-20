# Databricks notebook source
# MAGIC %md # Introduction to Deep Learning Frameworks
# MAGIC 
# MAGIC In this notebook, we're going to experiment with image classification using a variant of logistic regression.
# MAGIC 
# MAGIC We're not going to do any deep learning quite yet; we're going to use this as an opportunity to become familiar with concepts found across deep learning frameworks. We'll cover these things:
# MAGIC 
# MAGIC * [Acquiring Training Data](#acquire)
# MAGIC * [Configuring the model training process](#configure)
# MAGIC * [Defining the model](#define)
# MAGIC * [Training the model](#train)
# MAGIC * [Hyperparameter Tuning](#tune)
# MAGIC 
# MAGIC First, we need to get a few Python package imports out of the way (as well as a *%matplotlib inline* to allow us to display graphs/plots inline within our notebook).

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

import tensorflow as tf
import numpy as np
import pandas
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# COMMAND ----------

# MAGIC %md ## Acquiring Training Data
# MAGIC <a name="acquire"> </a>
# MAGIC 
# MAGIC The first thing we need to do is acquire training data. In this notebook, we're going to be working with the [MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/). The MNIST dataset contains 60,000 examples of handwritten digits (0-9) and 10,000 seperate test examples.
# MAGIC 
# MAGIC <img style="float: left;" src="https://avanadeaibootcamp.blob.core.windows.net/images/mnist.png"> 

# COMMAND ----------

# MAGIC %md TensorFlow's tutorials package contains helper functions to download and ingest the MNIST dataset. Run the following cell to download the MNIST dataset to your Azure Notebook environment.

# COMMAND ----------

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, validation_size=12000)

# COMMAND ----------

# MAGIC %md ### Visualizing a single training example
# MAGIC 
# MAGIC Let's take a look at what a single training example looks like. A single training example contains two parts - a 28 x 28 grayscale input image, and a label. We'll use the **next_batch()** function to get a single random training example.

# COMMAND ----------

sample_x, sample_y = mnist.train.next_batch(1)

# COMMAND ----------

print("Image data shape:  {}".format(sample_x.shape))
print("Image label shape: {}".format(sample_y.shape))

# COMMAND ----------

# MAGIC %md The shape of the input image data is pretty straightforward - we have a single 28 x 28 image - the first component of the shape (1, 784), flattened into a 784 dimensional vector - the second component of the shape (1, 784).
# MAGIC 
# MAGIC But the image label is less obvious - why is a single label a 10-dimensional vector? To understand, let's take a look at the label:

# COMMAND ----------

print(sample_y[0])

# COMMAND ----------

# MAGIC %md This is what is known as a **one-hot encoding**. We're classifying against 10 possible digits (0-9); a one-hot encoded label is a vector where all elements are 0 except for the single index that maps to the categorical value of the label - e.g.:
# MAGIC 
# MAGIC ```
# MAGIC 0:   [1 0 0 0 0 0 0 0 0 0]  
# MAGIC 1:   [0 1 0 0 0 0 0 0 0 0]  
# MAGIC 2:   [0 0 1 0 0 0 0 0 0 0]  
# MAGIC 3:   [0 0 0 1 0 0 0 0 0 0]  
# MAGIC 4:   [0 0 0 0 1 0 0 0 0 0]  
# MAGIC 5:   [0 0 0 0 0 1 0 0 0 0]  
# MAGIC 6:   [0 0 0 0 0 0 1 0 0 0]  
# MAGIC 7:   [0 0 0 0 0 0 0 1 0 0]  
# MAGIC 8:   [0 0 0 0 0 0 0 0 1 0]  
# MAGIC 9:   [0 0 0 0 0 0 0 0 0 1]
# MAGIC ```
# MAGIC 
# MAGIC The sample's label can be interpreted like this:

# COMMAND ----------

print('Label: {}'.format(sample_y[0]))
print('')

_, index = np.where(sample_y == 1)

print("Corresponds to digit: {}".format(index[0]))

# COMMAND ----------

# MAGIC %md Now that we've examined a label, let's take a quick look at the image data for a single training example. Below is a small function that reshapes the 784-dimensional flat image vector into a 28x28 image and displays it inline within your notebook.

# COMMAND ----------

def show_image(input):
    image = (np.reshape(input, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(image, interpolation='nearest', cmap="Greys")
    plt.show()

# COMMAND ----------

show_image(sample_x)

# COMMAND ----------

# MAGIC %md Try this with a few other sample images. To do so, scroll back up to this cell:
# MAGIC 
# MAGIC ```sample_x, sample_y = mnist.train.next_batch(1)```
# MAGIC 
# MAGIC Re-run that cell, then run each subsequent cell back down to this point to see the output.

# COMMAND ----------

# MAGIC %md ## Configuring the model training process
# MAGIC <a name="configure"> </a>
# MAGIC 
# MAGIC Next, we'll configure ***hyperparameters*** for our model. These hyperparameters are levers & knobs we can use to control the training process, as we'll see shortly. For our model, the important hyperparameters we're working with are:
# MAGIC 
# MAGIC * **Learning rate** - determines how fast weights (in case of a neural network) or the cooefficents (in case of linear regression or logistic regression) change
# MAGIC * **Training iterations** - how long we train the model for
# MAGIC * **Batch size** - how many examples are included in each minibatch

# COMMAND ----------

hyperparameters = {
    #############################
    # Hyperparameters for model
    #############################
    'learning_rate': 0.01,
    'training_iters': 100000,
    'batch_size': 100,

    #############################
    # Input data configuration
    #############################
    'n_pixels': 784,       # MNIST data input (img shape: 28*28)
    'n_classes': 10,       # MNIST total classes (0-9 digits)

    #############################
    # Debug verbosity
    #############################
    'display_step': 10
}

# COMMAND ----------

# MAGIC %md ## Defining the model
# MAGIC <a name="define"> </a>
# MAGIC 
# MAGIC Now, we'll define our **model graph**.
# MAGIC 
# MAGIC ### Inputs to model graph
# MAGIC 
# MAGIC First, we need *entry points* for data flowing into our model. In TensorFlow, these entry points are defined with *tensor placeholders*. These represent tensors that are fed at model evaluation time with input data (e.g. training data). We'll define shapes for our input tensors which correspond to the shapes of our MNIST data - both the shape of the input image data, and the shape of the target image label.

# COMMAND ----------

x = tf.placeholder(tf.float32, [None, hyperparameters['n_pixels']])
y = tf.placeholder(tf.float32, [None, hyperparameters['n_classes']])

# COMMAND ----------

# MAGIC %md You might ask yourself - what does **None** mean in this context? In this example, **None** is itself a placeholder for how many training examples exist within a single batch. Using **None** lets us dynamically control how many training examples flow into the model at training and at inference time.
# MAGIC 
# MAGIC You can see the placeholder nature of **None** if you print the shape of ***x*** and ***y***:

# COMMAND ----------

print(x.shape)
print(y.shape)

# COMMAND ----------

# MAGIC %md ### Model weights
# MAGIC 
# MAGIC The next thing we need to do is define TensorFlow **variable tensors** for our model's coefficients. If this were a neural network, these variables might represent the *weights* for a given layer. For sake of convienence, we'll refer to them as weights (e.g. W).
# MAGIC 
# MAGIC Notes regarding the shape of the variables:  
# MAGIC 1. They are explicitly defined, and importantly, there are no placeholder numbers for batch size. This is because these variables contain tensors that represent a single instance of the model.
# MAGIC 2. We have two variables - W (weights) and b (biases). Note for both the output is framed in terms of number of output classes. 

# COMMAND ----------

W = tf.Variable(tf.zeros([hyperparameters['n_pixels'], hyperparameters['n_classes']]))
b = tf.Variable(tf.zeros([hyperparameters['n_classes']]))

# COMMAND ----------

# MAGIC %md ### Model graph definition
# MAGIC 
# MAGIC Then, we'll define the model itself. Note the flow here - training data enters the model through the **x** placeholder tensor; image vectors (**x**) are multiplied against the **W** weight matrix (our **W** variable tensor), which **b**, our bias vector, is added to. The result of this is passed through the TensorFlow softmax operation (which can be thought of as a node within our model graph). The result of the softmax operation is return as the output of the model graph.

# COMMAND ----------

# Create model
def model(x, W, b):
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    return pred

# COMMAND ----------

# MAGIC %md Visually, this is what our model is doing:

# COMMAND ----------

# MAGIC %md <div>
# MAGIC <img style="float: left; width: 600px" src="https://avanadeaibootcamp.blob.core.windows.net/images/logistic_reg_vs_softmax_reg.png">
# MAGIC </div>
# MAGIC <div style="clear: both;"></div>
# MAGIC <br/>
# MAGIC From [What is Softmax Regression and How is it Related to Logistic Regression?](https://www.kdnuggets.com/2016/07/softmax-regression-related-logistic-regression.html)

# COMMAND ----------

# MAGIC %md ## Training the model
# MAGIC <a name="train"> </a>
# MAGIC 
# MAGIC ### Preparing to train the model
# MAGIC 
# MAGIC We're almost ready to train the model. Before we do, we need to "inflate" or construct our graph. Importantly, no data will actually flow through the graph yet - data does not flow within a TensorFlow graph until it is run within the context of a TensorFlow ***session***. But to run a model graph in a session, first, we need to construct the graph.
# MAGIC 
# MAGIC We also need to define a few critical nodes in our graph:
# MAGIC 
# MAGIC * **Cost** - this *operation* node is a function, evaluated *every* training iteration, that calculates the cost (loss/error) for the network given the current minibatch's training examples as the input to the network, evaluated against the current minibatch's training targets.
# MAGIC * **Optimizer** - this node is the TensorFlow optimizer that implements the gradient descent algorithm that is used during training to update the weights of the model. Note that we're effectively implementing ***minibatch stochastic*** gradient descent by virtue of our incremental approach to updating the weights (minibatch by minibatch of training examples).
# MAGIC * **Accuracy** - this *operation* node is a function, evaluated periodically during training, that calculates accuracy for the ***current*** minibatch.

# COMMAND ----------

def construct_training_graph(hyperparameters):
    # Construct model graph
    pred = model(x, W, b)

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

    # Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(hyperparameters['learning_rate']).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    return {'pred': pred,
            'cost': cost,
            'optimizer': optimizer,
            'accuracy': accuracy,
            'init': init}

training_graph = construct_training_graph(hyperparameters)

# COMMAND ----------

# MAGIC %md ### TensorFlow Session and Model Training
# MAGIC 
# MAGIC We're now ready to create a TensorFlow [session](https://www.tensorflow.org/programmers_guide/graphs) and run our SGD training algorithm. Each iteration, we:
# MAGIC 
# MAGIC - Extract **batch_size** training examples (**batch_x**, **batch_y**)
# MAGIC - Run a single step of gradient descent optimization, feeding the batch into the graph via **feed_dict**.
# MAGIC - Periodically, we'll calculate and print metrics for the current batch.
# MAGIC 
# MAGIC Once we've iterated **training_iter** steps, we print a set of final metrics for a select number of images from the seperate MNIST database test set.
# MAGIC 
# MAGIC Now, run the cell below to train your model.

# COMMAND ----------

def train(training_graph, hyperparameters, verbose=True):
    # Training metrics
    train_costs = []
    valid_costs = []
    train_accs = []    
    valid_accs = []
    
    # Nodes from training graph
    init = training_graph['init']
    cost = training_graph['cost']
    accuracy = training_graph['accuracy']
    optimizer = training_graph['optimizer']

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1

        while step * hyperparameters['batch_size'] < hyperparameters['training_iters']:
            x_train, y_train = mnist.train.next_batch(hyperparameters['batch_size'])

            # Run single step of gradient descent optimization
            sess.run(optimizer, feed_dict={x: x_train, y: y_train})

            # Periodically calculate current batch loss and accuracy
            if step % hyperparameters['display_step'] == 0:
                # Calculate training loss, accuracy
                cost_train, acc_train = sess.run([cost, accuracy], feed_dict={x: x_train, y: y_train}) 
                train_costs.append(cost_train)
                train_accs.append(acc_train)

                # Calculate validation loss, accuracy
                x_valid, y_valid = mnist.validation.next_batch(hyperparameters['batch_size'])
                cost_valid, acc_valid = sess.run([cost, accuracy], feed_dict={x: x_valid, y: y_valid})         
                valid_costs.append(cost_valid)
                valid_accs.append(acc_valid)

                if (verbose):
                    print("Iter " + str(step * hyperparameters['batch_size']) + "        Train Cost: " + \
                          "{:.5f}".format(cost_train) + "    Accuracy: " + \
                          "{:.2f}".format(acc_train * 100.0) + "%        Validation Cost: " + \
                          "{:.5f}".format(cost_valid) + "    Accuracy: " + \
                          "{:.2f}".format(acc_train * 100.0) + "%")

            step += 1

        print('')
        print("Optimization Finished!")
        print('')

        # Calculate accuracy for withheld test image set
        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Testing Accuracy: " + "{:.2f}".format(acc_test * 100.0) + "%")
        
    return train_costs, valid_costs

hyperparameters['learning_rate'] = 0.01
training_graph = construct_training_graph(hyperparameters)

train_costs, valid_costs = train(training_graph, hyperparameters)

# COMMAND ----------

# MAGIC %md ## Hyperparameter Tuning
# MAGIC <a name="tune"> </a>
# MAGIC 
# MAGIC Congratulations, at this point, you've trained a working model with TensorFlow. How did your model do? You're likely in the 86-88% accuracy range on the withheld test set.
# MAGIC 
# MAGIC The question now is - can we do any better? What are the levers and switches we can play with to improve the final accuracy of the model on the test data?
# MAGIC 
# MAGIC Let's take a look at the end-to-end process of training - specifically, the **cost over time** on the training and validation sets. In particular, we're interested how the **learning rate** we defined much earlier impacts the process of training.

# COMMAND ----------

def plot_costs(train_costs, valid_costs):
    fig, ax = plt.subplots()
    fig_size = [12, 9]
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(train_costs, label='Training Cost')
    plt.plot(valid_costs, label='Validation Cost')
    plt.title('Cost over time during training')
    legend = ax.legend(loc='upper right')

# Plot the training and validation costs over time of the model we just trained
plot_costs(train_costs, valid_costs)

# COMMAND ----------

# MAGIC %md  

# COMMAND ----------

# MAGIC %md What if we try a different learning rate? Let's try a much smaller learning rate - 0.00000001:

# COMMAND ----------

hyperparameters['learning_rate'] = 0.00000001
training_graph = construct_training_graph(hyperparameters)

train_costs, valid_costs = train(training_graph, hyperparameters, verbose=False)
plot_costs(train_costs, valid_costs)

# COMMAND ----------

# MAGIC %md Not particularly great. Why is performance worse in this case?
# MAGIC 
# MAGIC The intuition here is: because the learning rate impacts how fast weights are updated, a much smaller learning rate means the rate of learning itself is much slower. You can see that in the slopes of the training/validation curves - a steeper curve - a quicker reduction in cost - roughly maps to faster learning. We might be able to get back up to that 87% range if we trained longer, but that isn't the ideal solution here.
# MAGIC 
# MAGIC What if we try a much higher learning rate? Let's try a learning rate of 1.0:

# COMMAND ----------

# MAGIC %md  

# COMMAND ----------

hyperparameters['learning_rate'] = 1.0
training_graph = construct_training_graph(hyperparameters)

train_costs, valid_costs = train(training_graph, hyperparameters, verbose=False)
plot_costs(train_costs, valid_costs)

# COMMAND ----------

# MAGIC %md The end result here is better - you're likely hitting accuracy around the 90-91% range. Learning proceeds much faster (steep drop in cost early in training) - but notice how the cost is a bit less stable than our original learning rate (of 0.1).
# MAGIC 
# MAGIC Let's go even faster now - let's double the learning rate to 2.0:

# COMMAND ----------

hyperparameters['learning_rate'] = 2.0
training_graph = construct_training_graph(hyperparameters)

train_costs, valid_costs = train(training_graph, hyperparameters, verbose=False)
plot_costs(train_costs, valid_costs)

# COMMAND ----------

# MAGIC %md Hmm - it likely didn't help as much as you might have hoped - and the magnitude of the instability is a bit higher than before. What if we go even faster, with an even higher learning rate - 3.0:

# COMMAND ----------

hyperparameters['learning_rate'] = 3.0
training_graph = construct_training_graph(hyperparameters)

train_costs, valid_costs = train(training_graph, hyperparameters, verbose=False)
plot_costs(train_costs, valid_costs)

# COMMAND ----------

# MAGIC %md The accuracy has most likely collapsed here into the < 10% range. So for this problem, it seems there is a rough upper bound somewhere between 2.0 and 3.0.
# MAGIC 
# MAGIC We'll try one more learning rate - 0.1. This is a bit closer to the sweet spot we want to be in for this particular problem, and you should most likely get accuracy in the 90-91% range:

# COMMAND ----------

hyperparameters['learning_rate'] = 0.1
training_graph = construct_training_graph(hyperparameters)

train_costs, valid_costs = train(training_graph, hyperparameters, verbose=False)
plot_costs(train_costs, valid_costs)

# COMMAND ----------

# MAGIC %md ## Conclusion

# COMMAND ----------

# MAGIC %md Congratulations, you've completed the Introduction to Deep Learning Frameworks, having covered these concepts:
# MAGIC 
# MAGIC * [Acquiring Training Data](#acquire)
# MAGIC * [Configuring the model training process](#configure)
# MAGIC * [Defining the model](#define)
# MAGIC * [Training the model](#train)
# MAGIC * [Hyperparameter Tuning](#tune)

# COMMAND ----------

