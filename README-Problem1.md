# GAN
A project to demonstrate how Generative Adversarial Networks (GAN) work.

Problem-1:

In this problem, we are going to learn how the GAN works. So, first of all, we explain
and discuss GAN.
One neural network, called the generator, generates new data instances, while the other,
the discriminator, evaluates them for authenticity; i.e. the discriminator decides whether
each instance of data that it reviews belongs to the actual training dataset or not.
Assume, we are going to generate hand-written numerals like those found in the MNIST
dataset, which is taken from the real world. The goal of the discriminator, when shown an
instance from the true MNIST dataset, is to recognize those that are authentic.
Meanwhile, the generator is creating new, synthetic images that it passes to the
discriminator. It does so in the hopes that they, too, will be deemed authentic, even though
they are fake. The goal of the generator is to generate passable hand-written digits: to lie
without being caught. The goal of the discriminator is to identify images coming from the
generator as fake.

GAN takes some steps in its algorithm:
  The generator takes in random numbers and returns an image.
  This generated image is fed into the discriminator alongside a stream of images taken from the actual, ground-truth dataset.
  The discriminator takes in both real and fake images and returns probabilities, a number between 0 and 1, with 1 representing a prediction of authenticity and 0 representing fake.

The discriminator is in a feedback loop with the ground truth of the images, which we know.
The generator is in a feedback loop with the discriminator. We have a double feedback loop in GAN (Fig. 1):

As you can see in fig. 1:
  The discriminator is in a feedback loop with the ground truth of the images, which we know.
  The generator is in a feedback loop with the discriminator.


For MNIST, the discriminator network is a standard convolutional network that can
categorize the images fed to it, a binomial classifier labeling images as real or fake. The
generator is an inverse convolutional network, in a sense: While a standard convolutional
classifier takes an image and down-samples it to produce a probability, the generator takes
a vector of random noise and up-samples it to an image. The first throws away data through
down-sampling techniques like max-pooling, and the second generates new data (See fig.2).
Both nets are trying to optimize a different and opposing objective function, or loss
function, in a mini-max game. As the discriminator changes its behavior, so does the
generator, and vice versa. Their losses push against each other.

Mathematically modeling a GAN:
A neural network G(z, θ₁) is used to model the Generator mentioned above. Its role is
mapping input noise variables z to the desired data space x (say images). Conversely, a
second neural network D(x, θ₂) models the discriminator and outputs the probability that the
data came from the real dataset, in the range (0,1). In both cases, θᵢ represents the weights
or parameters that define each neural network.
As a result, the Discriminator is trained to correctly classify the input data as either real
or fake. This means it’s weights are updated as to maximize the probability that any real data
input x is classified as belonging to the real dataset, while minimizing the probability that
any fake image is classified as belonging to the real dataset. In more technical terms, the
loss/error function used maximizes the function D(x), and it also minimizes D(G(z)).
Furthermore, the Generator is trained to fool the Discriminator by generating data as
realistic as possible, which means that the Generator’s weights are optimized to maximize
the probability that any fake image is classified as belonging to the real dataset. Formally
this means that the loss/error function used for this network maximizes D(G(z)).
After several steps of training, if the Generator and Discriminator have enough capacity
(if the networks can approximate the objective functions), they will reach a point at which
both cannot improve anymore. At this point, the generator generates realistic synthetic data,
and the discriminator is unable to differentiate between the two types of input.
Since during training both the Discriminator and Generator are trying to optimize
opposite loss functions, they can be thought of two agents playing a minimax game with
value function V(G, D). In this minimax game, the generator is trying to maximize its
probability of having its outputs recognized as real, while the discriminator is trying to
minimize this same value

Training a GAN:
Since both the generator and discriminator are being modeled with neural, networks, a
gradient-based optimization algorithm can be used to train the GAN. In our coding example
we’ll be using stochastic gradient descent, as it has proven to be successful in multiple fields.
Fig. 3 shows an algorithm to train a GAN.

Some critical points in training a GAN:
When you train the discriminator, hold the generator values constant; and when you train
the generator, hold the discriminator constant. Each should train against a static adversary.
For example, this gives the generator a better read on the gradient it must learn by. By the
same token, pre-training the discriminator against MNIST before you start training the
generator will establish a clearer gradient.
Each side of the GAN can overpower the other. If the discriminator is too good, it will
return values so close to 0 or 1 that the generator will struggle to read the gradient. If the
generator is too good, it will persistently exploit weaknesses in the discriminator that lead
to false negatives. This may be mitigated by the nets’ respective learning rates. The two
neural networks must have a similar “skill level”.

Part. A (Activation function)
We use Leaky-RELU function as activation function in hidden layers of our model. The
activation functions like Sigmoid or Tanh have some significant problem which are
mentioned in below.
Sigmoid:
 Vanishing gradient problem.
 Its output isn’t zero centered. It makes the gradient updates go too far in different
directions. 0 < output < 1, and it makes optimization harder.
 Sigmoid saturate and kill gradients.
 Sigmoid have slow convergence.
Tanh:
Its output is zero centered because its range in between -1 to 1 (i.e -1 < output < 1). Hence
optimization is easier in this method hence in practice it is always preferred over Sigmoid
function. But still it suffers from Vanishing gradient problem.
RELU:
It has become very popular in the past couple of years. It was recently proved that it had 6
times improvement in convergence from Tanh function. It’s just R(x) = max(0,x) i.e if x <
0 , R(x) = 0 and if x >= 0 , R(x) = x. Hence as seeing the mathematical form of this function
we can see that it is very simple and efficient. A lot of times in Machine learning and
computer science we notice that most simple and consistent techniques and methods are
only preferred and are best. Hence it avoids and rectifies vanishing gradient problem.
Almost all deep learning Models use RELU nowadays.
The problem with RELU is that some gradients can be fragile during training and can die. It
can cause a weight update which will makes it never activate on any data point again. Simply
saying that RELU could result in Dead Neurons.
Leaky-RELU:
To fix the aforementioned problem with RELU, another modification was introduced
called Leaky-RELU to fix the problem of dying neurons. It introduces a small slope to keep
the updates alive (Fig. 4).

Conclusion:
Nowadays, we should use RELU which should only be applied to the hidden layers. And
if your model suffers from dead neurons during training we should use Leaky-RELU.
It’s just that Sigmoid and Tanh should not be used nowadays due to the vanishing Gradient
Problem which causes a lots of problems to train, degrades the accuracy and performance
of a deep Neural Network Model.

Part. B (Generating Noise)
In this part, we are going to explain how we generate a proper noise to apply on generator
as an input. We apply a uniform noise using random numbers between -1, and 1. Generally,
we send a white noise to generator while we are using a uniform noise as input. Therefore,
we give a same chance to all members of our feature space to make a number as generator’s
output. In other word, by applying uniform noise all images, or numbers have a same chance
to be produced as output of generator. Whereas, if we use a normal random noise as input
of generator, we give more chance to some specific images, or numbers which their features
are near to mean of the noise


Part. C (Optimizer)
We use Adam as optimizer in our model, because this optimizer have some features which
make our model more accurate and efficient. In the sequel, we describe these
aforementioned features.
Adam:
Adaptive Moment Estimation (Adam) is another method that computes adaptive learning
rates for each parameter. In addition to storing an exponentially decaying average of past
squared gradients vt like Adadelta and RMSprop, Adam also keeps an exponentially
decaying average of past gradients mt, similar to momentum. Whereas momentum can be
seen as a ball running down a slope, Adam behaves like a heavy ball with friction, which
thus prefers flat minima in the error surface. We compute the decaying averages of past and
past squared gradients mt and vt respectively as follows:
𝑚𝑡 = 𝛽1𝑚𝑡−1 + (1 − 𝛽1)𝑔𝑡
𝑣𝑡 = 𝛽2𝑣𝑡−1 + (1 − 𝛽2)𝑔𝑡 2
The mt and vt are estimates of the first moment (the mean) and the second moment (the
uncentered variance) of the gradients respectively, hence the name of the method.
As mt and vt are initialized as vectors of O's, the authors of Adam observe that they are
biased towards zero, especially during the initial time steps, and especially when the decay
rates are small (i.e. β1 and β2 are close to 1).
They counteract these biases by computing bias-corrected first and second moment
estimates.

They then use these to update the parameters just as we have seen in Adadelta and
RMSprop, which yields the Adam update rule.

The authors propose default values of 0.9 for β1, 0.999 for β2, and 10-8 for ϵ. They show
empirically that Adam works well in practice and compares favorably to other adaptive
learning-method algorithms.

Part. D (Batch normalization)
Batch normalization is a technique that normalizes the feature vectors to have no mean
or unit variance. It is used to stabilize learning and to deal with poor weight initialization
problems. It is a pre-processing step that we apply to the hidden layers of the network and it
helps us to reduce internal covariate shift.
The benefits of batch normalization are as follows:
Reduces the internal covariate shift: Batch normalization helps us to reduce the internal
covariate shift by normalizing values.
Faster training: Networks will be trained faster if the values are sampled from a
normal/Gaussian distribution. Batch normalization helps to whiten the values to the internal
layers of our network. The overall training is faster, but each iteration slows down due to
the fact that extra calculations are involved.
Higher accuracy: Batch normalization provides better accuracy.
Higher learning rate: Generally, when we train neural networks, we use a lower learning
rate, which takes a long time to converge the network. With batch normalization, we can use
higher learning rates, making our network reach the global minimum faster.
Reduces the need for dropout: When we use dropout, we compromise some of the essential
information in the internal layers of the network. Batch normalization acts as a regulator,
meaning we can train the network without a dropout layer.

Part. E (Results)
Implementation and Results:
We implement a generative adversarial network (GAN) for generating new numbers using
MNIST data set. You can see the topology of generator and discriminator part of GAN in
fig. 5 and fig. 6, respectively. Furthermore, we served the binary-crossentropy as loss
function of the discriminator.

Our generator and discriminator model are made using multi-layer perceptron. We train our
model in 30000 epochs and we considered the batch-size equal to 120.
 Number of epochs = 30000
 Batch size = 120
Fig. 7 shows the growth of numbers’ quality through time. The first output of GAN is image
on the top-left and the last one is on the bottom-right. It is obvious that if we serve the CNN
to modeling the generator and discriminator, we will have better results, because CNN is
powerful in feature extracting and breaking correlations




