# GAN
A project to demonstrate how Generative Adversarial Networks (GAN) work.

Problem 2:

In this problem, we are going to get familiar with DCGAN networks.
Part. A (Defining a DCGAN)
Defining the network:
The core to the DCGAN architecture uses a standard CNN architecture on the discriminative
model. For the generator (Fig. 8), convolutions are replaced with up-convolutions (transpose
convolutions), so the representation at each layer of the generator is actually successively
larger, as it maps from a low-dimensional latent vector onto a high-dimensional image. To
implement a DCGAN we should follow some critical points:
 Replace any pooling layers with strided convolutions (discriminator) and fractionalstrided convolutions (generator).
 Use batch normalization in both the generator and the discriminator.
 Remove fully connected hidden layers for deeper architectures.
 Use RELU activation in generator for all layers except for the output, which uses
Tanh.
 Use Leaky-RELU activation in the discriminator for all layers.

Fig. 9 shows a DCGAN generator. This network takes in a 100*1 noise vector, denoted z,
and maps it into the G(Z) output which is 64*64*3. This architecture is especially interesting
the way the first layer expands the random noise. The network goes from 100*1 to
1024*4*4, this layer is denoted project and reshape.
We see that following this layer, classical convolutional layers are applied which reshape
the network with the (N+P - F)/S + 1 equation classically taught with convolutional layers.
In the diagram above we can see that the N parameter, (Height/Width), goes from 4 to 8 to
16 to 32, it doesn’t appear that there is any padding, the kernel filter parameter F is 5*5, and
the stride is 2 (fractional-stride = 1/2).

Part. B (Designing a model)
We design a DCGAN to apply on CIFAR-10 data set to reach some new images related to
categories of the CIFAR-10. Fig. 10, and fig. 11 show the architecture and topology of
generator and discriminator of the DCGAN. We considered the number of epochs and batchsize equal to 60, and 120, respectively. Also, we served Leaky-RELU as activation function,
binary-crossentropy as loss function, and Adam as optimizer.
In the meantime, it should be noticed that in the generator and discriminator we used
transposed convolution and convolution respectively.

Part. C (Performance of activations and loss functions)
In discriminator, at the last layer, we use Sigmoid as activation function to generate 0,
and 1 as output which the value of 1 is related to real data and 0 is related to fake data.
Whereas, at the last layer of the generator, we serve Tanh as activation function to produce
-1, and 1 in the output, considering our input noise is uniform which values are between -1
and 1. For the rest of layers, we use Leaky-RELU as activation function. We described the
Leaky-RELU before, completely (see Problem.1-Part. A).
The binary-crossentropy has been used as loss function in both generator and
discriminator parts of DCGAN. We used this loss function according to the literature.
Considering the below equation, while we are training the discriminator we label the real
data as 1 and the generated data as 0. Whereas, during training of generator the value of
function of V is considered as 1.


Part. D (Loss and Accuracy of generator and discriminator)
Fig. 12, and fig. 13 show the loss and accuracy of both generator and discriminator parts of
DCGAN. As we expected, the loss of generator is increasing and the loss of discriminator
is decreasing, because in GANs, discriminator and generator part are competing, and trying
to beat each other, so there is a mini-max game in this case. These explanations is true for
accuracy of discriminator and generator, too.

Part. E (Performance of Adam)
The Adam is described and explained explicitly before in Problem. 1-Part. C, but here
we present the full algorithm of Adam (Fig. 14).

RESULTS:
Here are the results and outcomes of the DCGAN. It is obvious that through time the results are
being made better and better. Fig. 15 shows the results at first, 5th, 35th, and last epoch from top-left
to the bottom right. (We trained the GAN for 60 epochs.)
Fig. 16 shows more results related to DCGAN network

Part. F (BEGAN)
Boundary Equilibrium Generative Adversarial Network
In the BEGAN framework, the discriminator is an auto-encoder. The objective function
is a mini-max game, but unlike standard GANs, the generator tries to minimize the
reconstruction error of the images it provides to the discriminator while the discriminator
wants to simultaneously maximize this error and minimize the reconstruction error of the
true images. The following figure gives a summary of this architecture (Fig. 17).

The BEGAN model is based on the assumption that the reconstruction error of an autoencoder follows a normal distribution, when it is the case we can compute the Wasserstein
distance between them relatively easily.
Let 𝐿(𝑣) = |𝑣 − 𝐷(𝑣)| be the reconstruction error of some image v, if
𝐿(𝑋) ~ 𝑁(𝑚1 , 𝐶1) and 𝐿(𝐺(𝑧)) ~ 𝑁(𝑚2 , 𝐶2) then the Wasserstein distance between the
true image reconstruction and the fake image reconstruction is given by:
𝑊(𝐿(𝑋) , 𝐿(𝐺(𝑧))2 = ‖𝑚1 − 𝑚2‖2 + (𝐶1 + 𝐶2 − 2√𝐶1𝐶2)
Therefore, if (𝐶1+ 𝐶2−2√𝐶1𝐶2)
‖𝑚1− 𝑚2‖2 is constant or monotonically increasing 𝑊(𝐿(𝑋) , 𝐿(𝐺(𝑧))2 ∝
‖𝑚1 − 𝑚2‖2 , thus, we can maximize this equation by minimizing 𝑚1 (which is equivalent
to auto-encoding the real images) and maximizing𝑚2.

The BEGAN objective is then given by:
{𝐿𝐷 =𝐿𝐺𝐿(=𝑋𝐿)(−𝐺𝐿(𝑧()𝐺)(𝑧))
We should consider the following critical tips in designing of out BEGAN.
 No batch-norm in the encoders.
 No RELU/Leaky-RELU activation, exponential linear unit (ELU) seemed to work
better.
 No activation function in the decoders output; instead clip the values between [-1, 1]
 Do not use an auto-encoder with too big capacity (e.g. Squeeze-Net) for the
discriminator.
 Penalizing the generator with the L2 distance between the generated samples and the
true images really helped stabilize training but resulted in square artefacts. (This one
significantly helped when both the generator and discriminator had similar capacity)
 Using Squeeze-Net as the encoder for the generator achieved better results than using
the same model as the discriminator, also, it did not require the need of any kind of
penalty on the loss to keep training stable.
Main goal of designing the BEGAN
It’s all about providing a better loss to the networks. It has been previously shown that
the first GAN architecture minimizes the Kullback-Leibler divergence between the real data
distribution PX and the generated data distribution PG(z). An unfortunate consequence of
trying to minimize this distance is that the discriminator D gives meaningless gradients to
the generator G if D gets too good too quickly.
Since then, a few publications focused their effort on trying to find better loss functions:
The Improved Wassertein GAN (since its first version) minimizes the Wasserstein
distance (also called the Earth-Mover distance) by giving very simple gradients to the
networks (+1 if the output should be considered real and -1 if the output should be considered
fake)
The Least Squares GAN uses a least squares loss function to minimize the
Pearson divergence between D‘s output and its target.
The Least Squares GAN uses a least squares loss function to minimize the
Pearson divergence between D‘s output and its target.
The main goal behind the BEGAN is also to change the loss function. This time, it is
achieved by making D as an auto-encoder. The loss is a function of the quality of
reconstruction achieved by D on real and generated images.

The idea
Let’s start by clarifying something important. The reconstruction loss is not the same thing
as the real loss that the nets are trying to minimize. The reconstruction loss is the error
associated to reconstructing images through the auto-encoder/discriminator. In the EBGAN
schema the reconstruction-loss is referred to as « Dist » and the real loss is referred to as
« L ».
The main idea behind the BEGAN is that matching the distributions of the reconstruction
losses can be a suitable proxy for matching the data distributions. The real loss is then
derived from the Wasserstein distance between the reconstruction losses of real and
generated data. Later, the networks are trained by using this real loss in conjunction with an
equilibrium term to balance D and G.

Training
The training goes like this:
 D (the auto-encoder) reconstructs real images better and better. Said differently, the
weights of D are updated so that the reconstruction loss of real images is minimized.
 D simultaneously increases the reconstruction loss of generated images.
 And G works adversarially to that by minimizing the reconstruction loss of generated
images.
First and second points can be rephrased as « D tries to discriminate real and generated
distributions ». So G can only succeed with point 3 by generating more realistic images.

Part. G (Topology and decoder and encoder)
The topology of a BEGAN is like fig. 19.

Part. H (Importance of encoder and decoder)
Generally, discriminator works as a classifier and presents a number as the probability of
originality of image as output, but here it works as an auto-encoder network, it extracts the
most hidden features of the image using the encoder and reconstructs it with a decoder and
announces the mean square error (MSE) for this reconstruction image as output.
First we use real images to train auto-encoder, then we use it to measure the generated
images. Reconstructed images will be false due to non-professional fake images, because
the extracted features by the encoder will be incomplete. Therefore, they will not have the
necessary information and features for the decoder part to reconstruct the image. But if the
image is professionally forged and very similar to the real image, the error of the
reconstructed image will be very small.
So, we need to change the target function till we can train the discriminator both as an
Auto-encoder and as an evaluator.

