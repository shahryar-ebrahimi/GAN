# GAN
A project to demonstrate how Generative Adversarial Networks (GAN) work.

Problem 3:

In this part of project, we have been asked to explore about variational auto- encoder
and design it. This problem contains six individual parts which we talk about and discuss
each part separately.
Part. A (Auto-encoder & Variational Auto-encoder)
In the sequel, we will describe auto-encoders, its problems and advantages of an
variational auto-encoder.

Auto-encoder & Variational auto-encoder:
An auto-encoder network is actually a pair of two connected networks, an encoder and a
decoder. An encoder network takes in an input, and converts it into a smaller, dense
representation, which the decoder network can use to convert it back to the original input.
For instance, consider a CNN, the convolutional layers of any CNN take in a large image,
and convert it to a much more compact, dense representation. This dense representation is
then used by the fully connected classifier network to classify the image (see fig. 17).

The encoder is similar, it is simply is a network that takes in an input and produces a much
smaller representation (the encoding), that contains enough information for the next part of
the network to process it into the desired output format. Typically, the encoder is trained
together with the other parts of the network, optimized via back-propagation, to produce
encodings specifically useful for the task at hand. In CNNs, the 1000-dimensional encodings
produced are such that they’re specifically useful for classification.
Auto-encoders take this idea, and slightly flip it on its head, by making the encoder generate
encodings specifically useful for reconstructing its own input (Fig. 18)

The entire network is usually trained as a whole. The loss function is usually either the
mean-squared error or cross-entropy between the output and the input, known as
the reconstruction loss, which penalizes the network for creating outputs different from the
input.

As the encoding (which is simply the output of the hidden layer in the middle) has far
less units than the input, the encoder must choose to discard information. The encoder learns
to preserve as much of the relevant information as possible in the limited encoding, and
intelligently discard irrelevant parts. The decoder learns to take the encoding and properly
reconstruct it into a full image. Together, they form an auto-encoder.
Now, we are going to talk about what is going wrong with a standard auto-encoder.
Standard auto-encoders learn to generate compact representations and reconstruct their
inputs well, but asides from a few applications like de-noising auto-encoders, they are fairly
limited. The fundamental problem with auto-encoders, for generation, is that the latent space
they convert their inputs to and where their encoded vectors lie, may not be continuous, or
allow easy interpolation.

For example, training an auto-encoder on the MNIST dataset, and visualizing the
encodings from a 2D latent space reveals the formation of distinct clusters. This makes
sense, as distinct encodings for each image type makes it far easier for the decoder to decode
them. This is fine if you are just replicating the same images. But when you’re building
a generative model, you do not want to prepare to replicate the same image you put in. You
want to randomly sample from the latent space, or generate variations on an input image,
from a continuous latent space.
If the space has discontinuities (e.g. gaps between clusters) and you sample/generate a
variation from there, the decoder will simply generate an unrealistic output, because the
decoder has no idea how to deal with that region of the latent space. During training, it never
saw encoded vectors coming from that region of latent space (see fig. 19).

Variational auto-encoders (VAEs) have one fundamentally unique property that
separates them from vanilla auto-encoders, and it is this property that makes them so useful
for generative modeling: their latent spaces are, by design, continuous, allowing easy
random sampling and interpolation. It achieves this by doing something that seems rather
surprising at first: making its encoder not output an encoding vector of size n, rather,
outputting two vectors of size n: a vector of means, μ, and another vector of standard
deviations, σ.

Part. B (Variational variety)
According to what we discussed and talked in previous part, when we are using autoencoder, the feature space has not required characteristics to generate a new image which is
different with original input images. As you can see in fig. 19, the features space is not
continuous, also we cannot interpolate the space. So, we cannot choose a new combined
feature to generate a new image due to aforementioned reasons. Variational auto-encoders
changes the feature space changing the statistical parameters like μ, and σ. We want to have
a feature space like fig. 20, for choosing a proper and new combined feature. By serving a
VAE we can generate a continuous feature space (Fig. 20). We will introduce the Kullback–
Leible loss function in the following parts.


Part. C (Designing a VAE)
We designed and modeled a variational auto-encoder which its architecture has been mentioned in
fig. 21, and fig. 22

Part. D (Describing the steps of our work)
 Step.1: Loading the MNIST data to use as input data.
 Step.2: Designing an encoder to create a proper feature space. In this part, we design
an encoder using a CNN. The activation function is ‘RELU’. In the last layer we
have some µ, and σ.
 Step.4: In this step, we sampling from our feature space to apply it on decoder as
input.
 Step.5: We designed a decoder using transposed convolution. We used RELU as
activation function.
 Step.6: We served two different loss function. We used binary-crossentropy for
reconstruction loss and Kullback–Leible to measure the divergence between two
probabilities.
 Step.7: We compile the model using Adam as optimizer and fit the model on data
set.
You can find the architecture of encoder and decoder in Part. C.

Part. E (Loss functions)
We served two different loss functions. We used the binary-crossentropy for reconstruction
loss. In other words, we are checking the distance between our original input and output of
decoder.

We served the Kullback–Leible divergence into the loss function. The KL divergence
between two probability distributions simply measures how much they diverge from each
other. Minimizing the KL divergence, here means optimizing the probability distribution
parameters µ, and σ to closely resemble that of the target distribution.
Intuitively, this loss encourages the encoder to distribute all encodings for all type of input,
evenly around the center of the latent space. If it tries to cheat by clustering them apart into
specific regions, away from the origin, it will be penalized.

Now, using Kl loss results in a latent space results in encodings densely placed randomly,
near the center of the latent space, with little regard for similarity among nearby encodings.
The decoder finds it impossible to decode anything meaningful from this space, simply
because there really is not any meaning.

Part. F (Clustering)
We designed an encoder to prepare the data set for clustering. Fig. 26 shows the topology of
encoder, however, it should be mentioned that first of all we design an auto-encoder and
after training the auto-encoder, we use only the encoder part. So, fig. 26, shows the all parts
of auto-encoder.

Now, the data is ready to cluster, to achieve this goal we used K-means clustering method.
We choose ten clusters. In k-means method, we consider Euclidean Distance as our
benchmark. We clustered the data and here are some results of the implementation.
Fig. 27 shows fourteen images that all are representing the number 0. So this cluster is
belong to class 0.

Fig. 28 shows twenty-five images. In this class you can see twenty-one image that is
representing 0. So we can say this cluster is belong to class 1
