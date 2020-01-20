# A Metric for Generalization of Generative Latent Variable Models

This metric is based on the Minimum Description Length (MDL) principle, a formalization of the Occam's Razor principle.

The complexity of a model is the total cost of describing the model and the discrepancy between the model's output and 
true data. 

Unlike discriminative models, in generative models, true data is a distribution. 
We define the discrepancy as the Wasserstein-2 distance between the two distributions. The two distributions are 
approximated using two empirical distributions, i.e. two uniform distributions over samples from these distributions. 
W2 is (approximately) computed using a Sinkhorn solver. The distance is computed in the data space for simple datasets 
like MNIST and in the feature space (perceptual distance) for more complicated datasets like CIFAR10 and ImageNet.

The complexity of a model is defined as the complexity of the model's distribution. That is in turn defined as the
 expected length of the path connecting two random samples.  
```text
C = W2(p_g, p_r) * E_{p_g}[len(x_1, x_2)] 
where p_g, p_r are the model's and the target distributions.  
```

This metric was initially developed for GANs but it can be applied to other generative latent variable models such as 
VAEs as well. 
## Requirements
```text
Python 3.6
Pytorch 0.4
```

## Run
```bash
./runMDL<Algo>.sh
```

