## Hierachical VAE 

Basically different priors, hierachical priors.

### [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/pdf/1906.00446.pdf)

  (**) lead to paper [Discriminator Rejection Sampling](https://arxiv.org/pdf/1810.06758.pdf). Without details of bottom-top structure, or just un-sampling and shortcut
  
### [Deep AutoRegressive Networks](https://arxiv.org/abs/1310.8499)

  (****) The first paper I know to use NN to learn a new prior for better generation, so it is done in two stages:
  
  (1) train encoder-decoder and
  
  (2) train NN from latent code (reconstruct) to data point.
  
### [Taming Transformers for High-Resolution Image Synthesis.](https://arxiv.org/pdf/2012.09841.pdf) 

published in 2020 citation 500+. Easy to inplement following the [code](https://github.com/CompVis/taming-transformers)

Based on the two-stage VAE, they replace the pixelCNN with Transformer and replace reconstruction loss with perception loss+patch-level GAN loss.

Code can be used, consisting VQ-VAE and learning of rich prior.

### Nonparametric Variational Auto-Encoders for Hierarchical Representation Learning

### Multi-manifold clustering: A graph-constrained deep nonparametric method

### Learning Hierarchical Priors in VAEs

(**)Continous hierarchical VAE (integral to aggregate multiple single-prior ), using Lagrangian constrained optimizer to rewrite the ELBO.

### [Taming VAEs](https://arxiv.org/pdf/1810.00597.pdf)

(****) A work prior to Learning Hierarchical Priors in VAEs, while using the weighted sum to derive a rich prior. It is the prior work to the "Learning Hierarchical Priors in VAEs".



##### **star only refers to personal rating, varies from my current research topic, not means paper quality.
