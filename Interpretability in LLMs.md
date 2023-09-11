# Paper List
Paper reading list in 💬 **Efficient Large Language Model** and 📝 **Understand LLMs**. This repository will keep updating ... 🤗

- [Efficient Large Language Model](#Efficient-Large-Language-Model)
- [Understand LLMs](#Understand-LLMs)
  - [Model Structure](#model-structure)
  - [Gradient Approximate](#gradient-approximate)
  - [LLM as latent variable Model](#LLM-as-latent-variable-Model)
- [Reasoning in LLMs](#Reasoning-in-LLMs)
- [Causal Inference in LLMs](#Causal-Inference)
***


## Efficient Large Language Model
* Fast inference from transformers via speculative decoding.,ICLR23 oral. [Paper](https://arxiv.org/abs/2211.17192)
* SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification. [Paper](https://arxiv.org/pdf/2305.09781.pdf)

👆 [Back to Top](#paper-list)

## Understand LLMs
**Tracr**: TRAnsformer Compiler for RASP

### Model Structure
* Refer to my another blog "注意力机制（induction head) contribute to In-context Learning"
hanqi-qi：注意力机制（induction head) contribute to In-context Learning

[1] In-context Learning and Induction Heads (Anthropic AI)
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
​transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

Other researches about Attention in Transformers：

[2]Transformers as Algorithms: Generalization and Stability in In-context Learning. ICML23
https://arxiv.org/pdf/2301.07067.pdf
​arxiv.org/pdf/2301.07067.pdf
contributions:
（a）ICL 可以被重新frame 成multiple task （MTL）的任务，并且从sequence data中学到full tasks的知识
（b）attention机制保证了在MTL框架下的generazability
  (c)   MTL的generalizability bound 可以推广到meta-learning/transfer learning的框架下

Prerequisite reading:
A Mathematical Framework for Transformer Circuits
https://transformer-circuits.pub/2021/framework/index.html#three-kinds-of-composition
​transformer-circuits.pub/2021/framework/index.html#three-kinds-of-composition
Key concepts:
【1】Privileged basis:
Residual stream to have “no privileged basis. By this we mean that there is no reason to expect the individual coordinates in the stream to have any particular meaning or significant property at all. This belief arises from the observation that every operation that reads from or writes to the residual stream does so via an arbitrary full-rank linear transformation. That in turn implies that we could transform the residual stream by an arbitrary full-rank linear transformation, and then also multiply the same transformation into every other matrix in the Transformer in the appropriate way, and arrive at an identical function with completely different coordinates.

### Gradient Descent

Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers2022

WHAT LEARNING ALGORITHM IS IN-CONTEXT LEARNING? INVESTIGATIONS WITH LINEAR MODELS. ICLR23
要注意的是两种不同的设定：一种是single self-attention layer在ICLsamples上的trained 之后，它的output等同于等同于linear model进行梯度优化的结果。 另外的是一种是：Transformer(TF) train on ICL objective but linear regression data（不同的分布见以下4种），对不同linear model的拟合程度(其中 设计了SFD, ILWD两种计算output 和weight 偏差的指标来衡量TF对这些linear predictor的拟合程度)：包括：
 k-neareset 
one-step stochastic optimization 
ridge-regression 
one-batch gradient descent. 
这些实验都是在仿真数据上进行的，x, w 均采样于N(0,I), y是通过以上不同方法计算得到的。
Transformers Learn In-Context by Gradient Descent. ICML23

相比于前两篇文章，这篇文章distinguished的贡献点也不少，怪不得引用这么多。乍一看，都是说LSA（linear self-attention）(linear attention layer 在in-context data上训练的结果的结果可以近似于 linear layer做一次GD的结果（理论说明），且在linear regression的task上可以得到近似结果（实验说明）。结果当然啦，ICLR23这篇文章说的是有MLP的self-attention）,但新的地方在于：
（1） 提出要判断两个model algorithm 是不是equivalent, 不能只看他们的prediction是否一致，还要看(a)interpolation (b) oov result (c) repeat GD多次是否能得到类似结果。

（2）在LSA之前引入MLP， 可以去做非线性的任务的拟合，比如sinWave.

(3) 结合induction-head中的发现，再一次验证了one-layer attention （copy) 和two-layer attention(induction head) 分别的作用 。这是我个人比较关注的重点。具体线索如下：

在接近GD效果之前，参照左图 在training step2000-3000之间，第一层的输出（参照右图），也就是当前token的representation e_j 与下一个token的e_{j+1}密切相关。那么第二层则是与GD相关。
那么copy是进行gradient descent 的前提条件，并且这个copy是通过softmax的实现：计算当前token 与其它token的correlation的相关，其实就是记录当前token attention值的过程，这个在我之前详解induction head circuit的blog也有提到。

### LLM as latent variable Model

AN EXPLANATION OF IN-CONTEXT LEARNING AS IMPLICIT BAYESIAN INFERENCE. cite160+ 2021
这是第一篇广为流传的用latent variable model解释大模型in-context learning的文章
An Explanation of In-context Learning as Implicit Bayesian Inference
​arxiv.org/abs/2111.02080

Schema-learning and rebinding as mechanisms of in-context learning and emergence. DeepMind, June23
TL;DR: Propose a sequence learning model based on action->latent variable->observed variable Generation process.
https://arxiv.org/pdf/2307.01201.pdf
​arxiv.org/pdf/2307.01201.pdf

### Training Data
Data Distributional Properties Drive Emergent In-Context Learning in Transformers
https://proceedings.neurips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf

Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression
​proceedings.neurips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf


**(E) multiple skills**
A Theory for Emergence of Complex Skills in Language Models
​arxiv.org/pdf/2307.15936.pdf

## Causal Inference
其实很多文章跟principle 的causal inference关系没那么紧密，就是用了一下causal 的几个重要概念。但也算是一个流派吧。
Causal interventions expose implicit situation models for commonsense language understanding. ACL-findings 23
https://arxiv.org/pdf/2306.03882.pdf

​arxiv.org/pdf/2306.03882.pdf

## Reasoning-in-LLMs
Make a Choice! Knowledge Base Question Answering with In-Context Learning

STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning.  Neurips22, cite110+

STaR: Bootstrapping Reasoning With Reasoning
​openreview.net/forum?id=_3ELRdg2sgI

ChatGPT is a Knowledgeable but Inexperienced Solver: An Investigation of Commonsense Problem in Large Language Models.
https://arxiv.org/pdf/2303.16421.pdf

​arxiv.org/pdf/2303.16421.pdf

**Cognitive Perspective**
https://arxiv.org/pdf/2301.06627.pdf

​arxiv.org/pdf/2301.06627.pdf
