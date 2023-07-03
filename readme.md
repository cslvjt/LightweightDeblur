# Lightweight Deep Deblurring Model with Discriminative Multi-scale Feature Fusion
## Abstract
Although existing learning-based deblurring methods achieve significant progress, these approaches tend to require lots of network parameters and huge computational costs, which limits their practical applications. 
Instead of pursuing larger deep models for boosting deblurring performance, we propose a lightweight deep convolutional neural network with lower computational costs and comparable restoration performance, which is based on a multi-scale framework with an encoder and decoder network architecture.
Specifically, we present an effective depth-wise separable convolution block (DSCB) as the fundamental building block of our method to reduce the model complexity.
In addition, to better utilize the features from different scales, we develop a simple yet effective discriminative multi-scale feature fusion (DMFF) module for achieving high-quality results. 
Experimental results on the benchmarks show that our method is about $10\times$ smaller than the state-of-the-art deblurring methods, MPRNet, in terms of model parameters and FLOPs while achieving competitive performance. 
## Architecture
![model architecture](assets\arh.png)
## Eval
```python eval.py```
## Performance
![Performance](assets\performance.png)
## Requirement
see requirements
## Acknowledgment
This code is based on the MIMO-UNet and BasicSR