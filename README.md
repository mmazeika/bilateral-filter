## Functionality
A differentiable bilateral filter CUDA kernel for PyTorch. It takes a tensor of shape (N,C,H,W) and applies a bilateral filter to each channel in parallel. The algorithm is a brute force bilateral filter using a 5x5 window and zero padding. The sigma parameters for distance and intensity can be modified.

## History
This was created in June 2018 with PyTorch 0.4 for research on adversarial examples, so it may not work with newer versions of PyTorch.

Gradients were derived by hand. I will try to find and upload the equations. The gradients with respect to the input match finite differences. The gradients with respect to sigma do not always match finite differences; I double checked the math and implementation and wasn't able to figure out why.

## Notes
Coming soon: Setup instructions and example code, including verification of the backward pass with a gradient checker and sample input-output images.

Warning: For CUDA reasons, do not use the function with too large of a batch size or channel count. If batch_size\*channels > 65535, the function will throw an error.
