# CUDA Kernels

A collection of CUDA kernel implementations and experiments for learning GPU programming. The files cover common operations in machine learning and parallel computing, with several alternative implementations for some operations.

## Inspiration

This repository grew out of three learning resources:

1. [*Programming Massively Parallel Processors: A Hands-on Approach*](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The book introduces CUDA programming, GPU architecture, and parallel patterns that appear throughout this repository. The publisher also provides [companion slides and lab materials](https://shop.elsevier.com/books/book-companion/9780323912310).
2. [Izzat El Hajj's Spring 2021 AUB lecture series](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4), published by the *Programming Massively Parallel Processors* channel. Its lectures work through topics such as tiling, convolution, reduction, scan, histogram, and sorting.
3. [Umar Jamil's 100 days of CUDA challenge](https://github.com/hkproj/100-days-of-gpu/blob/main/CUDA.md) and the Discord community around it. The challenge encourages consistent kernel practice and documenting what you learn as you go.

For readers looking for a freely accessible copy, IIT Delhi hosts a [PDF of the 2010 first edition](https://www.cse.iitd.ac.in/~rijurekha/col730_2022/cudabook.pdf) by Kirk and Hwu. It is an older edition than the one linked above.

## Contents

| Directory | Operations |
| --- | --- |
| `Activation Kernels/` | ReLU, Leaky ReLU, ELU, GELU, sigmoid, hard sigmoid, tanh, softplus, swish, Mish, SELU, and softmax |
| `Convolution/` | 1D and 2D convolution |
| `Embedding Kernels/` | Embedding lookup |
| `Fused Kernels/` | Fused dense, bias, and ReLU operation |
| `GeMM/` | Basic and tiled matrix multiplication |
| `Histogram/` | Histogram kernels and a performance comparison image |
| `Loss Kernels/` | Hinge, Huber, and KL divergence losses |
| `Normalization Kernels/` | Batch, L1, L2, RMS, and Frobenius normalization |
| `Parallel Scan/` | Inclusive and exclusive Brent–Kung and Kogge–Stone scans |
| `Pooling Kernels/` | 1D average pooling |
| `Reduction/` | Reduction variants |
| `Sorting Kernels/` | Merge sort |
| `Stencil/` | 3D stencil kernels |
| `Vector Addition/` | Standalone vector addition example |

## Getting started

You need a CUDA-capable GPU and the NVIDIA CUDA Toolkit, including `nvcc`, to build and run these files. The vector addition example has its own `main()` function:

```bash
nvcc "Vector Addition/Vector_add.cu" -o vector_add
./vector_add
```

It prints the kernel execution time and the number of elements processed.

Most other files are individual examples rather than parts of one executable. Files exporting `extern "C" void solution(...)` were written for problems on [Tensara](https://tensara.org/problems), a GPU programming challenge platform. Tensara defines the problem and calls this entry point; these files are not standalone programs. Other files define kernels without a `solution` wrapper. Build and integrate them one at a time, using the function signature and input assumptions in each file. In particular, some vectorized kernels require element counts divisible by four.

## Project status

This repository does not yet include a shared build system or automated test suite. Treat performance comments and example kernels as experiments, and validate correctness and performance for your own shapes and GPU before using them in an application.
