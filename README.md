# micrograd.cpp

A lightweight scalar autograd engine and neural network library implemented in modern **C++23**, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Features

- **Scalar Autograd Engine (`Value`)**: Reverse-mode automatic differentiation over dynamically constructed DAGs using topological sorting (DFS).
- **Clean Value Semantics**: Handle/body pattern with `std::shared_ptr<Node>` under the hood—no raw pointer management needed.
- **Neural Network Primitives**: `Neuron`, `Layer`, and `MLP` abstractions with gradient descent training.
- **Modern C++23**: Uses modern features like `std::println`, uniform initialization, and lambda-based backward closures.

## Quick Start

### Prerequisites
- Clang (Apple Clang or LLVM) / GCC supporting **C++23**
- Make

### Build & Run
```bash
make run
```

### Clean
```bash
make clean
```

## Example Output

```text
Epoch 0: loss = 4.2987
Epoch 10: loss = 0.4024
Epoch 20: loss = 0.1129
Epoch 30: loss = 0.0748
Epoch 40: loss = 0.0576
Epoch 49: loss = 0.0456

Final Predictions:
y_pred =>  1.1195, ys =>  1.0
y_pred => -0.9999, ys => -1.0
y_pred => -0.9840, ys => -1.0
y_pred =>  0.8272, ys =>  1.0
```
