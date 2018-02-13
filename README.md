[WIP]

v1 support: fc networks
v2 support: conv networks

## Supported Hyperparameters

- [x] Activation
    - [x] all
    - [x] per layer
- [x] L1/L2 regularization (weights & biases)
    - [x] all
    - [x] per layer
- [x] Add Batch Norm
    - [x] sandwiched between every layer
- [x] Add Dropout
    - [x] sandwiched between every layer
- [ ] Add Layers
    - [ ] Conv Layers
    - [ ] FC Layers
- [ ] Change Layer Params
    - [x] change FC output size
    - [ ] change Conv # filters
    - [ ] change Conv filter size
- [x] Optimizer
- [x] Learning Rate

## Order of Params

`conv/fc -> ReLU -> Batch Norm -> Dropout`

## Default

- optimizer: `Adam(lr=1e-3)`
