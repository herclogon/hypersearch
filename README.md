## Supported Hyperparameters

- [x] Activation
    - [x] all
    - [x] per layer
- [x] L1/L2 regularization (weights & biases)
    - [x] all
    - [x] per layer
- [x] Add Batch Norm
    - [x] sandwiched between ever layer
- [x] Add Dropout
    - [x] sandwiched between ever layer
- [ ] Add Hidden Layer
- [ ] Change Hidden Layer Size
- [x] Learning Rate
- [x] Optimizer


## Order of Params

`conv/fc -> ReLU -> Batch Norm -> Dropout`

## Default

- `optimizer`: `Adam(lr=1e-3)`
