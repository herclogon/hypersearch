[WIP]

* v1 support: fc networks
* v2 support: conv networks

## Todo

- [ ] setup checkpointing
- [ ] train loop
- [ ] multi-gpu support

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
    - [ ] conv Layers
    - [ ] fc Layers
- [ ] Change Layer Params
    - [x] change fc output size
    - [ ] change conv params
- [ ] Optimization
    - [x] batch size
    - [x] learning rate
    - [x] optimizer (adam, sgd)