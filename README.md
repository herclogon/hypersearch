[WIP]

- [x] v1 support: fc networks
- [ ] v2 support: conv networks

## Restrictions

* Only fully-connected networks currently supported
* You need to define the model using `nn.Sequential` in `model.py`
* Make sure the training loop `train_one_epoch()` fits your need:
    * the correct loss is being computed. The default one is currently `F.nll_loss`

## Todo

- [x] setup checkpointing
- [x] train loop
- [x] < 1 epoch support
- [x] filter out early stopping configs
- [ ] visualization + logging support
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