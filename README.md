## Dataset Format

- name: CIFAR-10
- splits: train, valid, test
- sizes: 40k, 10K, 10K
- transforms

- iteration = 10,000 samples
- batch size: 100

## Currently Supported Ops

- [x] Regularization

## Stochastic Expressions

- `choice(l)`: returns a random element from the list `l`
- `randint(min, max)`: returns a random integer bounded by `min` and `max`
- `uniform(min, max)`: returns a random value ~ `U(min, max)`
- 'loguniform(min, max)': returns a random value ~ `exp(U(min, max))` such that the logarithm is uniformly distributed

#### Initialization

Changing initialization for every layer

#### Regularization

- L1 decay
- L2 decay

- Kernel Regularizer
- Bias Regualizer

- FC
- Conv

#### Dropout and Normalization

- Adding Dropout Layers
- Adding Batch Norm Layers
- Varying probability of Dropout Layers

#### Activation Function

Change activation function for fc or conv layers

#### Constraints

- Max Norm
- Non-Negativity

- FC
- Conv

#### Optimization

- changing optimizer
- learning rate scheduling
- learning rate on plateau

#### Hidden Units

- Change # of hidden units in fc layers