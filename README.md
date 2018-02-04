## HYPERBAND: A Novel Bandit-Based Approach to Hyperparameter Optimization

* exploits the iterative algorithms of machine learning and embarassing parallelism of hyperparameter optimization
* poses the problem as a hyperparameter *evaluation* problem rather than a hyperparameter *selection* problem
* allocates more resources to promising hyperparameter configurations while quickly eliminating poor ones
* adaptive computation increases efficiency, thereby examining orders of magnitude more hyperparameter configurations
* relies on an early-stopping strategy for pruning hyperparameter configurations
* overall, a general-purpose technique, making minimal assumptions about the problem

## Intuition

If a hyperparameter configuration is destined to be the best after a large number of iterations, it is more likely than not to perform in the top half of configurations after a small number of iterations.

Another way of phrasing this is that even if performance after a small number of iterations is very unrepresentative of the configuration's *absolute* performance, its *relative* performance compared with many alternatives trained with the same number of iterations is roughly maintained. This is particulary intuitive for machine learning algorithms which are trained using SGD: the iterative nature of training means that if a certain set of parameters have been performing badly for a few iterations, then we might as well kill it off and try a different set of parameters.

We should note, however, that there are some counterexamples to this. For the learning rate, for example, smaller values will likely appear to perform worse for a small number of iterations but may outperform the pack after a large number of iterations. We'll see how the authors fight this in the next section.

## Algorithm

#### Notation

* `B`: total, fixed, finite time budget
* `n`: number of hyperparameter configurations. Here, we sample the `n` i.i.d. samples from some distribution defined over the hyperparameter configuration space.

#### Dilemna

HYPERBAND extends the *Successive Halving* algorithm and calls it as a subroutine. The idea behind this algorithm is to uniformly allocate a budget to a set of hyperparameter configurations, evaluate the performance of all configurations, throw out the worst half, and repeat until one configurations remains. 

Thus, successive halving allocates **exponentially** more resources to more promising configurations. We can show that on average, given a fixed time budget `B` and `n` hyperparameter configurations, *Successive Halving* allocates `B/n` resources across the configurations.

The authors argue that forcing the user to pick a value of n is a crucial drawback to the algorithm since it involves a certain tradeoff.

* a small n means a smaller number of configurations with longer average training times, i.e. more resourcers per config.
* a large n means a larger number of configurations with smaller average training times, i.e. less resources per config.

There are cases that warrant a small n (when 2 configs converge slowly, it gets harder to differentiate between the 2, and similarly, when 2 configs have very similar performace, it gets harder to tell them apart, so we need to train for longer, i.e. dedicate more resources per config) and other cases that warrant a larger n (if the iterative training method converges very quickly and the quality of configs is revealed using minimal resources, so we need to train less, i.e. dedicate less resources per config).

#### Solution

HYPERBAND tries to address this “**n versus B/n**” problem by performing a grid search over feasible value of n for a given B. It consists in 2 components:

* an inner loop that invokes the *Successive Halving* algorithm for a fixed `n` and `r`.
* an outer loop which iterates over different values of `n` and `r`.

It requires 2 inputs:

- `R`: maximum amount of resource that can be allocated to a single configuration
- `nu`: controls the proportion of configurations discarded in each round of *Successive Halving* (classically, `nu=2`)

These 2 inputs dictate how many runs of *Successive Halving* happen. The outer loop ranges from `s_max + 1` to `0` where `s_max` is a function of `R`. In each s (also called bracket), we reduce `n` successively by `nu`.

## Example

* `R = 81`
* `nu = 3`
* `s_max = log_3(81) = 4` 
* `B = (4+1)R = 5R`

In the normal case, we would have run *Successive Halving* **once**, for a user-defined number of configurations `n`, for a total of `log_nu(n)` iterations.

With HYPERBAND, we run *Successive Halving* `s_max+1` times (`0` to `s_max`), where in each SH bracket, we calculate the number of configurations `n` to sample along with `r`, and run each bracket for `log_nu(n)` iterations. Here, `r` is the minimum amount of iterations to run each configuration for.

## Implementation

HYPERBAND levarages 2 functions:

* `get_random_config()`: returns a set of n i.i.d. samples from some distribution defined over the hyperparameter configuration space.
* `run_config(t, r)`: takes a hyperparameter configuration `t` and resource allocation `r` and returns the validation loss after training for the allocated resources.

## Parameter Advice

* Set `R` to the usual amount one would train neural networks for. It's mostly a rule fo thumb, but something in the range `[80, 150]`.
* Larger values of `nu` correspond to a more aggressive elimination schedule and thus fewer rounds of elimination. Increase to receive faster results at the cost of a sub-optimal performance. Authors advise a value of `3` or `4`.
* Resume from a checkpoint to reduce work (for configurations that are kept).

## Proofs

* On average, given a fixed time budget `B` and `n` hyperparameter configurations, *Successive Halving* allocates `B/n` resources across the configurations - [click here](https://github.com/kevinzakka/pyperband/tree/master/proofs/proof1.pdf)
* Successive halving sees more parameter configurations than pure random search - [click here]()

## Resources

- [arXiv Paper](https://arxiv.org/abs/1603.06560)
- [Hyperband Demo](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html)