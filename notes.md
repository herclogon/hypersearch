## Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization

* exploits the iterative algorithms of machine learning and embarassing parallelism of hyperparameter optimization
* poses the problem as a hyperparameter *evaluation* problem rather than a hyperparameter *selection* problem
* select configurations randomly
* looks at more hyperparameter configurations by computing more efficiently, which makes up for the ineffectiveness of RS
* relies on an early-stopping strategy for iterative algorithms
* rate of convergence does not need to be known in advance

## Intuition

If a hyperparameter configuration is destined to be the best after a large number of iterations, it is more likely than not to perform in the top half of configurations after a small number of iterations.

Another way of phrasing this is that even if performance after a small number of iterations is very unrepresentative of the configuration's *absolute* performance, its *relative* performance compared with many alternatives trained with the same number of iterations is roughly maintained.

A counterexample to this is for the learning rate: smaller valus will likely appear to perform worse for a small number of iterations but may outperform the pack after a large number of iterations. To account for this, we hedge and loop over vatying degrees of the agressiveness of balancing breadth vs. depth based search.

## Algorithm

Hyperband requires the ability to:

* sample a hyperparameter configuration (`get_random_config()`)
* train a particular configuration for a set number of iterations and evaluate loss on the validation set (`run_config()`)

The basic and obvious sampling choice is uniform random sampling.

```python
max_iter = 81 # maximum iters per config
eta = 3 # downsampling rate
logeta = lambda x: log(x) / log(eta)
s_max = int(logeta(max_iter)) # number of unique executions
B = (s_max + 1)*max_iter # total number of iterations (without reuse) per execution of successive halving (n, r)

# begin finite horizon outerloop
for s in reversed(range(s_max+1)):

	# initial number of configs
	n = int(ceil(B/max_iter/(s+1)*eta**s))
	# initial number of iterations to run configs for
	r = max_iter*eta**(-s)
	
	# begin finite horizon succ halving with (n, r)
	T = [get_random_config() for i in range(n)]
	for i in range(s+1):
		# run each of the n_i configs for r_i iters
		n_i = n*eta**(-i)
		r_i = r*eta**(-i)
		val_losses = [run_config(num_iters=r_i, params=t) for t in T]
		
		# keep best n_i / eta
		T = [T[i] for i in argsort(val_losses)[0:int(n_i/eta)]]
```





































