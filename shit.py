from math import log, ceil

max_iter = 81
eta = 3
logeta = lambda x: log(x) / log(eta)
s_max = int(logeta(max_iter))
B = (s_max+1) * max_iter

for s in reversed(range(s_max+1)):
    print("s: {}".format(s))

    n = int(ceil(B / max_iter / (s + 1) * eta ** s))
    r = max_iter * eta ** (-s)

    for i in range(s+1):
        n_i = n * eta ** (-i)
        r_i = r * eta ** (i)
        print("\tn_i: {}, r_i: {}".format(n_i, r_i))