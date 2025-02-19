import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt


def ln(p):
    return np.log(1 / p)


def Q(f):
    return 1 - norm.cdf(f)


def ff(f):
    return norm.pdf(f)


n = 2000  # number of samples
m = 100
p = 0.01
a = m / (n * p)

t = np.arange(-4 - a / 2, a / 2 + 10, 0.01)
z0 = (t + a / 2) / np.sqrt(a)
z1 = (t - a / 2) / np.sqrt(a)
Qf = Q(z0)
Rp = Q(z1)
Fn = 1 - Q(z1)
Pr = p * Rp / (p * Rp + (1 - p) * Qf)

plt.plot(t, Rp, "r", label="recall")
# plt.plot(t, Qf)
plt.plot(t, Pr, label="precission")
# plt.plot(t, Fn)
plt.plot(t, Rp + Pr)

D = p * Rp + (1 - p) * norm.cdf(norm.ppf(Rp) - np.sqrt(a))
D2 = p + (1 - p) * np.exp(np.sqrt(a) * norm.ppf(Rp) - a / 2)

plt.plot(t, p * Rp / D, label="precission2")
plt.plot(t, p / D2, label="precission3")
plt.vlines(-a / 2, 0, 1, colors="r", linestyles="dashed")
plt.vlines(a / 2, 0, 1, colors="r", linestyles="dashed")
plt.vlines(np.log(1 / p), 0, 1, colors="black", linestyles="dashed")
plt.legend()

# plt.figure()

# pp = np.arange(0.01, 1, 0.01)
# plt.plot(pp, norm.ppf(pp))

# plt.plot(t, Rp / Qf)
# plt.plot(t, np.exp(t))
plt.show()

# "$\int{u f_{-}{t}} = u Q(t) - \int(Q$"
