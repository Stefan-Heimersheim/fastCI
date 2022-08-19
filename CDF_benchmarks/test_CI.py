import anesthetic
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sip
import scipy.optimize as sop
import scipy.stats as sst


cdefault = plt.rcParams['axes.prop_cycle'].by_key()['color']
random_weights = False
np.random.seed(0)

data = np.random.normal(size=10)
data.sort() #!
weights = np.random.uniform(size=10)
weights /= weights.sum() #!
w_ = np.concatenate([[0.],weights[1:]+weights[:-1]]).cumsum()
w_/=w_[-1]
sym = sip.interp1d(data, w_)
lh = sip.interp1d(data, np.cumsum(weights))
rh = sip.interp1d(data, 1-np.cumsum(weights[::-1])[::-1])
wpct = sip.interp1d(data, (weights.cumsum()-weights/2)/weights.sum())
anest = anesthetic.utils.cdf(data, w=weights)
dcdf = lambda x: weights[(data < x)].sum()
vdcdf = np.vectorize(dcdf)
xplot = np.linspace(np.min(data), data.max(), 10000)
plt.figure(constrained_layout=True, figsize=(6,4))
plt.scatter(data, np.cumsum(weights), color="k")
plt.scatter(data, 1-np.cumsum(weights[::-1])[::-1], color="k")
plt.plot(xplot, sym(xplot), label="sym")
plt.plot(xplot, vdcdf(xplot), label="dcdf")
plt.plot(xplot, lh(xplot), label="lh")
plt.plot(xplot, rh(xplot), label="rh")
plt.plot(xplot, wpct(xplot), label="wpct")
plt.plot(xplot, anest(xplot), label="anest", ls=":", color="pink")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("CDF")
plt.savefig("intuition.png", dpi=600)
plt.close()

# TEST
#xplot = np.linspace(0,1,10000)
#inv_dcdf2 = sip.interp1d([0, *np.cumsum(weights), 1], [np.min(data), *data, data.max()], kind="zero")
#dcdf2 = sip.interp1d([np.min(data), *data, data.max()], [0, *np.cumsum(weights), 1], kind="zero")
#dcdf = lambda x: weights[(data < x)].sum()
#vdcdf = np.vectorize(dcdf)
#plt.plot(xplot, dcdf2(xplot))
#plt.plot(xplot, vdcdf(xplot), ls="--")
#plt.show()

# Define invCDFs:
anest = anesthetic.utils.cdf(data, w=weights, inverse=True)
# skipping sym as equivalent
lh = sip.interp1d([0, *np.cumsum(weights), 1], [data.min(), *data, data.max()])
# skipping rh in favour of lh
wpct = sip.interp1d(data, (weights.cumsum()-weights/2)/weights.sum())
dcdf = sip.interp1d([0, *np.cumsum(weights), 1], [data.min(), *data, data.max()], kind="zero")
# Test if they all work for the whole range
xplot = np.linspace(0,1,10000)
anest(xplot)
lh(xplot)
wpct(xplot)
dcdf(xplot)

def credibility_interval_of_invCDF(invCDF, level):
    distance = lambda Y, level=level: invCDF(Y+level)-invCDF(Y)
    res = minimize_scalar(distance, bounds=(0, 1-level), method="Bounded")
    return np.array([invCDF(res.x), invCDF(res.x+level)])


for random_weights in [False, True]:
	for size in [1e1, 1e2, 1e3]:
		print("size", size)
		N_data = int(size)
		true_cdf = []
		discrete_cdf = []
		sym_cdf = []
		wpct_cdf = []
		anest_cdf = []
		lh_cdf = []
		rh_cdf = []
		for i in range(10000):
			data = np.random.normal(size=N_data)
			data.sort() #!
			if random_weights:
				weights = np.random.uniform(size=N_data)
			else:
				weights = np.ones(N_data)
			weights /= weights.sum() #!
			