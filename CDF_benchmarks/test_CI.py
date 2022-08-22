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
wpct = sip.interp1d([0, *(weights.cumsum()-weights/2)/weights.sum(), 1], [data.min(), *data, data.max()])
dcdf = sip.interp1d([0, *np.cumsum(weights), 1], [data.min(), *data, data.max()], kind="zero")
# Test if they all work for the whole range
xplot = np.linspace(0,1,10000)
anest(xplot)
lh(xplot)
wpct(xplot)
dcdf(xplot)

def credibility_interval_of_invCDF(invCDF, level):
    distance = lambda Y, level=level: invCDF(Y+level)-invCDF(Y)
    res = sop.minimize_scalar(distance, bounds=(0, 1-level), method="Bounded")
    return np.array([invCDF(res.x), invCDF(res.x+level)])


cases = ["wpct", "anest", "dcdf", "lh"]
accuracyleft = {}
accuracyright = {}
intervalsize = {}
for c in cases:
	accuracyleft[c] = []
	accuracyright[c] = []
	intervalsize[c] = []
sizes = [1e2]
for random_weights in [True]:
	for size in sizes:
		print("size", size)
		N_data = int(size)
		t_CI = []
		anest_CI = []
		lh_CI = []
		wpct_CI = []
		dcdf_CI = []
		for i in range(10000):
			level = np.random.uniform()
			data = np.random.normal(size=N_data)
			data.sort() #!
			if random_weights:
				weights = np.random.uniform(size=N_data)
			else:
				weights = np.ones(N_data)
			weights /= weights.sum() #!
			t = sst.norm.ppf
			anest = anesthetic.utils.cdf(data, w=weights, inverse=True)
			lh = sip.interp1d([0, *np.cumsum(weights), 1], [data.min(), *data, data.max()])
			wpct = sip.interp1d([0, *(weights.cumsum()-weights/2)/weights.sum(), 1], [data.min(), *data, data.max()])
			dcdf = sip.interp1d([0, *np.cumsum(weights), 1], [data.min(), *data, data.max()], kind="zero")
			t_CI.append(credibility_interval_of_invCDF(t, level))
			anest_CI.append(credibility_interval_of_invCDF(anest, level))
			accuracyleft["anest"].append(anest_CI[-1][0]-t_CI[-1][0])
			accuracyright["anest"].append(anest_CI[-1][1]-t_CI[-1][1])
			intervalsize["anest"].append(anest_CI[-1][1]-anest_CI[-1][0]-t_CI[-1][1]+t_CI[-1][0])
			lh_CI.append(credibility_interval_of_invCDF(lh, level))
			accuracyleft["lh"].append(lh_CI[-1][0]-t_CI[-1][0])
			accuracyright["lh"].append(lh_CI[-1][1]-t_CI[-1][1])
			intervalsize["lh"].append(lh_CI[-1][1]-lh_CI[-1][0]-t_CI[-1][1]+t_CI[-1][0])
			wpct_CI.append(credibility_interval_of_invCDF(wpct, level))
			accuracyleft["wpct"].append(wpct_CI[-1][0]-t_CI[-1][0])
			accuracyright["wpct"].append(wpct_CI[-1][1]-t_CI[-1][1])
			intervalsize["wpct"].append(wpct_CI[-1][1]-wpct_CI[-1][0]-t_CI[-1][1]+t_CI[-1][0])
			dcdf_CI.append(credibility_interval_of_invCDF(dcdf, level))
			accuracyleft["dcdf"].append(dcdf_CI[-1][0]-t_CI[-1][0])
			accuracyright["dcdf"].append(dcdf_CI[-1][1]-t_CI[-1][1])
			intervalsize["dcdf"].append(dcdf_CI[-1][1]-dcdf_CI[-1][0]-t_CI[-1][1]+t_CI[-1][0])

fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(8,4))
ax1.axvline(0, color="k")
ax1.set_title("Left interbal boundary Method-True")
ax2.axvline(0, color="k")
ax2.set_title("Right interbal boundary Method-True")
for key in accuracyleft.keys():
	ax1.hist(accuracyleft[key], label="accuracy", histtype="step", bins=100, density=False, range=[-3,3])
	ax2.hist(accuracyright[key], label="accuracy", histtype="step", bins=100, density=False, range=[-3,3])
plt.savefig("bias.png", dpi=600)
plt.show()

plt.axvline(0, color="k")
for key in intervalsize.keys():
	plt.hist(intervalsize[key], label="accuracy", histtype="step", bins=100, density=False, range=[-3,3])
plt.title("Interval size Method-True")
plt.savefig("size.png", dpi=600)
plt.show()