import anesthetic
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sip
import scipy.optimize as sop
import scipy.stats as sst
cdefault = plt.rcParams['axes.prop_cycle'].by_key()['color']
from cdfs import *

illustration_path = "../illustrations/"
random_weights = False

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

def credibility_interval_of_invCDF(invCDF, level):
    distance = lambda Y, level=level: invCDF(Y+level)-invCDF(Y)
    res = minimize_scalar(distance, bounds=(0, 1-level), method="Bounded")
    return np.array([invCDF(res.x), invCDF(res.x+level)])

N_data = 10
np.random.seed(3)
#Out[3]: [0.6769716479761074, 0.6634099521594515, 0.6360125963233798]
#Out[1]: [0.6431209950881398, 0.6703904770745072, 0.6259829356541247]
#Out[1]: [0.686789261495815, 0.6867781341563864, 0.6791169046984594]

scores = {"discrete": [], "wpct": [], "anesthetic": []}
for loc in [0]:
	for scale in 10**np.random.uniform(-4,4, size=1000):
		level = np.random.uniform(0,1)
		data = np.random.normal(loc=loc, scale=scale, size=N_data)
		data.sort()
		weights = np.random.uniform(0,1,size=N_data)
		d_CDF_if = true_CDF2(data, weights, inverse=True)
		d_CDF_f = true_CDF2(data, weights, inverse=False)
		wpct_if = wpct(data, weights, inverse=True)
		wpct_f = wpct(data, weights, inverse=False)
		anesthetic_if = anesthetic.utils.cdf(data, w=weights, inverse=True)
		anesthetic_f = anesthetic.utils.cdf(data, w=weights, inverse=False)
		true_if = lambda x: sst.norm.ppf(x, loc, scale)
		true_f = lambda x: sst.norm.cdf(x, loc, scale)
		functions = {"true": true_f , "discrete": d_CDF_f, "wpct": wpct_f, "anesthetic": anesthetic_f}
		ifunctions = {"true": true_if , "discrete": d_CDF_if, "wpct": wpct_if, "anesthetic": anesthetic_if}
		#real_c = [sst.norm.pdf(level-(1-level)/2), sst.norm.pdf(level+(1-level)/2)]
		#plt.plot(real_c, [0, 0], label="Real")
		plotting = False
		if plotting:
			y = 0
			xplot = np.linspace(np.min(data), np.max(data), 1000)
			fig, [a,b] = plt.subplots(ncols=2)
			fig.suptitle("Level = "+str(level)+", N = "+str(N_data))
			for key in ["discrete", "wpct", "anesthetic", "true"]:
				print(key)
				invCDF = ifunctions[key]
				CDF = functions[key]
				c = credibility_interval_of_invCDF(invCDF, level)
				print(c)
				if key=="true":
					a.axvline(c[0], color="k")
					a.axvline(c[1], color="k")
				else:
					a.plot(c, [y, y], label=key)
				b.plot(xplot, CDF(xplot), label=key)
				y += 0.1
			a.legend()
			a.set_ylim(-0.1,2)
			b.legend()
			plt.show()
		else:
			intervals = True
			if intervals:
				invCDF = ifunctions["true"]
				true_c = credibility_interval_of_invCDF(invCDF, level)
				for key in scores.keys():
					this_c = credibility_interval_of_invCDF(ifunctions[key], level)
					scores[key].append(np.abs(this_c[0]-true_c[0])/scale+np.abs(this_c[1]-true_c[1])/scale)
			else:
				tCDF = functions["true"]
				a = np.random.uniform(np.min(data), np.max(data))
				b = np.random.uniform(np.min(data), np.max(data))
				t = tCDF(b)-tCDF(a)
				for key in scores.keys():
					CDF = functions[key]
					c = CDF(b)-CDF(a)
				scores.append(np.abs(c-t))

for key in scores.keys():
	plt.hist(scores[key], label=key, histtype="step", bins=100)
plt.show()



import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate a dataset
np.random.seed(3)
data = np.random.randn(30)
weights = np.random.rand(30)
i = np.argsort(data)
data = data[i]
weights = weights[i]/weights.sum()

# Symmetric empirical CDF
w_ = np.concatenate([[0.],weights[1:]+weights[:-1]]).cumsum()
w_/=w_[-1]
sym = interp1d(data, w_)

# Left hand empirical CDF
lh = interp1d(data, np.cumsum(weights))

# Right hand empirical CDF
rh = interp1d(data, 1-np.cumsum(weights[::-1])[::-1])

W, SYM, LH, RH, T = [], [], [], [], []
for _ in range(10000):
    a, b = np.sort(np.random.uniform(data.min(),data.max(),size=(2)))
    w = weights[(data < b) & (data > a)].sum()
    W.append(w)
    T.append(norm.cdf(b)-norm.cdf(a))
    SYM.append(sym(b)-sym(a))
    LH.append(lh(b)-lh(a))
    RH.append(rh(b)-rh(a))

W, SYM, LH, RH, T = np.array([W, SYM, LH, RH, T])
        
plt.hist(W-T,bins=100)
plt.hist(SYM-T,bins=100)
plt.hist(LH-T,bins=100,alpha=0.8)
plt.hist(RH-T,bins=100,alpha=0.8)
plt.tight_layout()
plt.savefig('hist.png')