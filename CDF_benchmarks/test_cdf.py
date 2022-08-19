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
			# Symmetric empirical CDF
			w_ = np.concatenate([[0.],weights[1:]+weights[:-1]]).cumsum()
			w_/=w_[-1]
			sym = sip.interp1d(data, w_)
			# Left hand empirical CDF
			lh = sip.interp1d(data, np.cumsum(weights))
			# Right hand empirical CDF
			rh = sip.interp1d(data, 1-np.cumsum(weights[::-1])[::-1])
			# Weighted C=1/2
			wpct = sip.interp1d(data, (weights.cumsum()-weights/2)/weights.sum())
			# Discrete
			dcdf = lambda x: weights[(data < x)].sum()
			# Anesthetic
			anest = anesthetic.utils.cdf(data, w=weights)
			a, b = np.sort(np.random.uniform(data.min(),data.max(),size=(2)))
			true_cdf.append( sst.norm.cdf(b)-sst.norm.cdf(a) )
			discrete_cdf.append( dcdf(b)-dcdf(a) )
			#discrete_cdf.append( weights[(data < b) & (data > a)].sum() )
			#assert np.allclose( weights[(data < b) & (data > a)].sum(), dcdf(b)-dcdf(a), rtol=0, atol=1e-8)
			sym_cdf.append( sym(b)-sym(a) )
			wpct_cdf.append( wpct(b)-wpct(a) )
			anest_cdf.append( anest(b)-anest(a) )
			lh_cdf.append( lh(b)-lh(a) )
			rh_cdf.append( rh(b)-rh(a) )
		true_cdf = np.array(true_cdf);
		discrete_cdf = np.array(discrete_cdf);
		sym_cdf = np.array(sym_cdf);
		wpct_cdf = np.array(wpct_cdf);
		anest_cdf = np.array(anest_cdf);
		lh_cdf = np.array(lh_cdf);
		rh_cdf = np.array(rh_cdf);
		
		fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10,6), constrained_layout=True)
		ax1.plot([0,1], [0,1], label="True", color="gray")
		if not random_weights:
			assert np.allclose(lh_cdf, rh_cdf, atol=1e-8, rtol=0)
			assert np.allclose(lh_cdf, wpct_cdf, atol=1e-8, rtol=0)
			assert np.allclose(sym_cdf, anest_cdf, atol=1e-8, rtol=0)
			ax1.scatter(true_cdf, wpct_cdf, label="lh/rh/wpct_cdf")
			ax2.hist(wpct_cdf-true_cdf, histtype="step", bins=100, label="lh/rh/wpct_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(wpct_cdf-true_cdf), np.std(wpct_cdf-true_cdf)))
			ax1.scatter(true_cdf, anest_cdf, label="sym/anest_cdf", marker="x", alpha=0.5)
			ax2.hist(anest_cdf-true_cdf, histtype="step", bins=100, label="sym/anest_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(anest_cdf-true_cdf), np.std(anest_cdf-true_cdf)))
			ax1.scatter(true_cdf, discrete_cdf, label="discrete_cdf", marker="+", alpha=0.5)
			ax2.hist(discrete_cdf-true_cdf, histtype="step", bins=100, label="discrete_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(discrete_cdf-true_cdf), np.std(discrete_cdf-true_cdf)))
		else:
			ax1.scatter(true_cdf, wpct_cdf, label="wpct_cdf", marker="o", alpha=0.5)
			ax1.scatter(true_cdf, anest_cdf, label="anest_cdf", marker="x")
			ax1.scatter(true_cdf, discrete_cdf, label="discrete_cdf", marker="+")
			ax2.hist(wpct_cdf-true_cdf, histtype="step", bins=100, label="wpct_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(wpct_cdf-true_cdf), np.std(wpct_cdf-true_cdf)))
			ax2.hist(anest_cdf-true_cdf, histtype="step", bins=100, label="anest_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(anest_cdf-true_cdf), np.std(anest_cdf-true_cdf)))
			ax2.hist(discrete_cdf-true_cdf, histtype="step", bins=100, label="discrete_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(discrete_cdf-true_cdf), np.std(discrete_cdf-true_cdf)))
			ax1.scatter(true_cdf, sym_cdf, label="sym_cdf", marker="*")
			ax1.scatter(true_cdf, lh_cdf, label="lh_cdf", marker="1")
			ax1.scatter(true_cdf, rh_cdf, label="rh_cdf", marker="2")
			ax2.hist(sym_cdf-true_cdf, histtype="step", bins=100, label="sym_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(sym_cdf-true_cdf), np.std(sym_cdf-true_cdf)))
			ax2.hist(lh_cdf-true_cdf, histtype="step", bins=100, label="lh_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(lh_cdf-true_cdf), np.std(lh_cdf-true_cdf)))
			ax2.hist(rh_cdf-true_cdf, histtype="step", bins=100, label="rh_cdf: mean={0:.2e}, std={1:.2e}".format(np.mean(rh_cdf-true_cdf), np.std(rh_cdf-true_cdf)))

		ax1.legend()
		ax2.legend()
		title = "Data size N="+str(N_data)
		if random_weights:
			title += " and weights randomized"
		else:
			title += " and weights uniform"
		fig.suptitle(title)
		ax1.set_title("Compare CDF(b) - CDF(a)")
		ax2.set_title("Compare DeltaCDF(method)-DeltaCDF(true)")
		ax1.set_xlabel("True Delta CDF")
		ax2.set_xlabel("Method DeltaCDF - True DeltaCDF")
		ax1.set_ylabel("Method Delta CDF")
		plt.savefig("CDF_comparison"+str(N_data)+str(random_weights)+".png", dpi=600)
		plt.close()


scores = {}
cases = ["wpct", "anest", "discrete", "sym", "lh", "rh"]
for c in cases:
	scores[c] = []
sizes = np.geomspace(10,1e5,8)
for random_weights in [True]:
	for size in sizes:
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
			# Symmetric empirical CDF
			w_ = np.concatenate([[0.],weights[1:]+weights[:-1]]).cumsum()
			w_/=w_[-1]
			sym = sip.interp1d(data, w_)
			# Left hand empirical CDF
			lh = sip.interp1d(data, np.cumsum(weights))
			# Right hand empirical CDF
			rh = sip.interp1d(data, 1-np.cumsum(weights[::-1])[::-1])
			# Weighted C=1/2
			wpct = sip.interp1d(data, (weights.cumsum()-weights/2)/weights.sum())
			# Discrete
			dcdf = lambda x: weights[(data < x)].sum()
			# Anesthetic
			anest = anesthetic.utils.cdf(data, w=weights)
			a, b = np.sort(np.random.uniform(data.min(),data.max(),size=(2)))
			true_cdf.append( sst.norm.cdf(b)-sst.norm.cdf(a) )
			discrete_cdf.append( dcdf(b)-dcdf(a) )
			#discrete_cdf.append( weights[(data < b) & (data > a)].sum() )
			#assert np.allclose( weights[(data < b) & (data > a)].sum(), dcdf(b)-dcdf(a), rtol=0, atol=1e-8)
			sym_cdf.append( sym(b)-sym(a) )
			wpct_cdf.append( wpct(b)-wpct(a) )
			anest_cdf.append( anest(b)-anest(a) )
			lh_cdf.append( lh(b)-lh(a) )
			rh_cdf.append( rh(b)-rh(a) )
		true_cdf = np.array(true_cdf);
		discrete_cdf = np.array(discrete_cdf);
		sym_cdf = np.array(sym_cdf);
		wpct_cdf = np.array(wpct_cdf);
		anest_cdf = np.array(anest_cdf);
		lh_cdf = np.array(lh_cdf);
		rh_cdf = np.array(rh_cdf);
		scores["discrete"].append([np.mean(discrete_cdf-true_cdf), np.std(discrete_cdf-true_cdf)])
		scores["sym"].append([np.mean(sym_cdf-true_cdf), np.std(sym_cdf-true_cdf)])
		scores["wpct"].append([np.mean(wpct_cdf-true_cdf), np.std(wpct_cdf-true_cdf)])
		scores["anest"].append([np.mean(anest_cdf-true_cdf), np.std(anest_cdf-true_cdf)])
		scores["lh"].append([np.mean(lh_cdf-true_cdf), np.std(lh_cdf-true_cdf)])
		scores["rh"].append([np.mean(rh_cdf-true_cdf), np.std(rh_cdf-true_cdf)])

for key in cases:
	scores[key] = np.array(scores[key]).T


for key in cases:
	print(np.shape(scores[key]))

for key in cases:
	plt.errorbar(sizes, scores[key][0], yerr=scores[key][1], capsize=20, elinewidth=3, label=key)

plt.axhline(0, color="k")
plt.xscale("log")
plt.yscale("symlog", linthresh=3e-5)
plt.xlabel("N_data")
plt.ylabel("Mean difference to true CDF, and spread (standard deviation)")
plt.legend()
plt.savefig("function_of_Ndata.png", dpi=600)
plt.show()