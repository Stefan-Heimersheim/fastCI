import matplotlib.pyplot as plt
import scipy.stats as sst
import scipy.interpolate as sip
import scipy.optimize as sop
import numpy as np
cdefault = plt.rcParams['axes.prop_cycle'].by_key()['color']


#Code
def credibility_interval(samples, weights=None, level=0.68):
    assert level<1, "Level >= 1!"
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples)[order]
    weights = np.array(weights)[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    cumsum = np.cumsum(weights)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invCDF = sip.interp1d(CDF, S)
    # Find smallest interval
    distance = lambda Y, level=level: invCDF(Y+level)-invCDF(Y)
    res = sop.minimize_scalar(distance, bounds=(0, 1-level), method="Bounded")
    #res = sop.minimize(distance, (1-level)/2, bounds=[(0,1-level)], method="Nelder-Mead")
    return np.array([invCDF(res.x), invCDF(res.x+level)])


# Generate data, small example
np.random.seed(1)
data = np.random.normal(loc=7, scale=3, size=10)
weights = np.ones(len(data))/len(data)

# Plot data
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 3))
ax.set_title("Histogram")
ax.set_ylabel("Number of samples")
ax.set_xlabel("Variable x")
for d in data:
	ax.set_ylim(0, 1.5)
	ax.axvline(d, ymin=0, ymax=2/3, color="black", lw=2)
	ax.scatter(d, 0.5)

# Intervals
data.sort()
ax.plot(data[[1,6]], [1.1]*2, lw=3, color=cdefault[0])
ax.text(np.mean(data[[1,6]]), 1.12, 'Containing 60% of the data',
	verticalalignment='bottom', horizontalalignment='center',
	color=cdefault[0], fontsize=8)
ax.plot(data[[1,7]], [1.3]*2, lw=3, color=cdefault[1])
ax.text(np.mean(data[[1,7]]), 1.32, 'Containing 70% of the data',
	verticalalignment='bottom', horizontalalignment='center',
	color=cdefault[1], fontsize=8)

# Uncertainty arrows
a = data[6]
b = data[7]
c = data[1]
ax.arrow(a+(b-a)/2, 1.2, (b-a)/2, 0, length_includes_head=True, color=cdefault[2], head_width=0.025, head_length=0.2, lw=2, zorder=2)
ax.arrow(a+(b-a)/2, 1.2, (a-b)/2, 0, length_includes_head=True, color=cdefault[2], head_width=0.025, head_length=0.2, lw=2, zorder=2)
ax.arrow(a, 0.8, 0.5, 0, length_includes_head=True, color=cdefault[3], head_width=0.025, head_length=0.2, lw=2, zorder=-2)
ax.arrow(a, 0.8, -0.5, 0, length_includes_head=True, color=cdefault[3], head_width=0.025, head_length=0.2, lw=2, zorder=-2)
ax.arrow(b, 0.8, 0.5, 0, length_includes_head=True, color=cdefault[3], head_width=0.025, head_length=0.2, lw=2, zorder=-2)
ax.arrow(b, 0.8, -0.5, 0, length_includes_head=True, color=cdefault[3], head_width=0.025, head_length=0.2, lw=2, zorder=-2)
ax.arrow(c, 0.8, 0.5, 0, length_includes_head=True, color=cdefault[3], head_width=0.025, head_length=0.2, lw=2, zorder=-2)
ax.arrow(c, 0.8, -0.5, 0, length_includes_head=True, color=cdefault[3], head_width=0.025, head_length=0.2, lw=2, zorder=-2)

plt.savefig("illustrations/low_sample_example.png", dpi=600)
plt.close()

# Reconsider common-sense
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 3))
ax.set_title("Histogram")
ax.set_ylabel("Number of samples")
ax.set_xlabel("Variable x")
for d in data:
    ax.set_ylim(0, 1.5)
    ax.axvline(d, ymin=0, ymax=1/1.5, color="black", lw=2)
    ax.scatter(d, 0.5)

ax.plot(data[[1,7]], [1.25]*2, lw=3, color=cdefault[1])
ax.plot([data[0]+0.1, data[8]-0.1], [1.3]*2, lw=3, color=cdefault[1])
ax.plot([data[1], data[8]], [1.15]*2, lw=3, color=cdefault[2])
ax.text(np.mean(data[[1,7]]), 1.32, 'Containing 70% of the data',
    verticalalignment='bottom', horizontalalignment='center',
    color=cdefault[1], fontsize=8)

plt.savefig("illustrations/low_sample_example_v2.png", dpi=600)
plt.close()



data.sort()
# Note: Data and weights are already sorted here, and weights are normalized
cumsum = np.cumsum(weights)
CDF_x = np.array([np.min(data), *data, np.max(data)])
CDF_y = np.array([0, *cumsum, 1])
CDF = sip.interp1d(CDF_x, CDF_y)
invCDF = sip.interp1d(CDF_y, CDF_x)



# Plot CDF
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 3))
ax.plot(CDF_x, CDF_y, label="Weights from 0 to x", color=cdefault[2])
ax.plot(CDF_x[::-1], 1-CDF_y, label="1 - Weight from 1 to x", color=cdefault[3])
ax.grid(color="grey", lw=0.2)
ax.scatter(CDF_x, CDF_y, color="black", s=10)
ax.scatter(CDF_x[::-1], 1-CDF_y, color="black", s=10)
ax.legend()
ax.set_ylabel("CDF")
ax.set_xlabel("Variable x")
plt.savefig("illustrations/CDF_illustration.png", dpi=600)
plt.close()

# CDF y-level scanning
a,b = credibility_interval(data)
Y = CDF(a)
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 3))
ax.plot(CDF_x, CDF_y, label="Weights from 0 to x", color=cdefault[2])
#ax.plot(CDF_x[::-1], 1-CDF_y, label="1 - Weight from 1 to x", color=cdefault[3])
ax.grid(color="grey", lw=0.2)
ax.scatter(CDF_x, CDF_y, color="black", s=10)
#ax.scatter(CDF_x[::-1], 1-CDF_y, color="black", s=10)
ax.arrow(a+(b-a)/2, 0.12, (b-a)/2, 0, length_includes_head=True, color=cdefault[3], head_width=0.025, head_length=0.2, lw=2, zorder=2)
ax.arrow(a+(b-a)/2, 0.12, (a-b)/2, 0, length_includes_head=True, color=cdefault[3], head_width=0.025, head_length=0.2, lw=2, zorder=2)
ax.text(0.1, Y-0.06, 'Y',
    verticalalignment='bottom', horizontalalignment='left',
    color=cdefault[3], fontsize=8)
ax.text(0.1, Y+0.62, 'Y+0.68',
    verticalalignment='bottom', horizontalalignment='left',
    color=cdefault[3], fontsize=8)
ax.text(a+(b-a)/2, 0.04, 'Interval containing "68%" of the samples',
    verticalalignment='bottom', horizontalalignment='center',
    color=cdefault[3], fontsize=8)
ax.axhline(Y, color=cdefault[3], ls="--")
ax.axhline(Y+0.68, color=cdefault[3], ls="--")
ax.plot([invCDF(Y), invCDF(Y)], [0, Y], color=cdefault[3])
ax.plot([invCDF(Y+0.68), invCDF(Y+0.68)], [0, Y+0.68], color=cdefault[3])
ax.legend()
ax.set_ylabel("CDF")
ax.set_xlabel("Variable x")
plt.savefig("illustrations/CDF_distances.png", dpi=600)
plt.close()


# Reconsider common-sense
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 3))
ax.set_title("Histogram")
ax.set_ylabel("Number of samples")
ax.set_xlabel("Variable x")
for d in data:
    ax.set_ylim(0, 1.7)
    ax.axvline(d, ymin=0, ymax=1/1.7, color="black", lw=2)
    ax.scatter(d, 0.5)

ax.plot(data[[1,7]], [1.25]*2, lw=3, color=cdefault[1])
ax.plot([data[0]+0.1, data[8]-0.1], [1.3]*2, lw=3, color=cdefault[1])
ax.text(np.mean(data[[1,7]]), 1.32, 'Containing 70% of the data',
    verticalalignment='bottom', horizontalalignment='center',
    color=cdefault[1], fontsize=8)
ax.plot([data[1], data[8]], [1.15]*2, lw=3, color=cdefault[2])

ax.plot([a, b], [1.45]*2, lw=3, color=cdefault[3])
ax.text(np.mean(data[[1,7]]), 1.47, '68% interval from CDF-interpolation',
    verticalalignment='bottom', horizontalalignment='center',
    color=cdefault[3], fontsize=8)
plt.savefig("illustrations/low_sample_again.png", dpi=600)
plt.close()







# Generate data, large example
np.random.seed(2)
data = np.random.normal(loc=7, scale=3, size=1000)


# Plot data
fig, ax = plt.subplot_mosaic([['top', 'top'], ['left', 'right']],
                              constrained_layout=True, figsize=(6, 5))
ax["top"].hist(data, color="black", bins=50, range=[-4, 20], label="Samples")
ax["top"].set_title("Histogram")
ax["top"].set_ylabel("Number of samples")
ax["top"].set_ylim(0, 80)
ax["top"].set_xlim(-4, 20)
xplot=np.linspace(-10, 30, 1000)
ax["top"].plot(xplot, 1000/(50/24)*sst.norm.pdf(xplot, loc=7, scale=3), color=cdefault[0], lw=2, label="True distribution")
ax["top"].axvline(7-3, ymax=1000/(50/24)*sst.norm.pdf(7-3, loc=7, scale=3)/80, lw=2, ls=":", color=cdefault[0])
ax["top"].axvline(7+3, ymax=1000/(50/24)*sst.norm.pdf(7+3, loc=7, scale=3)/80, lw=2, ls=":", color=cdefault[0])
c1,c2 = credibility_interval(data)
ax["top"].axvline(c1, ymax=1, lw=2, ls="-", color=cdefault[1], label="68% of samples")
ax["top"].axvline(c2, ymax=1, lw=2, ls="-", color=cdefault[1])
ax["top"].legend()
# Zoom in
ax["left"].hist(data, color="black", bins=100, range=[3, 5])
ax["left"].set_xlim(3, 5)
ax["left"].axvline(c1, ymax=1, lw=2, ls="-", color=cdefault[1])
ax["right"].hist(data, color="black", bins=100, range=[9, 11])
ax["right"].set_xlim(9, 11)
ax["right"].axvline(c2, ymax=1, lw=2, ls="-", color=cdefault[1])

plt.savefig("illustrations/high_sample_example.png", dpi=600)
plt.close()

# Chi2 distribution example of HDPI vs quantiles
xplot = np.linspace(0,20,100000)
yplot = sst.chi2.pdf(xplot, 5)
cplot = sst.chi2.cdf(xplot, 5)

def waterline(level):
	f = lambda x: sst.chi2.pdf(x, 5)-level
	k = 5
	median = k*(1-2/9/k)**3
	lower = sop.bisect(f, 0, median)
	upper = sop.bisect(f, median, 100)
	return lower, upper, sst.chi2.cdf(upper, 5)-sst.chi2.cdf(lower, 5)

w = lambda l: waterline(l)[2]-0.68
w0 = sop.bisect(w, 0.001, 0.13)
a,b,_ = waterline(w0)
level = sst.chi2.pdf(a,5)

fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 2.5))
ax.set_title('Highest probability density interval ("waterline")')
ax.axvline(a, ymax=level/0.16, color=cdefault[1], lw=2)
ax.axvline(b, ymax=level/0.16, color=cdefault[1], lw=2)
ax.fill_between(np.linspace(a,b,1000), sst.chi2.pdf(np.linspace(a,b,1000), 5), color="grey")
#ax.plot([a,b], [level, level], color=cdefault[1], lw=2)
ax.plot([0,20], [level, level], lw=2, color=cdefault[2])
ax.text(10, level+0.005, "Waterline", color=cdefault[2])
ax.plot(xplot, yplot, color=cdefault[0], lw=2)
ax.set_xlim(0,20)
ax.set_ylabel("Probability density")
ax.set_ylim(0, 0.16)
ax.text((a+b)/2,0.01,"68% of\nvolume",color=cdefault[1], horizontalalignment='center')
plt.savefig("illustrations/highest-probability-density-interval.png", dpi=600)
plt.close()

def mean_interval(sigma):
    mean = 5
    return sst.chi2.cdf(mean+sigma, 5) - sst.chi2.cdf(mean-sigma, 5)
m = lambda sigma: mean_interval(sigma)-0.68
sigma0 = sop.bisect(m, 0.001, 4)
a = 5-sigma0
b = 5+sigma0

fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 2.5))
ax.set_title('Interval centered around the mean')
ax.axvline(a, ymax=sst.chi2.pdf(a, 5)/0.16, color=cdefault[1], lw=2)
ax.axvline(b, ymax=sst.chi2.pdf(b, 5)/0.16, color=cdefault[1], lw=2)
ax.fill_between(np.linspace(a,b,1000), sst.chi2.pdf(np.linspace(a,b,1000), 5), color="grey")
ax.plot(xplot, yplot, color=cdefault[0], lw=2)
ax.set_xlim(0,20)
ax.set_ylabel("Probability density")
ax.set_ylim(0, 0.16)
ax.text((a+b)/2,0.01,"68% of\nvolume",color=cdefault[1], horizontalalignment='center')
ax.axvline(5, color=cdefault[2])
plt.text(5.01, 0.14, "Mean", color=cdefault[2])
plt.savefig("illustrations/mean-centered-interval.png", dpi=600)
plt.close()

invCDF = sip.interp1d(cplot, xplot)
a = invCDF(0.32/2)
b = invCDF(0.68+0.32/2)
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 2.5))
ax.set_title('Equal-tailed interval')
ax.text(0.7,0.01,"16%",color=cdefault[2])
ax.text(9,0.01,"16%",color=cdefault[2])
ax.axvline(a, ymax=sst.chi2.pdf(a,5)/0.16, color=cdefault[1], lw=2)
ax.axvline(b, ymax=sst.chi2.pdf(b,5)/0.16, color=cdefault[1], lw=2)
ax.fill_between(np.linspace(a,b,1000), sst.chi2.pdf(np.linspace(a,b,1000), 5), color="grey")
ax.plot(xplot, yplot, color=cdefault[0], lw=2)
ax.set_xlim(0,20)
ax.set_ylabel("Probability density")
ax.set_ylim(0, 0.16)
ax.text((a+b)/2,0.01,"68% of\nvolume",color=cdefault[1], horizontalalignment='center')
plt.savefig("illustrations/equal-tailed-interval.png", dpi=600)
plt.close()
