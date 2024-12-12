#%% imports

import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression

import matplotlib as mpl
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% helper functions

def load_results(infile):
    return pickle.load(open(infile, 'rb'))

def group_results_by_key(Results):
    R = {}
    for results in Results:
        for result in results:
            if result['key'] not in R:
                R[result['key']] = []
            R[result['key']].append(result)
    return R

def get_all_matching_results(pattern, check_args=True):
    # find all results data where filename matches pattern
    infiles = glob.glob(pattern)
    results = [load_results(infile) for infile in infiles]
    
    keys_to_check = ['ntraining_trials', 'fixed_ntrials_per_episode', 'fixed_episode_length', 'hidden_size', 'gamma', 'lr', 'optimizer']
    args = [tuple(res['args'][k] for k in keys_to_check if k in res['args']) for res in results]
    if check_args:
        # ensure args are all the same across matching results
        if len(set(args)) > 1:
            print(f'ERROR: args are not all the same across matching results: {args}')
            return [], None
        else:
            args = dict((x,y) for x,y in results[0]['args'].items() if x in keys_to_check)

    return group_results_by_key([res['results'] for res in results]), args

#%% load results

# results, args = get_all_matching_results('data/temporal-scaling_40277*.pickle')
# results2, args2 = get_all_matching_results('data/temporal-scaling_404*.pickle')
# ress = [results, results2]

nm = 'temporal-scaling_52313'
nm = 'temporal-scaling_54826' # 30k
nm = 'temporal-scaling_56010' # 50k
# nm = 'temporal-scaling_56142' # gamma=0.99

results, args = get_all_matching_results(f'data/{nm}*.pickle')
ress = [results]

# burke
results, args = get_all_matching_results('data/temporal-scaling_burke_40467794*.pickle', check_args=False)
ress = [results]

# results where all sessions have fixed duration:
# temporal-scaling_52587496_14_2024-10-22_11-31-43.pickle
# results, args = get_all_matching_results('data/temporal-scaling_52587*10-22_11*.pickle', check_args=False)
# ress = [results]

# results where all sessions have fixed duration:
# results, args = get_all_matching_results('data/temporal-scaling_525024*.pickle')
# ress = [results]

# i forget
# results, args = get_all_matching_results('data/temporal-scaling_3856*.pickle')
# ress = [results]

print(args)
results0 = {}
for res in ress:
    for key, items in res.items():
        if key not in results0:
            results0[key] = []
        results0[key].extend(items)
results = results0

#%% recreate the key Gallistel & Gibbons plots

thresh = 0.15 # to define time to learning
# thresh = 0.0
xnse = 0.1 # noise added for jitter
clrs = {'I/T fixed': 'blue', 'I fixed': 'orange', 'other': 'red'}

counts = {'I': {}, 'I/T': {}}
for key, items in results.items():
    I, T = key
    IoT = I/T
    # IoT = (I+T)/T
    if I not in counts['I']:
        counts['I'][I] = 0
    if IoT not in counts['I/T']:
        counts['I/T'][IoT] = 0
    counts['I'][I] += len(items)
    counts['I/T'][IoT] += len(items)
fixedI = max(counts['I'].items(), key=lambda x: x[1])[0]
fixedIoT = max(counts['I/T'].items(), key=lambda x: x[1])[0]
print(f'{fixedI=}, {fixedIoT=}')

PtsT = {}
PtsIoT = []
PtsAll = []
failedCounts = {}
for key, items in results.items():
    I, T = key
    IoT = I/T
    # IoT = (I+T)/T
    # print(I, T, len(items))

    # valKey = 'CS_resps'
    valKey = 'RPE_resps'
    # ys = np.vstack([item['RPE_resps'][:,1:].mean(axis=1) for item in items])
    ys = np.vstack([item[valKey].flatten() for item in items])

    if IoT == fixedIoT:
        grp = 'I/T fixed'
    elif I == fixedI:
        grp = 'I fixed'
        # grp = 'other'
    else:
        grp = 'other'
    if grp not in PtsT:
        PtsT[grp] = []

    # thresh = 0.1 * (0.98 ** T)

    for j in range(ys.shape[0]):
        ts = np.where(ys[j,:] > thresh)[0]
        if len(ts) == 0 or ts[0] == 0:
            if key not in failedCounts:
                failedCounts[key] = 0
            failedCounts[key] += 1
            continue
        y = ts[0]
        # print(I, T, IoT, y, grp)
        PtsT[grp].append((T, y))
        PtsIoT.append((IoT, y))
        PtsAll.append((I, T, IoT, y))
print(f'{failedCounts=}')

fontsize = 12
ncols = 2; nrows = 2
plt.figure(figsize=(3*ncols,3*nrows)); c = 1
plt.subplot(nrows,ncols,c); c += 1
xsas = []
for grp, pts in PtsT.items():
    if grp not in clrs:
        continue
    print(grp, 'nsamples=', len(pts))
    pts = np.vstack(pts)
    xsa = np.unique(pts[:,0])
    mus = []
    for x in xsa:
        ys = pts[pts[:,0] == x,1]
        mu = np.median(ys); lb = np.percentile(ys, 25); ub = np.percentile(ys, 75)
        mu = np.mean(ys); se = np.std(ys)/np.sqrt(len(ys)); lb = mu-se; ub = mu+se
        mus.append(mu)

        xs = x * np.ones(len(ys)) + xnse*np.random.randn(len(ys))
        plt.scatter(xs, ys, s=2, c=clrs[grp], alpha=0.3, zorder=-2)
        plt.plot(x, mu, 'o', color=clrs[grp], markeredgecolor='k')
        plt.plot(x * np.ones(2), [lb, ub], 'k-', alpha=0.5, zorder=-1)
    lbl = f'{grp}={fixedIoT}' if grp == 'I/T fixed' else f'{grp}={fixedI}'
    lbl = lbl.replace(' fixed', '')
    lbl = grp
    plt.plot(xsa, mus, '-', c=clrs[grp], zorder=-1, label=lbl)
    xsas.append(xsa)
xsa = np.hstack(xsas)
xsa = np.arange(0, max(xsa)+1, 5)
plt.xticks(ticks=xsa, labels=xsa)#, fontsize=fontsize)

# plt.yscale('log')
# ysa = [2500, 5000, 10000]
# plt.yticks(ticks=ysa, labels=ysa)

plt.xlabel('T = Delay of Reinforcement', fontsize=fontsize)
plt.ylabel('Reinforcements to Acquisition', fontsize=fontsize)
plt.legend(fontsize=fontsize)

plt.subplot(nrows,ncols,c); c += 1
# pts = np.vstack(PtsIoT)
pts = np.vstack(PtsAll)
# gs = pts[:,0].astype(int)
gs = pts[:,1].astype(int)
grps = np.unique(gs)
xss = pts[:,2]#.astype(int)
xsa = np.unique(xss)
all_mus = []
for grp in grps:
    ix = (gs == grp)
    mus = []
    xsas = []
    for x in xsa:
        ixc = ix & (xss == x)
        # if ixc.sum() < 2: # need at least two to connect lines
        #     continue
        ys = pts[ixc,-1]
        mu = np.median(ys); lb = np.percentile(ys, 25); ub = np.percentile(ys, 75)
        # mu = np.mean(ys); se = np.std(ys)/np.sqrt(len(ys)); lb = mu-se; ub = mu+se
        mus.append(mu)
        xsas.append(x)

        xs = x * np.ones(len(ys)) + xnse*np.random.randn(len(ys))
        # plt.scatter(xs, ys, s=2, c='k', alpha=0.2, zorder=-2)
        # plt.plot(x, mu, 'o', color='k', markeredgecolor='k')
        plt.plot(x * np.ones(2), [lb, ub], 'k-', alpha=0.2, zorder=-1)
    # if len(mus) < 2:
    #     continue
    # print(grp, xsas, mus)
    # plt.plot(xsas, mus, 'o', zorder=-1, label=f'T={grp}')
    all_mus.append((xsas, mus))
xsc = np.hstack([x for x,y in all_mus])
ysc = np.hstack([y for x,y in all_mus])
ix = np.argsort(xsc); xsc = xsc[ix]; ysc = ysc[ix]
plt.plot(xsc[ix], ysc[ix], 'o', color='k', markersize=3, zorder=0)

# fit and show linear fit
xsc = np.log(xsc); ysc = np.log(ysc)
if len(xsc) > 1:
    clf = LinearRegression(); mdl = clf.fit(xsc[:,None], ysc[:,None])
    plt.plot(np.exp(xsc), np.exp(mdl.predict(xsc[:,None])), 'k-', zorder=-2)

plt.xlabel('I/T', fontsize=fontsize)
plt.ylabel('Reinforcements to Acquisition', fontsize=fontsize)
plt.xscale('log')
plt.yscale('log')
xsa = np.unique(np.array(xsa).astype(int)); xsa = xsa[xsa > 0]
xsa = xsa[::2]
ysa = [2500, 5000, 10000]
plt.yticks(ticks=ysa, labels=ysa)
plt.xticks(ticks=xsa, labels=xsa, rotation=90)
# plt.xlim([min(xsa), max(plt.xlim())])
plt.minorticks_off()
# plt.ylim([1000, 15000])
# yticks = [700,500,300]
# plt.yticks(ticks=yticks, labels=yticks)
# plt.legend(fontsize=6)

plt.subplot(nrows,ncols,c); c += 1
pts = np.vstack(PtsAll)
Ts = pts[:,1].astype(int)
Ts_all = np.unique(Ts)
clrs = plt.get_cmap('Reds', len(Ts_all))
xsas = []
for i, T in enumerate(Ts_all):
    cpts = pts[Ts == T,:]
    xs = cpts[:,0]
    ys = cpts[:,-1]
    xsa = np.unique(xs)
    ysa = np.array([np.mean(ys[xs == x]) for x in xsa])

    ix = np.argsort(xsa)
    xsc = xsa[ix]
    ysc = ysa[ix]
    xsas.extend(xsc)
    if len(xsc) == 1:
        continue
    
    plt.plot(xsc, ysc, '.-', color=clrs(i), label=f'{T=}')
plt.xlabel('I = Intertrial Interval')
plt.ylabel('Reinforcements to Acquisition')
plt.xscale('log')
plt.yscale('log')
xsa = np.unique(xsas).astype(int)[::2]
plt.xticks(ticks=xsa, labels=xsa, rotation=90)
plt.minorticks_off()
plt.legend(fontsize=8)

plt.tight_layout()

#%% visualize individual models

showLoss = False
showTrials = False # if False, shows episodes
separateSubplots = False # False for Burke

ncols = 2; nrows = 1 # for Burke
ncols = 3; nrows = 1 # for Burke
# ncols = 6; nrows = 2
thresh = 0.17

clrs = ['#d4bb51', '#916bad']*len(results)

plt.figure(figsize=(3*ncols,3*nrows)); c = 1
keys = sorted(results.keys())
for j, key in enumerate(keys):
    items = results[key]
    I, T = key
    IoT = I/T
    # print(I, T, IoT)
    
    # if T < 60:
    #     continue
    # if I != fixedI:
    #     continue
    # if IoT != fixedIoT:
    #     continue

    # thresh = 0.5 * (0.98 ** T)
    # thresh = 0.17

    if showLoss:
        ys = np.vstack([item['scores'] for item in items])
    else:
        valKey = 'RPE_resps'
        if showTrials:
            ys = np.vstack([item[valKey].flatten() for item in items])
        else:
            ys = np.vstack([item[valKey].mean(axis=1) for item in items])

        # ys = np.vstack([item['RPE_resps'][:,1:].mean(axis=1) for item in items])
        vs = [np.where(y > thresh)[0] for y in ys]
        vs = [v[0] for v in vs if len(v)]
        plt.subplot(nrows,ncols,nrows*ncols)
        plt.plot([T]*len(vs), vs, '.', label=f'{I=}, {T=}', color=clrs[j] if j < len(clrs) else None)
        plt.xlabel('T = Delay of Reinforcement')
        plt.ylabel('{} to Acquistion'.format('Reinforcements' if showTrials else 'Episodes'))
        plt.legend(fontsize=8)
    # print(key, ys.shape)

    if not showLoss and c == nrows*ncols:
        continue
    plt.subplot(nrows,ncols,c)
    if separateSubplots:
        c += 1
        plt.title(f'{I=}, {T=}', fontsize=8)
        plt.plot(ys.T)
    else:
        mus = ys.mean(axis=0) # show average
        ses = ys.std(axis=0) / np.sqrt(ys.shape[0])
        xs = np.arange(len(mus))
        plt.plot(xs, mus, color=clrs[j] if j < len(clrs) else None)
        plt.fill_between(xs, mus-ses, mus+ses, color=clrs[j] if j < len(clrs) else None, zorder=-1, alpha=0.2)
    # [plt.plot(v, thresh, '.') for v in vs]
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if c == 1:
        plt.xlabel('# Reinforcements' if showTrials else '# Episodes', fontsize=12)
        plt.ylabel('Loss' if showLoss else 'RPE', fontsize=12)
    if showLoss:
        plt.xlabel('# Episodes', fontsize=12)
    else:
        plt.plot(plt.xlim(), thresh*np.ones(2), 'k--', zorder=-1, alpha=0.5)
        plt.ylim([-0.1, 1.0])
    # plt.xlim([0, 1300]); plt.ylim([-0.2, 1.0])

plt.tight_layout()

#%%

# results, args = get_all_matching_results('data/temporal-scaling_burke_40278*.pickle')
# results, args = get_all_matching_results('data/temporal-scaling_burke_403*.pickle')

results, args = get_all_matching_results('data/temporal-scaling_burke_4046*.pickle')

# %%
