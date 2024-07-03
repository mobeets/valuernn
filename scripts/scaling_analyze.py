#%% imports

import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    if not check_args:
        return results

    # ensure args are all the same across matching results
    keys_to_check = ['ntraining_trials', 'fixed_ntrials_per_episode', 'fixed_episode_length', 'hidden_size', 'gamma', 'lr', 'optimizer']
    args = [tuple(res['args'][k] for k in keys_to_check if k in res['args']) for res in results]
    if len(set(args)) > 1:
        print(f'ERROR: args are not all the same across matching results: {args}')
        return [], None
    else:
        args = dict((x,y) for x,y in results[0]['args'].items() if x in keys_to_check)

    return group_results_by_key([res['results'] for res in results]), args

#%% load results

results, args = get_all_matching_results('data/temporal-scaling_385*.pickle')

#%% recreate the key Gallistel & Gibbons plots

thresh = 0.1 # to define time to learning
xnse = 0.1 # noise added for jitter
clrs = {'I/T fixed': 'blue', 'I fixed': 'orange'}

fixedIoT = 5; fixedI = 24
fixedIoT = 10; fixedI = 48

PtsT = {}
PtsIoT = []
PtsAll = []
for key, items in results.items():
    I, T = key
    IoT = I/T
    C = I+T # cycle length

    ys = np.vstack([item['RPE_resps'][:,1:].mean(axis=1) for item in items])

    if IoT == fixedIoT:
        grp = 'I/T fixed'
    elif I == fixedI:
        grp = 'I fixed'
    else:
        grp = 'other'
    if grp not in PtsT:
        PtsT[grp] = []

    for j in range(ys.shape[0]):
        ts = np.where(ys[j,:] > thresh)[0]
        if len(ts) == 0 or ts[0] == 0:
            continue
        y = ts[0]
        # print(I, T, IoT, y, grp)
        PtsT[grp].append((T, y))
        PtsIoT.append((IoT, y))
        PtsAll.append((I, T, IoT, y))

ncols = 3; nrows = 1
plt.figure(figsize=(3*ncols,3*nrows)); c = 1
plt.subplot(nrows,ncols,c); c += 1
for grp, pts in PtsT.items():
    if grp not in clrs:
        continue
    pts = np.vstack(pts)
    xsa = np.unique(pts[:,0])
    mus = []
    for x in xsa:
        ys = pts[pts[:,0] == x,1]
        mu = np.median(ys); lb = np.percentile(ys, 25); ub = np.percentile(ys, 75)
        # mu = np.mean(ys); se = np.std(ys)/np.sqrt(len(ys)); lb = mu-se; ub = mu+se
        mus.append(mu)

        xs = x * np.ones(len(ys)) + xnse*np.random.randn(len(ys))
        plt.scatter(xs, ys, s=3, c=clrs[grp], alpha=0.8)
        plt.plot(x, mu, 'o', color=clrs[grp], markeredgecolor='k')
        plt.plot(x * np.ones(2), [lb, ub], 'k-', zorder=-1)
    plt.plot(xsa, mus, '-', c=clrs[grp], zorder=-1, label=grp)

plt.xlabel('T = Delay of Reinforcement')
plt.ylabel('Reinforcements to Acquisition')
plt.xticks(ticks=xsa, labels=[int(x) for x in xsa])
# plt.yscale('log')
plt.legend()

plt.subplot(nrows,ncols,c); c += 1
pts = np.vstack(PtsIoT)
xss = pts[:,0].astype(int)
xsa = np.unique(xss)
mus = []
for x in xsa:
    ys = pts[xss == x,1]
    mu = np.median(ys); lb = np.percentile(ys, 25); ub = np.percentile(ys, 75)
    # mu = np.mean(ys); se = np.std(ys)/np.sqrt(len(ys)); lb = mu-se; ub = mu+se
    mus.append(mu)

    xs = x * np.ones(len(ys)) + xnse*np.random.randn(len(ys))
    plt.scatter(xs, ys, s=3, c='k', alpha=0.8)
    plt.plot(x, mu, 'o', color='k', markeredgecolor='k')
    plt.plot(x * np.ones(2), [lb, ub], 'k-', zorder=-1)
plt.plot(xsa, mus, '-', c='k', zorder=-1, label=grp)

plt.xlabel('I/T')
plt.ylabel('Reinforcements to Acquisition')
plt.xscale('log')
# plt.yscale('log')
plt.xticks(ticks=xsa, labels=xsa, rotation=90)
plt.minorticks_off()
yticks = [700,500,300]
plt.yticks(ticks=yticks, labels=yticks)

plt.subplot(nrows,ncols,c); c += 1
pts = np.vstack(PtsAll)
Ts = pts[:,1].astype(int)
Ts_all = np.unique(Ts)
xsas = []
for T in Ts_all:
    cpts = pts[Ts == T,:]
    xs = cpts[:,0]
    ys = cpts[:,-1]
    xsa = np.unique(xs)
    ysa = np.array([np.mean(ys[xs == x]) for x in xsa])

    ix = np.argsort(xsa)
    xsc = xsa[ix]
    ysc = ysa[ix]
    xsas.extend(xsc)
    
    plt.plot(xsc, ysc, '.-', label=f'{T=}')
plt.xlabel('I = Intertrial Interval')
plt.ylabel('Reinforcements to Acquisition')
plt.xscale('log')
# plt.yscale('log')
xsa = np.unique(xsas).astype(int)
plt.xticks(ticks=xsa, labels=xsa, rotation=90)
plt.minorticks_off()
plt.legend(fontsize=6)
plt.tight_layout()

#%%

showLoss = False

ncols = 5; nrows = 2
plt.figure(figsize=(3*ncols,3*nrows)); c = 1
for key, items in results.items():
    I, T = key
    IoT = I/T
    # if I != 48:
    #     continue
    if IoT != 10:
        continue

    if showLoss:
        ys = np.vstack([item['scores'] for item in items])
    else:
        ys = np.vstack([item['RPE_resps'].mean(axis=1) for item in items])
        # ys = np.vstack([item['RPE_resps'][:,1:].mean(axis=1) for item in items])
        vs = [np.where(y > thresh)[0][0] for y in ys]
        plt.subplot(nrows,ncols,10); plt.plot([T]*len(vs), vs, '.')

    if not showLoss and c == nrows*ncols:
        continue
    plt.subplot(nrows,ncols,c); c+=1
    plt.plot(ys.T)
    # [plt.plot(v, thresh, '.') for v in vs]
    plt.title(key, fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if not showLoss:
        plt.plot(plt.xlim(), thresh*np.ones(2), 'k-', zorder=-1, alpha=0.5)
    if c == 2:
        plt.xlabel('# reinforcements', fontsize=8)
        plt.ylabel('loss' if showLoss else 'RPE', fontsize=8)

plt.tight_layout()
