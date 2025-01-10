#%%

trial_modes = ['conditioning']#, 'degradation', 'cue-c']
model_types = ['conditioning_weights']#, 'cue-c_weights', 'degradation_weights']
pre_iti = 5
max_isi = 7
post_reward = 7
cue = 0

# lick_rates = {}
for model_type, infiles in weightfiles.items():
    if model_type not in model_types:
        continue
    if model_type == 'initial_weights':
        model_type += '_conditioning'
    
    # create experiment
    trial_mode = next(x for x in trial_modes if x in model_type)
    np.random.seed(666)
    E_test = Contingency(mode=trial_mode, ntrials=1000, iti_min=20, ntrials_per_episode=20, jitter=1, t_padding=0, rew_times=[8]*3)
    dataloader_test = make_dataloader(E_test, batch_size=100)

    # if trial_mode not in lick_rates:
    #     lick_rates[trial_mode] = []
    # c_lick_rates = []
    Pts = []
    for item in infiles:
        # load model and eval on experiment
        weights = torch.load(item['filepath'], map_location=torch.device('cpu'))
        input_size = weights['rnn.weight_ih_l0'].shape[-1]
        hidden_size = weights['value.weight'].shape[-1]
        weights['bias'] = weights['bias'][None]
        model = ValueRNN(input_size=input_size, hidden_size=hidden_size, gamma=0.83)
        model.restore_weights(weights)
        trials = probe_model(model, dataloader_test, inactivation_indices=None)
        trials = [trial for trial in trials if trial.index_in_episode > 0]
        Pts.append(get_pts(trials))
    break

#%%

def get_pts(trials):
    pts = []
    for trial in trials:
        if trial.cue != 0:
            continue
        value_at_cs = trial.value[trial.iti,0]
        rpe_at_cs = trial.rpe[trial.iti-1,0]
        pts.append((value_at_cs, rpe_at_cs))
    pts = np.vstack(pts)
    return pts
    # plt.plot(pts[:,0], pts[:,1], '.')

#%%

for i, pts in enumerate(Pts):
    plt.plot(pts[:,0], pts[:,1], '.', alpha=1 if i == 4 else 0.1)
