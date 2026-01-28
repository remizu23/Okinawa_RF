import numpy as np
import pandas as pd
from optimparallel import minimize_parallel
from core.model import PrismRL, RL
from core.graph import Graph
from core.utils import Timer
from core.dataset import *
import time
import json
import argparse
np.random.seed(111)
import os  # osがない場合は追加

network_ = 'okinawa_output'  # フォルダ名になります
os.makedirs(f'data/{network_}/results', exist_ok=True)

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

def float_or_none(value):
    try:
        return float(value)
    except:
        return None

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--rl', type=str2bool, default=True, help='if estimate RL or not')
model_arg.add_argument('--prism', type=str2bool, default=True, help='if estimate prism RL or not')
model_arg.add_argument('--parallel', type=str2bool, default=False, help='if implement parallel computation or not')
model_arg.add_argument('--version', type=str, default='test', help='version name')

# Hyperparameters
model_arg.add_argument('--state_key', type=str, default='d', help='od or d')
model_arg.add_argument('--T', type=int, default=15, help='time constraint')
model_arg.add_argument('--T_range', type=float, default=1.34, help='range for T')
model_arg.add_argument('--uturn', type=str2bool, default=True, help='if add uturn dummy or not')
model_arg.add_argument('--uturn_penalty', type=float, default=-10., help='penalty for uturn')
model_arg.add_argument('--min_n', type=int, default=0, help='minimum number observed for d')

# parameters
default_vars = ['length', 'park_ave', 'first_st', 'gate_st', 'national_rd', 'parking', 'plaza_stay']
n_vars = len(default_vars)

# parameters
model_arg.add_argument('--vars', nargs='+', type=str, default=default_vars, help='explanatory variables')
# ★ここが重要: 変数の数だけ -1 を用意する
model_arg.add_argument('--init_beta', nargs='+', type=float, default=[-0.5] * n_vars, help='initial parameter values')
# lb, ub も数合わせ
model_arg.add_argument('--lb', nargs='+', type=float_or_none, default=[None] * n_vars, help='lower bounds')
model_arg.add_argument('--ub', nargs='+', type=float_or_none, default=[None] * n_vars, help='upper bounds')

# Validation
model_arg.add_argument('--n_samples', type=int, default=1, help='number of samples')
model_arg.add_argument('--test_ratio', type=float, default=0., help='ratio of test samples')
model_arg.add_argument('--two_step', type=str2bool, default=False, help='if using prism results')
model_arg.add_argument('--isBootstrap', type=str2bool, default=False, help='if bootstrapping or not')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed

Niter = 1
def callbackF(x):
    global Niter
    txt = f'{Niter: d}'
    for i in range(len(x)): txt += f'\t{x[i]:.4f}'
    print(txt)
    Niter += 1


if __name__ == '__main__':
    config, _ = get_config()
    config.version += '_' + time.strftime("%Y%m%dT%H%M")
    timer = Timer()
    _ = timer.stop()

    # read data
    # network_ = 'SiouxFalls'
    # dir_ = f'data/{network_}/'
    # link_data = pd.read_csv(dir_+'link.csv')
    # node_data = pd.read_csv(dir_+'node.csv')
    # obs_data = pd.read_csv(dir_+'synthetic_data.csv')

    link_data = pd.read_csv('link.csv')
    node_data = pd.read_csv('node.csv')
    obs_data = pd.read_csv('synthetic_data.csv')


    # define variables
    link_data['capacity'] /= link_data['capacity'].max()
    link_data['caplen'] = link_data['capacity'] * link_data['length']
    features = link_data

    # reset index
    obs_data = reset_index(link_data, node_data, obs_data)
    links = {link_id: (from_, to_) for link_id, from_, to_ in link_data[['link_id', 'from_', 'to_']].values}

    # read map-matching results
    dests, obs, obs_filled, n_paths, max_len, od_data, samples = read_mm_results(
        obs_data, links, min_n_paths=config.min_n, n_samples=config.n_samples, test_ratio=config.test_ratio, seed_=111, isBootstrap=config.isBootstrap)

    # number of paths
    print(f"number of paths observed: {n_paths}")
    # loop counts
    n_loops = count_loops(obs_filled)
    for d, n_loop in n_loops.items():
        if n_loop > 0:
            print(f"number of paths including loops observed: {n_loop} for destination {d}")

    ## Define Graph
    g = Graph()
    g.read_data(node_data=node_data, link_data=link_data, od_data=od_data)
    # g.update(T=T)

    ## Define choice-stage structured network
    if config.prism:
        detour_df = analyze_detour_rate(g, obs)
        g.define_T_from_obs(detour_df, range=config.T_range, default_T=0)
        print(f"T = {g.T}")
        timer.start()
        g.get_state_networks(method=config.state_key, parallel=True)
        snet_time = timer.stop()
        print(f"time to get snets is {snet_time}s.")

    ## Define variables
    xs = {}
    for var_name in config.vars:
        # features (link_data) から該当列を取得
        if var_name in features.columns:
            xs[var_name] = (features[var_name].values, 'link')
        else:
            raise KeyError(f"Variable '{var_name}' is not in link.csv columns!")

    # add uturn dummy
    U = (g.senders[:,np.newaxis] == g.receivers[np.newaxis,:]) * (g.receivers[:,np.newaxis] == g.senders[np.newaxis,:])
    U = np.where(U == True)[0]
    uturns = np.zeros_like(g.senders)
    uturns[U] = 1.
    xs['uturn'] = (uturns, 'edge')

    ## Define parameters
    betas = []
    for var_name, init_val, lb, ub in zip(config.vars, config.init_beta, config.lb, config.ub):
        betas.append(
            (f'b_{var_name}', init_val, lb, ub, var_name, 0)
        )
    if config.uturn:
        betas.append(('b_uturn', config.uturn_penalty, None, None, 'uturn', 1))

    ## Define models
    rl = RL(g, xs, betas)
    prism = PrismRL(g, xs, betas, method=config.state_key, parallel=config.parallel, print_process=False)

    ### FOR TWO-STEP ESTIMATION
    if config.two_step:
        prism_res_data = pd.read_csv(f'data/{network_}/prism_res.csv', index_col=0).T
        twostep_init_beta = prism_res_data[['beta_length', 'beta_caplen']].values[0]
        print(f"Two step estimation: starting with {twostep_init_beta}")

    ### Model Estimation
    # output
    outputs = {}
    if config.rl: outputs['RL'] = {i:{} for i in range(config.n_samples)}
    if config.prism: outputs['PrismRL'] = {i:{} for i in range(config.n_samples)}
    # function for record results
    def record_res(i, model_type, res, stderr, t_val, L0, L_val, runtime):
        outputs[model_type][i] = {
            'L0': L0,
            'LL': -res.fun,
            'Lv': L_val,
            'runtime': runtime,
        }
        for var_name, b, s, t in zip(config.vars, res.x, stderr, t_val):
            outputs[model_type][i].update({
                f'beta_{var_name}': b, f'se_{var_name}': s, f't_{var_name}': t,
            })

    for i, sample in enumerate(samples):
        train_obs = sample['train']
        test_obs = sample['test']

        if config.rl:
            # only observed destinations in samples
            rl.partitions = list(train_obs.keys())
            rl.beta = np.array(config.init_beta)
            if config.two_step: rl.beta = twostep_init_beta
            
            # ★ 変数初期化（計算失敗時に備えてNoneにしておく）
            LL0_rl = None 
            results_rl = None
            rl_time = 0
            stderr = []
            t_val = []
            LL_val_rl = 0

            try:
                # ★ tryの中に移動！
                LL0_rl = rl.calc_likelihood(observations=train_obs)
                print('RL model initial log likelihood:', LL0_rl)

                print(f"RL model estimation for sample {i}...")
                def f(x):
                    # compute probability
                    rl.eval_prob(x)
                    # calculate log-likelihood
                    LL = 0.
                    for key_, paths in train_obs.items():
                        p = rl.p[key_]
                        max_len, N = paths.shape
                        Lk = np.zeros(N, dtype=float)
                        for j in range(max_len - 1):
                            L = np.array(p[paths[j], paths[j+1]])[0]
                            assert (L > 0 ).all(), f'L includes zeros: key_={key_}, j={j}, pathj={paths[j]}, pathj+1={paths[j+1]}'
                            Lk += np.log(L)
                        LL += np.sum(Lk)
                    return -LL

                timer.start()
                results_rl = minimize_parallel(f, x0=rl.beta, bounds=rl.bounds, options={'disp':False, 'maxiter':100}, callback=callbackF)
                rl_time = timer.stop()
                print(f"estimation time is {rl_time}s.")
                
                rl.beta = results_rl.x
                # Note: std.err. should be calculated by bootstrapping
                cov_matrix = results_rl.hess_inv if type(results_rl.hess_inv) == np.ndarray else results_rl.hess_inv.todense()
                stderr = np.sqrt(np.diag(cov_matrix))
                t_val = results_rl.x / stderr
                rl.print_results(results_rl, stderr, t_val, LL0_rl)

                if config.test_ratio > 0:
                    # validation
                    rl.partitions = list(test_obs.keys())
                    LL_val_rl = rl.calc_likelihood(observations=test_obs)
                    print('RL model validation log likelihood:', LL_val_rl)
                else:
                    LL_val_rl = 0.

                # record results (成功時のみ記録)
                record_res(i, 'RL', results_rl, stderr, t_val, LL0_rl, LL_val_rl, rl_time)

            except Exception as e:
                # エラー内容を表示して、Prism-RLへ進む
                print(f"RL is not feasible for sample {i}")
                print(f"Reason: {e}")

        if config.prism:
            s_train_obs = prism.translate_observations(train_obs)
            s_test_obs = prism.translate_observations(test_obs)

            prism.beta = np.array(config.init_beta)
            prism.partitions = list(s_train_obs.keys())
            LL0 = prism.calc_likelihood(observations=s_train_obs)
            print('prism model initial log likelihood:', LL0)

            def f(x):
                # compute probability
                prism.eval_prob(x)
                # calculate log-likelihood
                LL = 0.
                for key_, paths in s_train_obs.items():
                    p = prism.p[key_]
                    max_len, N = paths.shape
                    Lk = np.zeros(N, dtype=float)
                    for j in range(max_len - 1):
                        L = np.array(p[paths[j], paths[j+1]])[0]
                        assert (L > 0 ).all(), f'L includes zeros: key_={key_}, j={j}, pathj={paths[j]}, pathj+1={paths[j+1]}'
                        Lk += np.log(L)
                    LL += np.sum(Lk)
                return -LL
            # estimation
            print(f"Prism RL model estimation for sample {i}...")
            Niter = 1
            timer.start()
            prism_res = minimize_parallel(f, x0=prism.beta, bounds=prism.bounds, options={'disp':False}, callback=callbackF)
            prism_time = timer.stop()
            print(f"estimation time is {prism_time}s.")
            # after estimation
            prism.beta = prism_res.x
            # Note: std.err. should be calculated by bootstrapping
            cov_matrix = prism_res.hess_inv if type(prism_res.hess_inv) == np.ndarray else prism_res.hess_inv.todense()
            stderr = np.sqrt(np.diag(cov_matrix))
            t_val = prism_res.x / stderr
            prism.print_results(prism_res, stderr, t_val, LL0)
            print(prism_res)

            # validation
            if config.test_ratio > 0:
                prism.partitions = list(s_test_obs.keys())
                LL_val = prism.calc_likelihood(observations=s_test_obs)
                print('Prism RL model validation log likelihood:', LL_val)
            else:
                LL_val = 0.

            # record results
            record_res(i, 'PrismRL', prism_res, stderr, t_val, LL0, LL_val, prism_time)

    if config.rl:
        df_rl = pd.DataFrame(outputs['RL']).T
        print(df_rl)
        if config.test_ratio > 0:
            # for validation
            df_rl.to_csv(f'data/{network_}/results/RL_valid_{config.version}.csv', index=True)
        else:
            # for estimation
            df_rl.T.to_csv(f'data/{network_}/results/RL_est_{config.version}.csv', index=True)
    if config.prism:
        df_prism = pd.DataFrame(outputs['PrismRL']).T
        print(df_prism)
        if config.test_ratio > 0:
            # for validation
            df_prism.to_csv(f'data/{network_}/results/PrismRL_valid_{config.version}.csv', index=True)
        else:
            # for estimation
            df_prism.T.to_csv(f'data/{network_}/results/PrismRL_est_{config.version}.csv', index=True)

    # write config file
    dir_ = f'data/{network_}/results/'
    file_name = 'valid' if config.test_ratio > 0 else 'est'
    file_name += '_' + config.version
    with open(f"{dir_}{file_name}.json", mode="w") as f:
        json.dump(config.__dict__, f, indent=4)
