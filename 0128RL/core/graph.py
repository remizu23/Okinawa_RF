import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
import copy
from tqdm import tqdm
import multiprocessing as mp
import sys

class Graph(object):

    def __init__(self):
        self.T = None
        self.ods = None
        self.state_networks = {}
        pass

    def read_data(self, node_data, link_data, features=[], od_data=None):
        """
        Arguments:
            node_data: pandas dataframe ('node_id', 'x', 'y')
            link_data: pandas dataframe ('link_id', 'from_', 'to_', 'length', ...)
            *** link id (sender, receiver) should be consistent with index ***
        """

        # node data
        self.nodes = node_data['node_id'].values # N,
        self.pos = np.vstack([node_data['x'].values, node_data['y'].values]).T # N,2

        # link data
        self.links = link_data.index.values # L,
        self.L = len(self.links)
        # self.links = np.append(links, [self.L, self.L+1]) # L + 2: for o and d
        link_nodes = np.vstack([link_data['from_'].values, link_data['to_'].values]).T # L,2

        # edge (link pair) data
        self.edges = []
        for k, (ki, kj) in enumerate(link_nodes):
            As = np.where(kj == link_nodes[:,0])[0]
            for a in As: self.edges.append((k,a))
        self.edges = np.array(self.edges)
        self.senders = self.edges[:,0]
        self.receivers = self.edges[:,1]
        # get forward/backward stars
        self.get_stars()

        # link features
        self.link_features = {}
        for f in features:
            self.link_features[f] = link_data[f].values

        # edge (pair) features
        self.edge_features = {}

        # od data
        if od_data is not None:
            self.origins = od_data['origin'].unique()
            self.dests = od_data['destination'].unique()
            self.ods = od_data[['origin', 'destination']].values

        # dummy edges to be added
        self.dummy_forward_stars = {} # for origin nodes
        self.dummy_backward_stars = {} # for dest nodes
        for o in self.origins:
            olinks = np.where(link_nodes[:,0] == o)[0]
            self.dummy_forward_stars[o] = olinks
        for d in self.dests:
            dlinks = np.where(link_nodes[:,1] == d)[0]
            self.dummy_backward_stars[d] = dlinks

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def define_T_from_obs(self, detour_df, range=1.34, default_T=10):
        T = {} # dict for each destination
        for d in self.dests:
            df = detour_df.query(f'destination == {d}')
            if len(df) == 0:
                Td = default_T
            elif len(df) > 1:
                Td = int(df['obs_step'].max())
            else:
                Td = 0
                for min_step, obs_step in df[['min_step', 'obs_step']].values:
                    cand = max(min_step * range, obs_step)
                    if Td < cand: Td = int(cand)
            T[d] = Td
        self.T = T

    def define_T_from_obs_new(self, detour_df, range=1.34, default_T=10):
        T = {} # dict for each destination
        for d in self.dests:
            df = detour_df.query(f'destination == {d}')
            if len(df) == 0:
                Td = default_T
            else:
                Td = 0
                for min_step, obs_step in df[['min_step', 'obs_step']].values:
                    cand = max(min_step * range, obs_step)
                    if Td < cand: Td = int(cand)
            T[d] = Td
        self.T = T

    def define_T_from_obs_def_min(self, detour_df, range=1.34, default_T=10, min_T=0):
        T = {} # dict for each destination
        for d in self.dests:
            df = detour_df.query(f'destination == {d}')
            if len(df) == 0:
                Td = default_T
            else:
                Td = 0
                for min_step, obs_step in df[['min_step', 'obs_step']].values:
                    cand = max(min_step * range, obs_step, min_T)
                    if Td < cand: Td = int(cand)
            T[d] = Td
        self.T = T

    def set_dummy_edges(self, o=None, d=None):
        # additions
        links = self.links.copy()
        forward_stars = copy.deepcopy(self.forward_stars)
        backward_stars = copy.deepcopy(self.backward_stars)

        dummy_edges = []
        olink, dlink = None, None
        if o is not None:
            olink = len(links)
            links = np.append(links, [olink])
            backward_stars[olink] = []
            forward_stars[olink] = self.dummy_forward_stars[o]
            for a in self.dummy_forward_stars[o]:
                backward_stars[a].append(olink)
                dummy_edges.append((olink, a))
        if d is not None:
            dlink = len(links)
            links = np.append(links, [dlink])
            forward_stars[dlink] = []
            backward_stars[dlink] = self.dummy_backward_stars[d]
            for k in self.dummy_backward_stars[d]:
                forward_stars[k].append(dlink)
                dummy_edges.append((k, dlink))
        edges = np.concatenate(
            [self.edges, dummy_edges], axis=0
        )
        return links, edges, forward_stars, backward_stars, olink, dlink, dummy_edges

    def get_stars(self):
        self.forward_stars = {i:[] for i in self.links}
        self.backward_stars = {j:[] for j in self.links}
        for sender, receiver in zip(self.senders, self.receivers):
            self.forward_stars[sender].append(receiver)
            self.backward_stars[receiver].append(sender)

    def _compute_minimum_steps(self, o=None, d=None, return_path=False):
        # set dummy edges
        newnet = self.set_dummy_edges(o=o, d=d)
        links, edges, forward_stars, backward_stars, olink, dlink, dummy_edges = newnet

        ## compute minimum steps using network x
        # define graph
        G = nx.DiGraph()
        G.add_nodes_from(links)
        G.add_edges_from(edges)

        # computation
        Do, Dd = None, None

        if o is not None:
            # minimum steps from origin link
            min_step_from_o = nx.shortest_path_length(G, source=olink)
            min_step_from_o = dict(sorted(min_step_from_o.items(), key=lambda x:x[0]))
            # convert to arrays Do, Dd
            Do = np.array(list(min_step_from_o.values()))

        if d is not None:
            # minimum steps to dest link
            G_rev = G.reverse(copy=False)
            min_step_to_d   = nx.shortest_path_length(G_rev, source=dlink)
            min_step_to_d   = dict(sorted(min_step_to_d.items(), key=lambda x:x[0]))
            # convert to arrays Do, Dd
            Dd = np.array(list(min_step_to_d.values()))
            if return_path:
                # obtain shortest paths
                path_d = nx.shortest_path(G_rev, source=dlink)
                for k, v in path_d.items(): v.reverse()
                return Do, Dd, newnet, path_d

        return Do, Dd, newnet

    def _compute_shortest_path_lengths(self, method='d'):
        # computation
        step_sp, len_sp = {}, {}

        for d in self.dests:
            # set dummy edges
            newnet = self.set_dummy_edges(o=None, d=d)
            links, edges, forward_stars, backward_stars, olink, dlink, dummy_edges = newnet
            weights = np.append(self.link_features['length'], 0.)
            weighted_edges = [(u, v, w) for (u,v), w in zip(edges, weights[edges[:,1]])]

            ## compute minimum steps using network x
            # define graph
            G = nx.DiGraph()
            G.add_nodes_from(links)
            G.add_weighted_edges_from(weighted_edges)

            # minimum steps to dest link
            G_rev = G.reverse(copy=False)
            min_step_to_d   = nx.shortest_path_length(G_rev, source=dlink)
            min_step_to_d   = dict(sorted(min_step_to_d.items(), key=lambda x:x[0]))
            min_len_to_d   = nx.single_source_dijkstra_path_length(G_rev, source=dlink)
            min_len_to_d   = dict(sorted(min_len_to_d.items(), key=lambda x:x[0]))
            # convert to arrays Do, Dd
            Sd = np.array(list(min_step_to_d.values()))
            Ld = np.array(list(min_len_to_d.values()))
            # output
            step_sp[d] = Sd
            len_sp[d] = Ld

        # store in self
        self.step_sp = step_sp
        self.len_sp = len_sp

    def get_state_networks(self, method='d', parallel=False):
        print('current size:', sys.getsizeof(self.state_networks))

        # define keys
        keys_ = {
            'd': self.dests,
            'od': self.ods,
        }.get(method)

        # define T
        T = self.T
        assert T is not None, 'define T first by g.update(T=?), then you can get state networks'
        # convert to arrays if T is integer
        if type(T) == int: T = {key_: T for key_ in keys_}

        # generate state networks only for the new keys or ones whose T have changed
        state_networks = {}
        do_keys_ = []
        for key_ in T.keys():
            if key_ not in self.state_networks.keys(): # if not yet generated
                do_keys_.append(key_)
            elif T[key_] != self.state_networks[key_]['T']: # if T has changed
                do_keys_.append(key_)
            else:
                state_networks[key_] = self.state_networks[key_]
        print('do_keys', len(keys_), len(do_keys_))

        self.state_networks = None
        # get state networks
        if not parallel:
            for key_ in tqdm(do_keys_):
                # get integer T
                Tod = T[key_]
                state_networks[key_] = self._get_state_network_partition([key_, Tod, method])
        else:
            n_threads = len(do_keys_)
            pool = mp.Pool(n_threads)
            argsList = [
                [do_keys_[r], T[do_keys_[r]], method] for r in range(n_threads)
                ]
            snetList = pool.map(self._get_state_network_partition, argsList)
            pool.close()
            for snet, args in zip(snetList, argsList):
                state_networks[args[0]] = snet
            # n_cpu = mp.cpu_count()
            # n_threads = n_cpu - 1
            # pool = mp.Pool(n_threads)
            # particles = [(len(do_keys_) + i) // n_threads for i in range(n_threads)]
            # cum = np.cumsum(particles)
            # argsList = [
            #     [do_keys_[cum[r-1]:cum[r]], method] if r > 0 else [do_keys_[:cum[r]], method]
            #     for r in range(n_threads)
            #     ]
            # snetList = pool.map(self._get_state_network_partition, argsList)
            # pool.close()
            # for probs, args in zip(snetList, argsList):
            #     for k, key_ in enumerate(args[0]):
            #         self.state_networks[key_] = snet
        self.state_networks = state_networks

    def _get_state_network_partition(self, params):
        key_, Tod, method = params
        # state space (list of state list at t)
        if method == 'od':
            Do, Dd, newnet = self._compute_minimum_steps(o=key_[0], d=key_[1])
            state_space = {
                t: list(np.where((Do <= t) * (Dd <= (Tod-t)))[0])
                for t in range(0, Tod+1)
            }
        elif method == 'd':
            # if d appears for the first time
            _, Dd, newnet = self._compute_minimum_steps(d=key_)
            state_space = {
                t: list(np.where((Dd <= (Tod-t)))[0])
                for t in range(0, Tod+1)
            }
        links, edges, forward_stars, backward_stars, olink, dlink, dummy_edges = newnet

        # state list
        states = []
        for t, st in state_space.items():
            for k in st:
                states.append((t,k))
        # print(f'key_={key_}, number of states={len(states)}')
        # state indexes
        states_idx = {s:i for i,s in enumerate(states)}
        # state senders and receivers
        transitions, static_edges, static_senders, static_receivers = [], [], [], []
        edge_idx = {tuple(edges[e]): e for e in range(edges.shape[0])}
        for i, s in enumerate(states):
            t, k = s
            if t == Tod or k == links[-1]:
                continue # no forward stars
            for a in forward_stars[k]:
                if (t+1, a) in states:
                    j = states_idx[(t+1,a)]
                    transitions.append((i,j))
                    # static pair index
                    idx = edge_idx[(k,a)]
                    static_edges.append(idx)
                    static_senders.append(k)
                    static_receivers.append(a)
        transitions = np.array(transitions)
        s_senders, s_receivers = transitions[:,0], transitions[:,1]

        # edge idx matrix
        S, E = len(states), len(transitions)
        trans_idx = csr_matrix(
            (np.arange(E), (s_senders, s_receivers)),
            shape=(S, S)
        )

        # new attributes for od_data
        init_idx = states_idx[(0,links[-2])] if method == 'od' else None
        fin_idx = states_idx[(Tod,links[-1])]
        snet = {
            'T': Tod,
            'state_space': state_space,
            'states': states,
            'states_idx': states_idx,
            'init_idx': init_idx,
            'fin_idx': fin_idx,
            's_senders': s_senders, # this should be defined by indexes
            's_receivers': s_receivers, # this should be defined by indexes
            'transitions': transitions, # state pairs
            'static_edges': static_edges, # for indexing static utilities v
            'static_senders': static_senders, # for indexing static mu
            'static_receivers': static_receivers, # for indexing static mu
            'trans_idx': trans_idx, # transition idxs (S x S csr_matrix)
            'backward_stars': backward_stars, # new BS including o and d (for backward computation)
            'dummy_static_edges': dummy_edges, # for v, v_dict modifications in the model
        }
        return snet

    def clone(self):
        return copy.deepcopy(self)
