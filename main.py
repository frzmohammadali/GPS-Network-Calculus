import argparse
import copy
import json
import logging
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

logging.basicConfig(format="[%(levelname)s] %(message)s (%(filename)s, %(funcName)s(), line %(lineno)d, %(asctime)s)", datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,  # level=logging.DEBUG,
                    )

import os, sys

sys.path.append(os.path.abspath('../'))
from src.arrival_curve import TokenBucket
from src.service_curve import WorkConservingLink, RateLatency
from src.gps import GPS
from src.nc import NC
from src.utilities import distinct_by, powerset_non_empty_generator, length_distinct_subsets, ReturnType, WeightsMode, clear_output, filter_generator, print_write, findIntersection


def setupInputs(number_of_flows, weight_mode: WeightsMode, target_util=0.75, seed=None, perFlowStability="all",
                flow_bursts = [], flow_rates = [], flow_weights = [], server_rate = None, server_delay = None):
    t = 1
    if seed:
        random.seed(seed)
    # b=random(1,5) Mb, r=random(3,30) Mb/s, R=2400 Mb/s (2.4 Gb/s), T=2.0 seconds
    #  [max number of flows would be 79 for stability condition to be satisfied]
    # constant b:
    # alphas = [TokenBucket(b=0.5, r=random.uniform(0.5, 5.0), t=t) for _ in
    #           range(number_of_flows)]
    # random b:
    # alphas = [TokenBucket(b=random.uniform(0.1, 1.5), r=1, t=t) for _ in range(number_of_flows)]
    # fix b:
    if not len(flow_rates):
        flow_rates = [1.0 for _ in range(number_of_flows)]
    if not len(flow_bursts):
        flow_bursts = [1.40, 1.5, 1.41, 0.28, 0.19, 1.07, 1.11, 1.01, 0.34, 1.04]
    alphas = [TokenBucket(r=flow_rates[_i], b=flow_bursts[_i], t=t) for _i in range(number_of_flows)]

    if weight_mode == WeightsMode.EQUAL:
        flow_weights = [1 for _ in alphas]
    elif weight_mode == WeightsMode.RPPS:
        flow_weights = [__a.r for __a in alphas]
    elif weight_mode == WeightsMode.RANDOM:
        flow_weights = [random.uniform(0.0, 1.0) for _ in alphas]
    elif weight_mode == WeightsMode.FIX:
        if not len(flow_weights):
            flow_weights = [0.95, 0.20, 0.25, 0.96, 0.61, 0.91, 0.33, 0.98, 0.72, 0.78]
    else:
        raise Exception("WeightMode not recognized : " + weight_mode)

    foi_index = 1
    if type(server_rate) != float:
        agg_arr_rate = (sum([a.r for a in alphas]))
        server_rate = agg_arr_rate / target_util
    if not server_delay:
        server_delay = 0.0
    # beta = WorkConservingLink(c=2400,t=t)
    # latency is usually boring, but add the sanity check, if safe set it to zero
    beta = RateLatency(R=server_rate, T=server_delay, t=t)
    if type(target_util) == float:
        if perFlowStability == "all":
            # fix per flow stability
            fixPerFLowStability_all(beta, alphas, flow_weights, target_util)
            # total stability
            assert beta.rate - (sum([a.r for a in alphas])) > 0, 'sys not stable'
        else:
            # fix stability only for foi
            fixPerFLowStability_foi(beta, foi_index, alphas, flow_weights, target_util)


    return t, alphas, flow_weights, foi_index, beta


def fixPerFLowStability_all(beta, alphas, alphas_weight, target_util):
    for ix, alpha in enumerate(alphas):
        loSc_PG = GPS.LoSC_PG(beta, ix, alphas_weight)
        loSc_PG_pct = loSc_PG.rate * target_util
        alpha.r = loSc_PG_pct
    return alphas
    
def fixPerFLowStability_foi(beta, foi_index, alphas, alphas_weight, target_util):
    fixPerFLowStability_all(beta, alphas, alphas_weight, target_util)
    crossflow_target_util = 1.10

    ## some over-utilized by random
    # counter = 1
    # while counter <= 5:
    #     _ix = random.randint(0,len(alphas)-1)
    #     if _ix == foi_index:
    #         continue
    #     alphas[_ix].r = GPS.LoSC_PG(beta, _ix, alphas_weight).rate * crossflow_target_util
    #     counter += 1

    # all over-utilized
    for _ix, alpha in enumerate(alphas):
        if _ix != foi_index:
            alphas[_ix].r = GPS.LoSC_PG(beta, _ix, alphas_weight).rate * crossflow_target_util


    return alphas




# homogeneous arrivals analysis optimized by delay bound (min)
def hetro_arr_analysis_OBDB(number_of_flows, weight_mode: WeightsMode):
    result = dict()
    t, alphas, alphas_weights, foi_index, beta = setupInputs(number_of_flows, weight_mode)

    result["PG (General)"] = {
        'LoSC': GPS.LoSC_PG(sc=beta, index=foi_index, weights=alphas_weights)
    }
    result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])

    print_write("Chang")
    print_write("-----\n\n")
    subsets = powerset_non_empty_generator(list(range(len(alphas))))
    result["Chang"] = {
        'LoSC': GPS.LoSC_Chang_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subsets)
    }
    result["Chang"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Chang']['LoSC'][0])
    print_write("--done\n\n")

    print_write("Bouillard")
    print_write("---------\n\n")
    result['Bouillard'] = {
        'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
    }
    result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'][0])
    print_write("--done\n\n")

    print_write("BL")
    print_write("--\n\n")
    # we change N\{i} to N such that always i in M to make its semantic consistent with chang
    # semantic
    _subset_BL = powerset_non_empty_generator(list(range(len(alphas))))
    subset_BL = filter_generator(lambda x: foi_index in x, _subset_BL)
    result["Burchard, Liebeherr"] = {
        'LoSC': GPS.LoSC_BL_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subset_BL)
    }
    result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'][0])
    print_write("--done\n\n")
    time.sleep(0.5)
    print_write()
    print_write("number of arrivals:", len(alphas))
    print_write('flow of interest:', alphas[foi_index])
    print_write('weights mode:', weight_mode)
    print_write('weight of the flow of interest:', alphas_weights[foi_index])
    print_write("distinct weights: ", list(set(alphas_weights)))
    for key, value in result.items():
        print_write(f'{key: <30}', ": ", value)
    print_write("\n\n\n")


class RuntimeMeasurementAnalysisDriver:
    def __init__(self):
        self.config = self.setupParameter()

    def setupParameter(self):
        confpath = os.path.abspath(os.path.dirname(__file__) + '/' + os.path.basename(__file__) + '.json')
        if not os.path.exists(confpath):
            raise Exception("config file not found!")
        with open(confpath, 'r') as f:
            config = json.load(f)
        if "finally" not in config:
            config["finally"] = "None"
        return config

    def plot_utilizationByDelayBound(self):
        graph_PG = {
            'x': [],
            'y': []
        }
        graph_chang = {
            'x': [],
            'y': []
        }
        graph_bouillard = {
            'x': [],
            'y': []
        }
        graph_bouillard_new = {
            'x': [],
            'y': []
        }
        graph_bouillard_burstyFirst = {
            'x': [],
            'y': []
        }
        graph_bouillard_burst = {
            'x': [],
            'y': []
        }
        graph_bouillard_rate = {
            'x': [],
            'y': []
        }
        graph_BL = {
            'x': [],
            'y': []
        }
        for tu in tqdm([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]):
            result = dict()
            t, alphas, alphas_weights, foi_index, beta = setupInputs(number_of_flows=self.config['number_of_flows'], weight_mode=eval(self.config['weights_mode']), target_util=tu,
                                                                     seed=eval(self.config['seed']), perFlowStability=self.config['flow_stability'],
                                                                     flow_rates=self.config['flow_rates'], flow_bursts=self.config['flow_bursts'],flow_weights=self.config['flow_weights'],
                                                                     server_rate=self.config['server_rate'], server_delay=self.config['server_delay'])

            result["PG (General)"] = {
                'LoSC': GPS.LoSC_PG(sc=beta, index=foi_index, weights=alphas_weights)
            }
            result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])
            graph_PG['x'].append(tu)
            graph_PG['y'].append(result["PG (General)"]['delay bound'])

            subsets = powerset_non_empty_generator(list(range(len(alphas))))
            result["Chang"] = {
                'LoSC': GPS.LoSC_Chang_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subsets)
            }
            result["Chang"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Chang']['LoSC'][0])
            graph_chang['x'].append(tu)
            graph_chang['y'].append(result["Chang"]['delay bound'])

            result['Bouillard'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'][0])
            graph_bouillard['x'].append(tu)
            graph_bouillard['y'].append(result["Bouillard"]['delay bound'])

            result['Bouillard_new'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_new(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_new']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_new']['LoSC'][0])
            graph_bouillard_new['x'].append(tu)
            graph_bouillard_new['y'].append(result["Bouillard_new"]['delay bound'])

            result['Bouillard_burstyFirst'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_burstyFirst(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_burstyFirst']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_burstyFirst']['LoSC'][0])
            graph_bouillard_burstyFirst['x'].append(tu)
            graph_bouillard_burstyFirst['y'].append(result["Bouillard_burstyFirst"]['delay bound'])

            result['Bouillard_burst'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_burst(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_burst']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_burst']['LoSC'][0])
            graph_bouillard_burst['x'].append(tu)
            graph_bouillard_burst['y'].append(result["Bouillard_burst"]['delay bound'])

            result['Bouillard_rate'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_rate(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_rate']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_rate']['LoSC'][0])
            graph_bouillard_rate['x'].append(tu)
            graph_bouillard_rate['y'].append(result["Bouillard_rate"]['delay bound'])

            _subset_BL = powerset_non_empty_generator(list(range(len(alphas))))
            subset_BL = filter_generator(lambda x: foi_index in x, _subset_BL)
            result["Burchard, Liebeherr"] = {
                'LoSC': GPS.LoSC_BL_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subset_BL)
            }
            result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'][0])
            graph_BL['x'].append(tu)
            graph_BL['y'].append(result["Burchard, Liebeherr"]['delay bound'])

        # plot
        datalist = [
                    (graph_PG['x'], graph_PG['y'], {'label':'PG','linewidth':1, 'color':'red', 'marker':'s', 'linestyle':'-'}),
                    (graph_chang['x'], graph_chang['y'], {'label':'Chang','linewidth':2, 'color':'blue', 'marker':'^', 'linestyle':'--'}),
                    (graph_bouillard['x'], graph_bouillard['y'], {'label':'BBLC','linewidth':1.2, 'color':'magenta', 'marker':'o', 'linestyle':'-.'} ),
                    (graph_bouillard_new['x'], graph_bouillard_new['y'], {'label': 'Bouillard_new','linewidth':1.5, 'color':'black', 'marker':'d', 'linestyle':'-.'}),
                    (graph_bouillard_burstyFirst['x'], graph_bouillard_burstyFirst['y'], {'label': 'BBLC_burstyFirst','linewidth':0.6, 'color':'cyan', 'marker':'s', 'linestyle':'-.'}),
                    (graph_bouillard_burst['x'], graph_bouillard_burst['y'], {'label': 'BBLC_burst','linewidth':0.8, 'color':'tab:orange', 'marker':'d', 'linestyle':'-.'}),
                    (graph_bouillard_rate['x'], graph_bouillard_rate['y'], {'label': 'BBLC_rate','linewidth':0.5, 'color':'brown', 'marker':'d', 'linestyle':'-.'}),
                    (graph_BL['x'], graph_BL['y'], {'label':'BL','linewidth':2, 'color':'green', 'marker':'p', 'linestyle':':'})
                ]

        ticks = {
            'x': np.arange(0.5, 1.01, 0.05),
            'y': np.arange(0.0, 5.0, 1.0)
        }
        server_type = "RateLatency"
        if beta.T == 0.0:
            server_type = "ConstantRate"
        self.plot(datalist,
                  title=f"Impact of utilization (weights: {eval(self.config['weights_mode'])})",
                  xlabel='utilization', ylabel='delay bound',
                  text=(f"flow_rates: {[round(a.r,2) for a in alphas]}\n"
                        f"flow_bursts: {[round(a.b,2) for a in alphas]}\n"
                        f"flow_weights: {alphas_weights}\n"
                        f"foi_index: {foi_index}\n"
                        f"server: <{server_type} R=<varies by utilization> T={round(beta.T,2)} />\n"
                        f"flow_stability: {self.config['flow_stability']}\n"
                        f"No. of flows: {self.config['number_of_flows']}"),
                  # ticks=ticks,
                  end=self.config['finally'],
                  figsize=(9,7))

    def plot_weightsOfFoiByDelayBound(self):
        graph_PG = {
            'x': [],
            'y': []
        }
        graph_chang = {
            'x': [],
            'y': []
        }
        graph_bouillard = {
            'x': [],
            'y': []
        }
        graph_bouillard_new = {
            'x': [],
            'y': []
        }
        graph_bouillard_burstyFirst = {
            'x': [],
            'y': []
        }
        graph_bouillard_burst = {
            'x': [],
            'y': []
        }
        graph_bouillard_rate = {
            'x': [],
            'y': []
        }
        graph_BL = {
            'x': [],
            'y': []
        }

        for foi_weight in tqdm([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]):
            result = dict()
            t, alphas, alphas_weights, foi_index, beta = setupInputs(number_of_flows=self.config['number_of_flows'], weight_mode=eval(self.config['weights_mode']), target_util=self.config['target_utilization'],
                                                                     seed=eval(self.config['seed']), perFlowStability=self.config['flow_stability'],
                                                                     flow_rates = self.config['flow_rates'], flow_bursts = self.config['flow_bursts'], flow_weights = self.config['flow_weights'], \
                                                                     server_rate = self.config['server_rate'], server_delay =self.config['server_delay'])
            alphas_weights[foi_index] = foi_weight

            result["PG (General)"] = {
                'LoSC': GPS.LoSC_PG(sc=beta, index=foi_index, weights=alphas_weights)
            }
            result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])
            graph_PG['x'].append(alphas_weights[foi_index]/sum(alphas_weights))
            graph_PG['y'].append(result["PG (General)"]['delay bound'])

            subsets = powerset_non_empty_generator(list(range(len(alphas))))
            result["Chang"] = {
                'LoSC': GPS.LoSC_Chang_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subsets)
            }
            result["Chang"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Chang']['LoSC'][0])
            graph_chang['x'].append(alphas_weights[foi_index]/sum(alphas_weights))
            graph_chang['y'].append(result["Chang"]['delay bound'])

            result['Bouillard'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'][0])
            graph_bouillard['x'].append(alphas_weights[foi_index]/sum(alphas_weights))
            graph_bouillard['y'].append(result["Bouillard"]['delay bound'])

            result['Bouillard_new'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_new(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_new']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_new']['LoSC'][0])
            graph_bouillard_new['x'].append(alphas_weights[foi_index]/sum(alphas_weights))
            graph_bouillard_new['y'].append(result["Bouillard_new"]['delay bound'])

            result['Bouillard_burstyFirst'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_burstyFirst(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_burstyFirst']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_burstyFirst']['LoSC'][0])
            graph_bouillard_burstyFirst['x'].append(alphas_weights[foi_index]/sum(alphas_weights))
            graph_bouillard_burstyFirst['y'].append(result["Bouillard_burstyFirst"]['delay bound'])

            result['Bouillard_burst'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_burst(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_burst']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_burst']['LoSC'][0])
            graph_bouillard_burst['x'].append(alphas_weights[foi_index]/sum(alphas_weights))
            graph_bouillard_burst['y'].append(result["Bouillard_burst"]['delay bound'])

            result['Bouillard_rate'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_rate(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_rate']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_rate']['LoSC'][0])
            graph_bouillard_rate['x'].append(alphas_weights[foi_index]/sum(alphas_weights))
            graph_bouillard_rate['y'].append(result["Bouillard_rate"]['delay bound'])

            _subset_BL = powerset_non_empty_generator(list(range(len(alphas))))
            subset_BL = filter_generator(lambda x: foi_index in x, _subset_BL)
            result["Burchard, Liebeherr"] = {
                'LoSC': GPS.LoSC_BL_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subset_BL)
            }
            result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'][0])
            graph_BL['x'].append(alphas_weights[foi_index]/sum(alphas_weights))
            graph_BL['y'].append(result["Burchard, Liebeherr"]['delay bound'])

        # plot
        list_PG = list(zip(graph_PG['x'], graph_PG['y']))
        list_PG.sort()
        PG_x, PG_y = zip(*list_PG)
        list_chang = list(zip(graph_chang['x'], graph_chang['y']))
        list_chang.sort()
        chang_x, chang_y = zip(*list_chang)
        list_bouillard = list(zip(graph_bouillard['x'], graph_bouillard['y']))
        list_bouillard.sort()
        bouillard_x, bouillard_y = zip(*list_bouillard)
        list_bouillard_new = list(zip(graph_bouillard_new['x'], graph_bouillard_new['y']))
        list_bouillard_new.sort()
        bouillard_new_x, bouillard_new_y = zip(*list_bouillard_new)

        list_bouillard_burstyFirst = list(zip(graph_bouillard_burstyFirst['x'], graph_bouillard_burstyFirst['y']))
        list_bouillard_burstyFirst.sort()
        bouillard_burstyFirst_x, bouillard_burstyFirst_y = zip(*list_bouillard_burstyFirst)

        list_bouillard_burst = list(zip(graph_bouillard_burst['x'], graph_bouillard_burst['y']))
        list_bouillard_burst.sort()
        bouillard_burst_x, bouillard_burst_y = zip(*list_bouillard_burst)

        list_bouillard_rate = list(zip(graph_bouillard_rate['x'], graph_bouillard_rate['y']))
        list_bouillard_rate.sort()
        bouillard_rate_x, bouillard_rate_y = zip(*list_bouillard_rate)
        
        list_BL = list(zip(graph_BL['x'], graph_BL['y']))
        list_BL.sort()
        BL_x, BL_y = zip(*list_BL)

        datalist = [
            (PG_x, PG_y, {'label':'PG','linewidth':1, 'color':'red', 'marker':'s', 'linestyle':'-'}),
            (chang_x, chang_y, {'label':'Chang','linewidth':3, 'color':'blue', 'marker':'^', 'linestyle':'--'}),
            (bouillard_x, bouillard_y, {'label':'BBLC','linewidth':2.5, 'color':'magenta', 'marker':'o', 'linestyle':'-.'}),
            # (bouillard_new_x, bouillard_new_y, {'label': 'Bouillard_new','linewidth':2, 'color':'black', 'marker':'d', 'linestyle':'-.'}),
            # (bouillard_burstyFirst_x, bouillard_burstyFirst_y, {
            #     'label'    : 'BBLC_burstyFirst',
            #     'linewidth': 1.5,
            #     'color'    : 'cyan',
            #     'marker'   : 'd',
            #     'linestyle': '-.'
            # }), (bouillard_burst_x, bouillard_burst_y, {
            #     'label'    : 'BBLC_burst',
            #     'linewidth': 1,
            #     'color'    : 'tab:orange',
            #     'marker'   : 'd',
            #     'linestyle': '-.'
            # }), (bouillard_rate_x, bouillard_rate_y, {
            #     'label'    : 'BBLC_rate',
            #     'linewidth': 0.5,
            #     'color'    : 'brown',
            #     'marker'   : 'd',
            #     'linestyle': '-.'
            # }),
            (BL_x, BL_y, {'label':'BL','linewidth':3.5, 'color':'green', 'marker':'p', 'linestyle':':'})
        ]
        ticks = {
            'x': [],
            'y': np.arange(1.0, 18.0, 2.0)
        }
        self.plot(datalist, title=f"Impact of foi weight (weights: {eval(self.config['weights_mode'])})", xlabel='foi weight normalized', ylabel='delay bound',
                  text=(f"flow_rates: {[round(a.r,2) for a in alphas]}\n"
                        f"flow_bursts: {[round(a.b,2) for a in alphas]}\n"
                        f"flow_weights: {alphas_weights}\n"
                        f"foi_index: {foi_index}\n"
                        f"server: {str(beta)}\n"
                        f"utilization: {self.config['target_utilization']}\n"
                        f"flow_stability: {self.config['flow_stability']}\n"
                        f"No. of flows: {self.config['number_of_flows']}"),
                  ticks=ticks,
                  end=self.config['finally'])

    def plot_burstOfFoiByDelayBound(self):
        graph_PG = {
            'x': [],
            'y': []
        }
        graph_chang = {
            'x': [],
            'y': []
        }
        graph_bouillard = {
            'x': [],
            'y': []
        }
        graph_bouillard_new = {
            'x': [],
            'y': []
        }
        graph_bouillard_burstyFirst = {
            'x': [],
            'y': []
        }
        graph_bouillard_burst = {
            'x': [],
            'y': []
        }
        graph_bouillard_rate = {
            'x': [],
            'y': []
        }
        graph_BL = {
            'x': [],
            'y': []
        }
        # for foi_burst in tqdm([0.5, 0.75, 0.90, 1.00, 1.15, 1.30, 1.50, 2.00, 2.5, 3.0, 4.0, 5.0, 7.0, 8.0]):
        for foi_burst in tqdm(np.arange(10.0, 16.1, 0.5)):
            result = dict()
            t, alphas, alphas_weights, foi_index, beta = setupInputs(number_of_flows=self.config['number_of_flows'], weight_mode=eval(self.config['weights_mode']), target_util=self.config['target_utilization'],
                                                                     seed=eval(self.config['seed']), perFlowStability=self.config['flow_stability'],
                                                                     flow_rates = self.config['flow_rates'], flow_bursts = self.config['flow_bursts'], flow_weights = self.config['flow_weights'], \
                                                                     server_rate = self.config['server_rate'], server_delay =self.config['server_delay'])

            alphas[foi_index].b = foi_burst

            result["PG (General)"] = {
                'LoSC': GPS.LoSC_PG(sc=beta, index=foi_index, weights=alphas_weights)
            }
            result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])
            graph_PG['x'].append(alphas[foi_index].b)
            graph_PG['y'].append(result["PG (General)"]['delay bound'])

            subsets = powerset_non_empty_generator(list(range(len(alphas))))
            result["Chang"] = {
                'LoSC': GPS.LoSC_Chang_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subsets)
            }
            result["Chang"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Chang']['LoSC'][0])
            graph_chang['x'].append(alphas[foi_index].b)
            graph_chang['y'].append(result["Chang"]['delay bound'])

            result['Bouillard'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'][0])
            graph_bouillard['x'].append(alphas[foi_index].b)
            graph_bouillard['y'].append(result["Bouillard"]['delay bound'])

            result['Bouillard_new'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_new(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_new']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_new']['LoSC'][0])
            graph_bouillard_new['x'].append(alphas[foi_index].b)
            graph_bouillard_new['y'].append(result["Bouillard_new"]['delay bound'])

            result['Bouillard_burstyFirst'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_burstyFirst(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_burstyFirst']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_burstyFirst']['LoSC'][0])
            graph_bouillard_burstyFirst['x'].append(alphas[foi_index].b)
            graph_bouillard_burstyFirst['y'].append(result["Bouillard_burstyFirst"]['delay bound'])

            result['Bouillard_burst'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_burst(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_burst']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_burst']['LoSC'][0])
            graph_bouillard_burst['x'].append(alphas[foi_index].b)
            graph_bouillard_burst['y'].append(result["Bouillard_burst"]['delay bound'])

            result['Bouillard_rate'] = {
                'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound_rate(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
            }
            result['Bouillard_rate']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard_rate']['LoSC'][0])
            graph_bouillard_rate['x'].append(alphas[foi_index].b)
            graph_bouillard_rate['y'].append(result["Bouillard_rate"]['delay bound'])

            _subset_BL = powerset_non_empty_generator(list(range(len(alphas))))
            subset_BL = filter_generator(lambda x: foi_index in x, _subset_BL)
            result["Burchard, Liebeherr"] = {
                'LoSC': GPS.LoSC_BL_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subset_BL)
            }
            result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'][0])
            graph_BL['x'].append(alphas[foi_index].b)
            graph_BL['y'].append(result["Burchard, Liebeherr"]['delay bound'])

        # plot
        list_PG = list(zip(graph_PG['x'], graph_PG['y']))
        list_PG.sort()
        PG_x, PG_y = zip(*list_PG)
        list_chang = list(zip(graph_chang['x'], graph_chang['y']))
        list_chang.sort()
        chang_x, chang_y = zip(*list_chang)
        list_bouillard = list(zip(graph_bouillard['x'], graph_bouillard['y']))
        list_bouillard.sort()
        bouillard_x, bouillard_y = zip(*list_bouillard)
        list_bouillard_new = list(zip(graph_bouillard_new['x'], graph_bouillard_new['y']))
        list_bouillard_new.sort()
        bouillard_new_x, bouillard_new_y = zip(*list_bouillard_new)

        list_bouillard_burstyFirst = list(zip(graph_bouillard_burstyFirst['x'], graph_bouillard_burstyFirst['y']))
        list_bouillard_burstyFirst.sort()
        bouillard_burstyFirst_x, bouillard_burstyFirst_y = zip(*list_bouillard_burstyFirst)

        list_bouillard_burst = list(zip(graph_bouillard_burst['x'], graph_bouillard_burst['y']))
        list_bouillard_burst.sort()
        bouillard_burst_x, bouillard_burst_y = zip(*list_bouillard_burst)

        list_bouillard_rate = list(zip(graph_bouillard_rate['x'], graph_bouillard_rate['y']))
        list_bouillard_rate.sort()
        bouillard_rate_x, bouillard_rate_y = zip(*list_bouillard_rate)

        list_BL = list(zip(graph_BL['x'], graph_BL['y']))
        list_BL.sort()
        BL_x, BL_y = zip(*list_BL)

        datalist = [
            (PG_x, PG_y, {'label':'PG','linewidth':1, 'color':'red', 'marker':'s', 'linestyle':'-'}),
            (chang_x, chang_y, {'label':'Chang','linewidth':3, 'color':'blue', 'marker':'^', 'linestyle':'--'}),
            (bouillard_x, bouillard_y, {'label':'BBLC','linewidth':2.5, 'color':'magenta', 'marker':'o', 'linestyle':'-.'}),
            # (bouillard_new_x, bouillard_new_y, {'label': 'Bouillard_new','linewidth':2, 'color':'black', 'marker':'d', 'linestyle':'-.'}),
            # (bouillard_burstyFirst_x, bouillard_burstyFirst_y, {
            #     'label'    : 'BBLC_burstyFirst',
            #     'linewidth': 1.5,
            #     'color'    : 'cyan',
            #     'marker'   : 'd',
            #     'linestyle': '-.'
            # }), (bouillard_burst_x, bouillard_burst_y, {
            #     'label'    : 'BBLC_burst',
            #     'linewidth': 1,
            #     'color'    : 'tab:orange',
            #     'marker'   : 'd',
            #     'linestyle': '-.'
            # }), (bouillard_rate_x, bouillard_rate_y, {
            #     'label'    : 'BBLC_rate',
            #     'linewidth': 0.5,
            #     'color'    : 'brown',
            #     'marker'   : 'd',
            #     'linestyle': '-.'
            # }),
            (BL_x, BL_y, {'label':'BL','linewidth':3.5, 'color':'green', 'marker':'p', 'linestyle':':'})
        ]
        ticks = {
            'x': [],
            'y': np.arange(3.0, 8.0, 1.0)
        }
        util = self.config['target_utilization']
        if type(util) != float:
            util = round(sum([a.r for a in alphas]) / beta.rate ,2)
        self.plot(datalist, title=f"Impact of foi burst (weights: {eval(self.config['weights_mode'])})", xlabel='foi burst', ylabel='delay bound',
                  text=(f"flow_rates: {[round(a.r,2) for a in alphas]}\n"
                        f"flow_bursts: {[round(a.b,2) for a in alphas]}\n"
                        f"flow_weights: {alphas_weights}\n"
                        f"foi_index: {foi_index}\n"
                        f"server: {str(beta)}\n"
                        f"utilization: {util}\n"
                        f"flow_stability: {self.config['flow_stability']}\n"
                        f"No. of flows: {self.config['number_of_flows']}"),
                  # ticks=ticks,
                  end=self.config['finally'])



    def plot_flows_and_ti(self):
        t, alphas, alphas_weights, foi_index, beta = setupInputs(number_of_flows=self.config['number_of_flows'], weight_mode=eval(self.config['weights_mode']), target_util=0.75,
                                                                 seed=eval(self.config['seed']))

        # plot
        fig, ax = plt.subplots()
        t = np.arange(0., 10., 0.01)
        for i in range(5):
            ax.plot(t, [alphas[i].getCurve(c) for c in t], label=f'flow[{i}]')
            ax.plot(t, [GPS.Bouillard_ti(alphas_weights, beta, alphas, i=i).getCurve(c) for c in t], label=f'residual service[{i}]', linestyle='--')
            result = findIntersection(alphas[i].getCurve, GPS.Bouillard_ti(alphas_weights, beta, alphas, i=i).getCurve, 0.5)
            ax.plot(result, alphas[i].getCurve(result), 'ro', label=f'intersection')

        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels)

        ax.set(xlabel='time', ylabel='arrival', title=f"attempt to demonstrate Bouillards ordering")
        ax.grid()
        plt.show()

    def plot(self, datalist, title, xlabel, ylabel, ticks=None, start="None", end="None", text=None, figsize=(9,7)):
        eval(start)
        font = {
            'family': 'calibri',
            'size':20
        }
        plt.rc('font', **font)  # pass in the font dict as kwargs
        plt.figure(figsize=figsize)
        for x, y, misc in datalist:
            plt.plot(x, y, **misc)
        plt.title(title, fontdict={
            'fontsize': 20
        })
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.ylim(bottom=0)
        if ticks and len(ticks['x']):
            plt.xticks(ticks['x'])
        if ticks and len(ticks['y']):
            plt.yticks(ticks['y'])
        plt.legend(prop={"size":14})
        plt.grid()
        if text:
            plt.text(0.02,0.76,text, wrap=True, fontsize=15, transform=plt.gcf().transFigure)
            plt.subplots_adjust(top=0.68)
        eval(end)
        plt.show()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    clear_output()
    print("\n")
    print("==== Starting ... ====")
    print("\n")

    driver = RuntimeMeasurementAnalysisDriver()
    runMethods = []
    for i in [a for a in dir(driver) if callable(getattr(driver, a))]:
        if re.findall(driver.config['run'], i):
            runMethods.append(i)
    if len(runMethods) == 1:
        run = getattr(driver, runMethods[0])
        run()
    else:
        logging.warning('Error: more than 1 run method have been found.')

    print("\n")
    exit('==== analysis finished successfully. ====')
