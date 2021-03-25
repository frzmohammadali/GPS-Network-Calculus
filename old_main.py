import argparse
import copy
import json
import logging
import random
import time
from numpy import mean, median
import matplotlib
import matplotlib.pyplot as plt

logging.basicConfig(format="[%(levelname)s] %(message)s (%(filename)s, %(funcName)s(), line %(lineno)d, %(asctime)s)", datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,  # level=logging.DEBUG,
                    )

import os, sys

sys.path.append(os.path.abspath('../'))
from src.arrival_curve import TokenBucket
from src.service_curve import WorkConservingLink, RateLatency
from src.gps import GPS
from src.nc import NC
from src.utilities import distinct_by, powerset_non_empty_generator, length_distinct_subsets, ReturnType, WeightsMode, clear_output, filter_generator, print_write


# homogeneous arrivals analysis optimized by leftover service curve rate (max)
def homo_arr_analysis_OBSCRate(number_of_flows):
    result = dict()
    t = 1
    # b=1 Mb, r=30 Mb/s, C=2400 Mb/s (2.4 Gb/s) [max number of flows would be 79 for stability
    # condition to be satisfied]
    alphas = [TokenBucket(b=1, r=30, t=t) for _ in range(number_of_flows)]
    # let's use RPPS
    __arrival_rates = [a.r for a in alphas]
    alphas_weights = __arrival_rates
    foi_index = 1
    beta = RateLatency(R=2400, T=2, t=t)
    # stability
    assert beta.rate - (sum(__arrival_rates)) > 0, 'sys not stable'

    result["PG (General)"] = {
        'LoSC': GPS.LoSC_PG(sc=beta, index=foi_index, weights=alphas_weights)
    }
    result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])

    subsets = length_distinct_subsets(alphas)
    result["Chang (homogeneous-optimised)"] = {
        'LoSC': GPS.LoSC_Chang_optimizeByMaxOverM(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subsets)
    }
    result["Chang (homogeneous-optimised)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Chang (homogeneous-optimised)']['LoSC'])

    result['Bouillard'] = {
        'LoSC': GPS.LoSC_Bouillard_optimizeByMaxOverM(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index)
    }
    result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'])

    arrivals_index = list(range(len(alphas)))
    arrivals_index.pop(foi_index)
    subset_Burchard_Liebeherr = length_distinct_subsets(arrivals_index, return_type=ReturnType.ITEM)
    result["Burchard, Liebeherr"] = {
        'LoSC': GPS.LoSC_BL_optimizeByMaxOverM(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subset_Burchard_Liebeherr)
    }
    result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'])

    time.sleep(0.5)
    print()
    print("number of arrivals:", len(alphas))
    for key, value in result.items():
        print(f'{key: <30}', ": ", value)

    print('\n', 'markdown table row:')
    print(f"| {len(alphas)} | homogeneous | doesn't matter | "
          f"rate={round(result['PG (General)']['LoSC'].rate,4)} "
          f"latency={round(result['PG (General)']['LoSC'].delay,4)} "
          f"delay_bound={result['PG (General)']['delay bound']} | "
          f"rate={round(result['Chang (homogeneous-optimised)']['LoSC'].rate,4)} "
          f"latency={round(result['Chang (homogeneous-optimised)']['LoSC'].delay,4)} "
          f"delay_bound={result['Chang (homogeneous-optimised)']['delay bound']} |"
          f"rate={round(result['Bouillard']['LoSC'].rate,4)} "
          f"latency={round(result['Bouillard']['LoSC'].delay,4)} "
          f"delay_bound={result['Bouillard']['delay bound']} | "
          f"rate={round(result['Burchard, Liebeherr']['LoSC'].rate,4)} "
          f"latency={round(result['Burchard, Liebeherr']['LoSC'].delay,4)} "
          f"delay_bound={result['Burchard, Liebeherr']['delay bound']} |")


# homogeneous arrivals analysis optimized by delay bound (min)
def homo_arr_analysis_OBDB(number_of_flows):
    result = dict()
    t = 1
    # b=1 Mb, r=30 Mb/s, C=2400 Mb/s (2.4 Gb/s)
    # [max number of flows would be 79 for stability condition to be satisfied]
    random.seed(30)
    alphas = [TokenBucket(b=2, r=30, t=t) for _ in range(number_of_flows)]
    # weights have to be equal1
    alphas_weights = [1 for _ in alphas]
    foi_index = 1
    # beta = WorkConservingLink(c=2400,t=t)
    beta = RateLatency(R=2400, T=2.0, t=t)
    # stability
    assert beta.rate - (sum([a.r for a in alphas])) > 0, 'sys not stable'

    result["PG (General)"] = {
        'LoSC': GPS.LoSC_PG(sc=beta, index=foi_index, weights=alphas_weights)
    }
    result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])

    subsets = length_distinct_subsets(alphas, return_type=ReturnType.INDEX, subseteq=True)
    result["Chang (homogeneous-optimised)"] = {
        'LoSC': GPS.LoSC_Chang_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subsets)
    }
    result["Chang (homogeneous-optimised)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Chang (homogeneous-optimised)']['LoSC'][0])

    result['Bouillard'] = {
        'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound(arrivals=copy.deepcopy(alphas), sc=beta, weights=copy.deepcopy(alphas_weights), foi=foi_index)
    }
    result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'][0])

    # we change N\{i} to N such that always i in M to make its semantic consistent with chang
    # semantic
    _subset_BL = length_distinct_subsets(alphas, return_type=ReturnType.INDEX, subseteq=True)
    subset_BL = list(filter(lambda x: foi_index in x, _subset_BL))
    result["Burchard, Liebeherr"] = {
        'LoSC': GPS.LoSC_BL_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subset_BL)
    }
    result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'][0])

    time.sleep(0.5)
    print()
    print("number of arrivals:", len(alphas))
    for key, value in result.items():
        print(f'{key: <30}', ": ", value)


def setupInputs(number_of_flows, weight_mode: WeightsMode, target_util=0.75):
    t = 1
    # b=random(1,5) Mb, r=random(3,30) Mb/s, R=2400 Mb/s (2.4 Gb/s), T=2.0 seconds
    #  [max number of flows would be 79 for stability condition to be satisfied]
    random.seed(10)
    # fix b
    # alphas = [TokenBucket(b=0.5, r=random.uniform(0.5, 5.0), t=t) for _ in
    #           range(number_of_flows)]
    # random b
    alphas = [TokenBucket(b=random.uniform(0.1, 1.5), r=1, t=t) for _ in range(number_of_flows)]

    if weight_mode == WeightsMode.EQUAL:
        alphas_weights = [1 for _ in alphas]
    elif weight_mode == WeightsMode.RPPS:
        alphas_weights = [__a.r for __a in alphas]
    elif weight_mode == WeightsMode.RANDOM:
        alphas_weights = [random.uniform(0.0, 1.0) for _ in alphas]
    else:
        raise Exception("WeightMode not recognized : " + weight_mode)

    foi_index = 1
    target_util = target_util
    agg_arr_rate = (sum([a.r for a in alphas]))
    # beta = WorkConservingLink(c=2400,t=t)
    # latency is usually boring, but add the sanity check, if safe set it to zero
    beta = RateLatency(R=agg_arr_rate / target_util, T=0.0, t=t)
    # stability
    assert beta.rate - (sum([a.r for a in alphas])) > 0, 'sys not stable'

    # print('utilization of the scenario:' + str((sum([a.r for a in alphas])) / beta.rate))
    # print('beta rate: ' + str(beta.rate))

    return t, alphas, alphas_weights, foi_index, beta


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


def get_train_best_M_data(number_of_flows, weight_mode):
    result = dict()
    t, alphas, alphas_weights, foi_index, beta = setupInputs(number_of_flows, weight_mode)
    subsets = powerset_non_empty_generator(list(range(len(alphas))))
    result["Chang"] = {
        'LoSC': GPS.LoSC_Chang_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights, foi=foi_index, subsetlist=subsets)
    }
    result["Chang"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=result['Chang']['LoSC'][0])

    __quarter_index = 1 if len(alphas) // 4 == 0 else len(alphas) // 4
    # return length of best M
    return (result["Chang"]['LoSC'][2], 'mean weights=' + str(round(mean(alphas_weights), 3)),
            'weight above mean selected=' + str(get_above_selected_percentage([w for ix, w in enumerate(alphas_weights) if ix in result["Chang"]['LoSC'][3]], alphas_weights)) + "%",
            'mean b=' + str(round(mean([a.b for a in alphas]), 3)), 'b above mean selected=' + str(
            get_above_selected_percentage([a.b for ix, a in enumerate(alphas) if ix in result["Chang"]['LoSC'][3]], [a.b for ix, a in enumerate(alphas) if ix in result["Chang"]['LoSC'][3]],
                                          round(mean([a.b for a in alphas]), 3))) + "%", {
                'mean(weights)'       : round(mean(alphas_weights), 3),
                'weight_above'        : get_above_selected_percentage([w for ix, w in enumerate(alphas_weights) if ix in result["Chang"]['LoSC'][3]], alphas_weights),
                'weight_above_quarter': get_above_selected_percentage([w for ix, w in enumerate(alphas_weights) if ix in result["Chang"]['LoSC'][3]],
                                                                      [w for ix, w in enumerate(alphas_weights) if ix in result["Chang"]['LoSC'][3]],
                                                                      (sorted([w for w in alphas_weights], reverse=True)[0:__quarter_index])[-1]),
                'weight_lower_quarter': get_lower_than_selected_percentage([w for ix, w in enumerate(alphas_weights) if ix in result["Chang"]['LoSC'][3]],
                                                                           [w for ix, w in enumerate(alphas_weights) if ix in result["Chang"]['LoSC'][3]],
                                                                           (sorted([w for w in alphas_weights], reverse=False)[0:__quarter_index])[-1]),
                'mean(b)'             : round(mean([a.b for a in alphas]), 3),
                'b_above'             : get_above_selected_percentage([a.b for ix, a in enumerate(alphas) if ix in result["Chang"]['LoSC'][3]],
                                                                      [a.b for ix, a in enumerate(alphas) if ix in result["Chang"]['LoSC'][3]], round(mean([a.b for a in alphas]), 3)),
                'b_above_quarter'     : get_above_selected_percentage([a.b for ix, a in enumerate(alphas) if ix in result["Chang"]['LoSC'][3]],
                                                                      [a.b for ix, a in enumerate(alphas) if ix in result["Chang"]['LoSC'][3]],
                                                                      (sorted([a.b for a in alphas], reverse=True)[0:__quarter_index])[-1]),
                'b_lower_quarter'     : get_lower_than_selected_percentage([a.b for ix, a in enumerate(alphas) if ix in result["Chang"]['LoSC'][3]],
                                                                           [a.b for ix, a in enumerate(alphas) if ix in result["Chang"]['LoSC'][3]],
                                                                           (sorted([a.b for a in alphas], reverse=False)[0:__quarter_index])[-1])
            })


def get_above_selected_percentage(subl, l, m=None):
    if not m:
        m = mean(l)
    return round(len([i for i in subl if i >= m]) / len(l) * 100, 2)


def get_lower_than_selected_percentage(subl, l, m=None):
    if not m:
        m = mean(l)
    return round(len([i for i in subl if i < m]) / len(l) * 100, 2)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    clear_output()
    print("\n")
    print("==========================")
    print("==== Analysis Started ====")
    print("==========================")
    print("\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("-wm", "--weight_mode", help="define execution mode of the script. could "
                                                     "be "
                                                     "['all', 'ew', 'rpps', 'rand']")
    parser.add_argument("-m", "--mode", help="define train or predict mode of the script. could be "
                                             "['train', 'predict']")
    args = parser.parse_args()
    if not args.weight_mode:
        args.weight_mode = "rpps"
    if args.mode and args.mode == 'train':
        print('analysis mode: train\n')
        lbm_res = []
        for i in range(2, 17):
            lbm = get_train_best_M_data(i, WeightsMode.RANDOM)
            print('optimal length: ' + str(lbm[0]) + f' out of {i} flows | {lbm[1]} | {lbm[2]} | '
                                                     f'{lbm[3]} | {lbm[4]}')
            # if not boring print parameters
            if lbm[0] != i:
                print()

            lbm_res.append(lbm[5])

        print()
        print(f'how often weight above average gets selected='
              f'{round(mean([l["weight_above"] for l in lbm_res]),2)}%')
        print(f'how often weight gets selected from highest quarter='
              f'{round(mean([l["weight_above_quarter"] for l in lbm_res]),2)}%')
        print(f'how often weight gets selected from lowest quarter='
              f'{round(mean([l["weight_lower_quarter"] for l in lbm_res]),2)}%')
        print(f'how often b above average gets selected='
              f'{round(mean([l["b_above"] for l in lbm_res]),2)}%')
        print(f'how often b gets selected from highest quarter='
              f'{round(mean([l["b_above_quarter"] for l in lbm_res]),2)}%')
        print(f'how often b gets selected from lowest quarter='
              f'{round(mean([l["b_lower_quarter"] for l in lbm_res]),2)}%')
    elif args.mode and args.mode == 'predict':
        print('analysis mode: predict\n not implemented yet')
    else:
        if not args.weight_mode or args.weight_mode == 'all':
            print('analysis weight mode: all\n')
            hetro_arr_analysis_OBDB(32, WeightsMode.EQUAL)
            hetro_arr_analysis_OBDB(32, WeightsMode.RPPS)
            hetro_arr_analysis_OBDB(32, WeightsMode.RANDOM)
        elif args.mode == 'ew':
            print('analysis weight mode: equal weights\n')
            hetro_arr_analysis_OBDB(32, WeightsMode.EQUAL)
        elif args.mode == 'rpps':
            print('analysis weight mode: rpps\n')
            hetro_arr_analysis_OBDB(32, WeightsMode.RPPS)
        elif args.mode == 'rand':
            print('analysis weight mode: rand\n')
            hetro_arr_analysis_OBDB(32, WeightsMode.RANDOM)





