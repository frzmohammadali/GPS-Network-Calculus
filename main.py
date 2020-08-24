import copy
import logging
import random
import time

logging.basicConfig(
        format="[%(levelname)s] %(message)s (%(filename)s, %(funcName)s(), line %(lineno)d, %(asctime)s)",
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,  # level=logging.DEBUG,
)

import os, sys

sys.path.append(os.path.abspath('../'))
from src.arrival_curve import TokenBucket
from src.service_curve import WorkConservingLink, RateLatency
from src.gps import GPS
from src.nc import NC
from src.utilities import distinct_by, powerset_non_empty_generator, length_distinct_subsets, \
    ReturnType, WeightsMode, clear_output


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
    result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])

    subsets = length_distinct_subsets(alphas)
    result["Chang (homogeneous-optimised)"] = {
        'LoSC': GPS.LoSC_Chang_optimizeByMaxOverM(arrivals=alphas, sc=beta, weights=alphas_weights,
                                                  foi=foi_index, subsetlist=subsets)
    }
    result["Chang (homogeneous-optimised)"][
        'delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=
    result['Chang (homogeneous-optimised)']['LoSC'])

    result['Bouillard'] = {
        'LoSC': GPS.LoSC_Bouillard_optimizeByMaxOverM(arrivals=alphas, sc=beta,
                                                      weights=alphas_weights, foi=foi_index)
    }
    result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'])

    arrivals_index = list(range(len(alphas)))
    arrivals_index.pop(foi_index)
    subset_Burchard_Liebeherr = length_distinct_subsets(arrivals_index, return_type=ReturnType.ITEM)
    result["Burchard, Liebeherr"] = {
        'LoSC': GPS.LoSC_BL_optimizeByMaxOverM(arrivals=alphas, sc=beta, weights=alphas_weights,
                                               foi=foi_index, subsetlist=subset_Burchard_Liebeherr)
    }
    result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'])

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
    alphas = [TokenBucket(b=2, r=30, t=t) for _ in
              range(number_of_flows)]
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
    result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])

    subsets = length_distinct_subsets(alphas, return_type=ReturnType.INDEX, subseteq=True)
    result["Chang (homogeneous-optimised)"] = {
        'LoSC': GPS.LoSC_Chang_optimizeByDelayBound(arrivals=alphas, sc=beta,
                                                    weights=alphas_weights, foi=foi_index,
                                                    subsetlist=subsets)
    }
    result["Chang (homogeneous-optimised)"][
        'delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=
    result['Chang (homogeneous-optimised)']['LoSC'][0])

    result['Bouillard'] = {
        'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound(arrivals=copy.deepcopy(alphas), sc=beta,
                                                        weights=copy.deepcopy(alphas_weights),
                                                        foi=foi_index)
    }
    result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'][0])

    # we change N\{i} to N such that always i in M to make its semantic consistent with chang
    # semantic
    _subset_BL = length_distinct_subsets(alphas, return_type=ReturnType.INDEX, subseteq=True)
    subset_BL = list(filter(lambda x: foi_index in x, _subset_BL))
    result["Burchard, Liebeherr"] = {
        'LoSC': GPS.LoSC_BL_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights,
                                                 foi=foi_index, subsetlist=subset_BL)
    }
    result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'][0])

    time.sleep(0.5)
    print()
    print("number of arrivals:", len(alphas))
    for key, value in result.items():
        print(f'{key: <30}', ": ", value)


# homogeneous arrivals analysis optimized by delay bound (min)
def hetro_arr_analysis_OBDB(number_of_flows, weight_mode: WeightsMode):
    result = dict()
    t = 1
    # b=random(1,5) Mb, r=random(3,30) Mb/s, R=2400 Mb/s (2.4 Gb/s), T=2.0 seconds
    # [max number of flows would be 79 for stability condition to be satisfied]
    random.seed(10)
    alphas = [TokenBucket(b=random.randint(1,5), r=random.randint(3,30), t=t) for _ in
              range(number_of_flows)]
    # let's set weights to be equal to 1
    if weight_mode == WeightsMode.EQUAL:
        alphas_weights = [1 for _ in alphas]
    elif weight_mode == WeightsMode.RPPS:
        alphas_weights = [__a.r for __a in alphas]
    elif weight_mode == WeightsMode.RANDOM:
        alphas_weights = [random.randint(1,5) for _ in alphas]
    else:
        raise Exception("WeightMode not recognized : " + weight_mode)

    foi_index = 1
    # beta = WorkConservingLink(c=2400,t=t)
    beta = RateLatency(R=2400, T=2.0, t=t)
    # stability
    assert beta.rate - (sum([a.r for a in alphas])) > 0, 'sys not stable'

    result["PG (General)"] = {
        'LoSC': GPS.LoSC_PG(sc=beta, index=foi_index, weights=alphas_weights)
    }
    result["PG (General)"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['PG (General)']['LoSC'])

    print("Chang")
    print("-----\n\n")
    subsets = powerset_non_empty_generator(list(range(len(alphas))))
    result["Chang (homogeneous-optimised)"] = {
        'LoSC': GPS.LoSC_Chang_optimizeByDelayBound(arrivals=alphas, sc=beta,
                                                    weights=alphas_weights, foi=foi_index,
                                                    subsetlist=subsets)
    }
    result["Chang (homogeneous-optimised)"][
        'delay bound'] = NC.delay_bound_token_bucket_rate_latency(alpha=alphas[foi_index], beta=
    result['Chang (homogeneous-optimised)']['LoSC'][0])
    print("--done\n\n")

    print("Bouillard")
    print("---------\n\n")
    result['Bouillard'] = {
        'LoSC': GPS.LoSC_Bouillard_optimizeByDelayBound(arrivals=copy.deepcopy(alphas), sc=beta,
                                                        weights=copy.deepcopy(alphas_weights),
                                                        foi=foi_index)
    }
    result['Bouillard']['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['Bouillard']['LoSC'][0])
    print("--done\n\n")

    print("BL")
    print("--\n\n")
    # we change N\{i} to N such that always i in M to make its semantic consistent with chang
    # semantic
    _subset_BL = powerset_non_empty_generator(list(range(len(alphas))))
    subset_BL = list(filter(lambda x: foi_index in x, _subset_BL))
    result["Burchard, Liebeherr"] = {
        'LoSC': GPS.LoSC_BL_optimizeByDelayBound(arrivals=alphas, sc=beta, weights=alphas_weights,
                                                 foi=foi_index, subsetlist=subset_BL)
    }
    result["Burchard, Liebeherr"]['delay bound'] = NC.delay_bound_token_bucket_rate_latency(
            alpha=alphas[foi_index], beta=result['Burchard, Liebeherr']['LoSC'][0])
    print("--done\n\n")
    time.sleep(0.5)
    print()
    print("number of arrivals:", len(alphas))
    print('flow of interest:', alphas[foi_index])
    print('weights mode:', str(weight_mode))
    print("distinct weights: ", list(set(alphas_weights)))
    for key, value in result.items():
        print(f'{key: <30}', ": ", value)
    print("\n\n\n")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    clear_output()
    print("\n")
    print("==========================")
    print("==== Analysis Started ====")
    print("==========================")
    print("\n")

    hetro_arr_analysis_OBDB(24, WeightsMode.EQUAL)
    hetro_arr_analysis_OBDB(24, WeightsMode.RPPS)
    hetro_arr_analysis_OBDB(24, WeightsMode.RANDOM)
