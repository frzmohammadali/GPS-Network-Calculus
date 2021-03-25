import datetime
import logging
import sys
import numpy as np
import copy


from src.arrival_curve import ArrivalCurve, TokenBucket
from src.service_curve import ServiceCurve, RateLatency, WorkConservingLink
from src.nc import NC
from src.utilities import clear_output, clear_last_line, print_write, write, findIntersection

_v = False
class GPS:
    def __init__(self):
        pass

    @staticmethod
    def LoSC_PG(sc: ServiceCurve, index, weights):
        assert weights[index], "index does not exist"
        return RateLatency(R=(weights[index] / sum(weights)) * sc.rate, t=1, T=sc.delay)

    @staticmethod
    def LoSC_Chang(arrivals, sc: ServiceCurve, weights, foi, M):
        # remember that M contains list of indices of arrivals
        # arr_not_in_M = [arr for ix, arr in enumerate(arrivals) if ix not in M]
        arrNotInM_deconvBetaPG = [TokenBucket(b=arr.b + (arr.r * GPS.LoSC_PG(sc, ix, weights).delay),
                                         r=arr.r, t=1) for ix, arr in enumerate(arrivals) if ix not in M]
        arrivalsAgg = ArrivalCurve.aggregateTokenBuckets(*arrNotInM_deconvBetaPG)
        weights_in_M = [w for ix, w in enumerate(weights) if ix in M]
        # arrAgg_deconv_beta = TokenBucket(b=arrivalsAgg.b + (arrivalsAgg.r * sc.delay),
        #                                  r=arrivalsAgg.r, t=1)
        sc_minus_gamma: ServiceCurve = RateLatency(R=sc.rate - arrivalsAgg.r, t=1, T=(arrivalsAgg.b + (
                                                                                                    sc.rate * sc.delay)) / (
                                                                                                    sc.rate - arrivalsAgg.r))

        res = RateLatency(R=(weights[foi] / sum(weights_in_M)) * sc_minus_gamma.rate,
                           T=sc_minus_gamma.delay, t=1)

        # sanity check for including PG
        if len(M) == len(arrivals):
            loscPG = GPS.LoSC_PG(sc, foi, weights)
            if not (res.rate == loscPG.rate and res.delay == loscPG.delay):
                raise Exception("sanity check failed. Chang doesn't include PG")

        return res

    @staticmethod
    def LoSC_Bouillard(arrivals, sc: ServiceCurve, weights, new_foi, j):
        arrivals_current = copy.deepcopy(arrivals)
        weights_current = copy.deepcopy(weights)
        arrivals_current.insert(0, TokenBucket(r=0,b=0))
        weights_current.insert(0, 0)
        new_foi += 1
        arr_k = [arr for ix, arr in enumerate(arrivals_current) if ix in range(1, j + 1)]
        arrivals_currentAgg = ArrivalCurve.aggregateTokenBuckets(*arr_k)
        weights_current_k = [w for ix, w in enumerate(weights_current) if ix in range(j + 1, new_foi + 1)]
        sc_minus_gamma: ServiceCurve = RateLatency(R=sc.rate - arrivals_currentAgg.r, t=1,
                                                   T=(arrivals_currentAgg.b + (sc.rate * sc.delay)) / (
                                                           sc.rate - arrivals_currentAgg.r))

        ret = RateLatency(R=(weights_current[new_foi] / sum(weights_current_k)) * sc_minus_gamma.rate, t=1,
                          T=sc_minus_gamma.delay)
        new_foi -= 1

        if j == 0:
            # all flows are included so Bouillard and PG has to match
            pg_losc = GPS.LoSC_PG(sc, new_foi, weights_current_k)
            if pg_losc.R == ret.R and pg_losc.T == ret.T:
                # all fine
                pass
            else:
                raise Exception("Bouillard does not include PG")

        return ret

    @staticmethod
    def LoSC_Bouillard_new(arrivals, sc: ServiceCurve, weights, new_foi, j):
        arrivals_current = copy.deepcopy(arrivals)
        weights_current = copy.deepcopy(weights)
        arrivals_current.insert(0, TokenBucket(r=0,b=0))
        weights_current.insert(0, 0)
        new_foi += 1
        arr_k = [arr for ix, arr in enumerate(arrivals_current) if ix in range(1, j)]
        arrivals_currentAgg = ArrivalCurve.aggregateTokenBuckets(*arr_k)
        weights_current_k = [w for ix, w in enumerate(weights_current) if ix in range(j, new_foi + 1)]
        sc_minus_gamma: ServiceCurve = RateLatency(R=sc.rate - arrivals_currentAgg.r, t=1,
                                                   T=(arrivals_currentAgg.b + (sc.rate * sc.delay)) / (
                                                           sc.rate - arrivals_currentAgg.r))

        ret = RateLatency(R=(weights_current[new_foi] / sum(weights_current_k)) * sc_minus_gamma.rate, t=1,
                          T=sc_minus_gamma.delay)
        new_foi -= 1

        if j == 0:
            # all flows are included so Bouillard and PG has to match
            pg_losc = GPS.LoSC_PG(sc, new_foi, weights)
            if pg_losc.R == ret.R and pg_losc.T == ret.T:
                # all fine
                pass
            else:
                raise Exception("Bouillard does not include PG")

        return ret

    @staticmethod
    def Bouillard_ti(weights, sc, alphas, i):
        if i == 0:
            return 0
        share = weights[i] / sum([w for ix, w in enumerate(weights) if ix >= i])
        sum_alphas_j = ArrivalCurve.aggregateTokenBuckets(*[a for ix, a in enumerate(alphas) if ix <= i-2])
        sc_minus_gamma: ServiceCurve = RateLatency(R=sc.rate - sum_alphas_j.r, t=1, T=(sum_alphas_j.b + (sc.rate * sc.delay)) / (sc.rate - sum_alphas_j.r))

        right_side_function = RateLatency(R=share * sc_minus_gamma.rate, t=1, T=sc_minus_gamma.delay)
        for p in np.arange(0., 20., 0.01):
            intrst = findIntersection(alphas[i].getCurve, right_side_function.getCurve, p)
            try:
                assert intrst[0] > 0
                break
            except AssertionError:
                continue
        if not len(intrst):
            raise Exception('ti calculation error')

        return intrst[0]

    @staticmethod
    def Bouillard_ti_new(weights, sc, alphas, tj_val):

        arrivals_current = copy.deepcopy(alphas)
        weights_current = copy.deepcopy(weights)

        M = [ix for ix, arr in enumerate(arrivals_current)]
        M_removed = [ix for ix in M if ix not in range(tj_val + 1)]
        tj_list = []
        for tj_attained_ix in M_removed:
            share = weights_current[tj_attained_ix] / sum([w for ix, w in enumerate(weights_current) if ix >= tj_val + 1])
            sum_alphas_j = ArrivalCurve.aggregateTokenBuckets(*[a for ix, a in enumerate(arrivals_current) if ix <= tj_val - 1])
            beta_i_1: ServiceCurve = RateLatency(R=sc.rate - sum_alphas_j.r, t=1, T=(sum_alphas_j.b + (sc.rate * sc.delay)) / (sc.rate - sum_alphas_j.r))

            beta_share = RateLatency(R=share * beta_i_1.rate, t=1, T=beta_i_1.delay)
            my_t = ((beta_share.rate * beta_share.delay) + arrivals_current[tj_attained_ix].b) / (beta_share.rate - arrivals_current[tj_attained_ix].r)
            tj_list.append((my_t, tj_attained_ix))

        min_tj_list = np.inf
        min_tj_list_ix = -1
        for tj_val, tj_attained_ix in tj_list:
            if tj_list == None or tj_val <= min_tj_list:
                min_tj_list = tj_val
                min_tj_list_ix = tj_attained_ix

        return min_tj_list, min_tj_list_ix

    @staticmethod
    def LoSC_BL_Consistent_Chang(arrivals, sc: ServiceCurve, weights, foi, M):
        # remember that M contains list of indices of arrivals
        arr_not_in_M = [arr for ix, arr in enumerate(arrivals) if ix not in M]
        weights_in_M = [w for ix, w in enumerate(weights) if ix in M]
        arrivalsAgg = ArrivalCurve.aggregateTokenBuckets(*arr_not_in_M)
        sc_minus_gamma: ServiceCurve = RateLatency(R=sc.rate - arrivalsAgg.r, t=1,
                                                   T=(arrivalsAgg.b + (sc.rate * sc.delay)) / (
                                                           sc.rate - arrivalsAgg.r))

        res = RateLatency(R=(weights[foi] / sum(weights_in_M)) * sc_minus_gamma.rate, t=1,
                          T=sc_minus_gamma.delay)

        # sanity check for including PG
        if len(M) == len(arrivals):
            loscPG = GPS.LoSC_PG(sc, foi, weights)
            if not (res.rate == loscPG.rate and res.delay == loscPG.delay):
                raise Exception("sanity check failed. BL doesn't include PG")

        return res

    @staticmethod
    def LoSC_Chang_optimizeByMaxOverM(arrivals, sc: ServiceCurve, weights, foi, subsetlist):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(
                weights) and weights[
                    foi]), 'pre-conditions failed for GPS.LoSC_Chang_optimizeByMaxOverM(...)'

        beta_i = RateLatency(R=0.0, T=0.0, t=1)
        for M in subsetlist:
            if len(M) == 0:
                continue
            beta_candidate = GPS.LoSC_Chang(arrivals, sc, weights, foi, M)
            if beta_candidate.rate > beta_i.rate:
                beta_i = beta_candidate
        return beta_i

    @staticmethod
    def LoSC_Chang_optimizeByDelayBound(arrivals, sc: ServiceCurve, weights, foi, subsetlist):
        global _v
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(
                weights) and weights[
                    foi]), 'pre-conditions failed for GPS.LoSC_Chang_optimizeByDelayBound(...)'

        beta_i = None
        min_delay = None
        # in terms of delay
        best_m = None
        _iter = 1
        start = datetime.datetime.now()
        # mod = (((2 ** len(arrivals)) - 1) // 2500) if (((2 ** len(arrivals)) - 1) // 2500) != 0 \
        #         else 1
        mod = 100000
        for M in subsetlist:
            if len(M) == 0:
                continue
            if round(_iter % mod) == 0 and _v:
                clear_last_line()
                logging.debug(f"M: {_iter} of {(2**len(arrivals)) - 1}")
                percentage = round(_iter / ((2**len(arrivals)) - 1) * 100)
                print(f"calculating {'#'*percentage}{'-'*(abs(100-percentage))} {percentage}%")

            beta_candidate = GPS.LoSC_Chang(arrivals, sc, weights, foi, M)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[foi],
                                                                       beta_candidate)

            if min_delay is None or delay_candidate <= min_delay:
                # we ignore negative delay bounds as they are not reasonable
                if delay_candidate >= 0:
                    beta_i = beta_candidate
                    min_delay = delay_candidate
                    best_m = M
            _iter += 1

        write(f"M: {_iter-1} of {(2**len(arrivals)) - 1}")
        duration = datetime.datetime.now() - start
        if _v:
            print_write("total computation time: ",
                  ":".join([str(round(float(i))).zfill(2) for i in str(duration).split(":")]))
        return beta_i, f'best_m (len)={len(best_m)}', len(best_m), best_m

    @staticmethod
    def LoSC_Bouillard_optimizeByMaxOverM(arrivals, sc: ServiceCurve, weights, foi):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(
                weights) and weights[
                    foi]), 'pre-conditions failed for GPS.LoSC_Bouillard_optimizeByMaxOverM(...)'

        # re-indexing
        arr_foi = arrivals.pop(foi)
        arrivals.append(arr_foi)
        weights_foi = weights.pop(foi)
        weights.append(weights_foi)
        new_foi = len(arrivals) - 1

        beta_i = RateLatency(R=0.0, T=0.0, t=1)
        for j in range(new_foi):
            beta_candidate = GPS.LoSC_Bouillard(arrivals, sc, weights, new_foi, j)
            if beta_candidate.rate > beta_i.rate:
                beta_i = beta_candidate
        return beta_i

    @staticmethod
    def LoSC_Bouillard_optimizeByDelayBound(arrivals, sc: ServiceCurve, weights, foi):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(
                weights) and weights[
                    foi]), 'pre-conditions failed for GPS.LoSC_Bouillard_optimizeByDelayBound(...)'
        # re-indexing and sorting
        arr_foi = arrivals.pop(foi)
        weights_foi = weights.pop(foi)

        arrivals_ti_weight = []
        for i, a in enumerate(arrivals):
            ti = GPS.Bouillard_ti(weights, sc,arrivals, i)
            arrivals_ti_weight.append((a, ti, weights[i]))

        arrivals_ti_weight.sort(key=lambda x: x[1], reverse=False)
        arrivals_, _, weights_ = list(zip(*arrivals_ti_weight))
        arrivals = list(arrivals_)
        weights = list(weights_)

        arrivals.append(arr_foi)
        weights.append(weights_foi)
        new_foi = len(arrivals) - 1

        beta_i = None
        min_delay = None
        # in terms of delay
        best_j = None
        _iter = 1
        start = datetime.datetime.now()
        for j in range(new_foi):
            if _iter % 5 <= 5:
                clear_last_line()
                logging.debug(f"j: {_iter} of {new_foi}")
                percentage = round(_iter / new_foi * 100)
                print(f"calculating {'#'*percentage}{'-'*(abs(100-percentage))} {percentage}%")

            # ask whether all flows are included or not
            #if all included, then compare to PG result and they have to match
            # else throw error

            beta_candidate = GPS.LoSC_Bouillard(arrivals, sc, weights, new_foi, j)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[new_foi],
                                                                       beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                # we ignore negative delay bounds as they are not reasonable
                if delay_candidate >= 0:
                    beta_i = beta_candidate
                    min_delay = delay_candidate
                    best_j = j
            _iter += 1

        write(f"j: {_iter-1} of {new_foi}")
        duration = datetime.datetime.now() - start
        print_write("total computation time: ",":".join([str(round(float(i))).zfill(2) for i in str(
                duration).split(":")]))
        return beta_i, f'best_j={best_j}'

    @staticmethod
    def LoSC_Bouillard_optimizeByDelayBound_new(arrivals, sc: ServiceCurve, weights, foi):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(weights) and weights[foi]), 'pre-conditions failed for GPS.LoSC_Bouillard_optimizeByDelayBound(...)'
        # re-indexing and sorting
        arr_foi = arrivals.pop(foi)
        weights_foi = weights.pop(foi)

        arrivals.insert(0, TokenBucket(r=0, b=0))
        weights.insert(0, 0)

        for i in range(len(arrivals) - 1):
            ti_p1, ti_p1_ix = GPS.Bouillard_ti_new(weights, sc, arrivals, i)
            # swap
            ai_p1 = arrivals[i+1]
            wi_p1 = weights[i+1]
            arrivals[i+1] = arrivals[ti_p1_ix]
            arrivals[ti_p1_ix] = ai_p1
            weights[i+1] = weights[ti_p1_ix]
            weights[ti_p1_ix] = wi_p1
            # after the loop "arrivals" and "weights" are already sorted

        arrivals.pop(0)
        weights.pop(0)

        arrivals.append(arr_foi)
        weights.append(weights_foi)
        new_foi = len(arrivals) - 1

        beta_i = None
        min_delay = None
        # in terms of delay
        best_j = None
        _iter = 1
        start = datetime.datetime.now()
        for j in range(new_foi):
            if _iter % 5 <= 5:
                clear_last_line()
                logging.debug(f"j: {_iter} of {new_foi}")
                percentage = round(_iter / new_foi * 100)
                print(f"calculating {'#'*percentage}{'-'*(abs(100-percentage))} {percentage}%")


            beta_candidate = GPS.LoSC_Bouillard_new(arrivals, sc, weights, new_foi, j)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[new_foi], beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                # we ignore negative delay bounds as they are not reasonable
                if delay_candidate >= 0:
                    beta_i = beta_candidate
                    min_delay = delay_candidate
                    best_j = j
            _iter += 1

        write(f"j: {_iter-1} of {new_foi}")
        duration = datetime.datetime.now() - start
        print_write("total computation time: ", ":".join([str(round(float(i))).zfill(2) for i in str(duration).split(":")]))
        return beta_i, f'best_j={best_j}'

    @staticmethod
    def LoSC_Bouillard_optimizeByDelayBound_burstyFirst(arrivals, sc: ServiceCurve, weights, foi):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(weights) and weights[foi]), 'pre-conditions failed for GPS.LoSC_Bouillard_optimizeByDelayBound(...)'
        # re-indexing and sorting
        arr_foi = arrivals.pop(foi)
        weights_foi = weights.pop(foi)

        # heuristic for sorting here
        # b/r is the parameter we use here (ascending)
        arrivals_weights = list(zip(arrivals, weights))
        arrivals_weights.sort(key=lambda x: x[0].b/x[0].r)
        arrivals_, weights_ = list(zip(*arrivals_weights))
        arrivals = list(arrivals_)
        weights = list(weights_)

        arrivals.append(arr_foi)
        weights.append(weights_foi)
        new_foi = len(arrivals) - 1

        beta_i = None
        min_delay = None
        # in terms of delay
        best_j = None
        _iter = 1
        start = datetime.datetime.now()
        for j in range(new_foi):
            if _iter % 5 <= 5:
                clear_last_line()
                logging.debug(f"j: {_iter} of {new_foi}")
                percentage = round(_iter / new_foi * 100)
                print(f"calculating {'#'*percentage}{'-'*(abs(100-percentage))} {percentage}%")

            beta_candidate = GPS.LoSC_Bouillard_new(arrivals, sc, weights, new_foi, j)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[new_foi], beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                # we ignore negative delay bounds as they are not reasonable
                if delay_candidate >= 0:
                    beta_i = beta_candidate
                    min_delay = delay_candidate
                    best_j = j
            _iter += 1

        write(f"j: {_iter-1} of {new_foi}")
        duration = datetime.datetime.now() - start
        print_write("total computation time: ", ":".join([str(round(float(i))).zfill(2) for i in str(duration).split(":")]))
        return beta_i, f'best_j={best_j}'

    @staticmethod
    def LoSC_Bouillard_optimizeByDelayBound_burst(arrivals, sc: ServiceCurve, weights, foi):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(weights) and weights[foi]), 'pre-conditions failed for GPS.LoSC_Bouillard_optimizeByDelayBound(...)'
        # re-indexing and sorting
        arr_foi = arrivals.pop(foi)
        weights_foi = weights.pop(foi)

        # heuristic for sorting here
        # b is the parameter we use here (descending)
        arrivals_weights = list(zip(arrivals, weights))
        arrivals_weights.sort(key=lambda x: x[0].b, reverse=True)
        arrivals_, weights_ = list(zip(*arrivals_weights))
        arrivals = list(arrivals_)
        weights = list(weights_)

        arrivals.append(arr_foi)
        weights.append(weights_foi)
        new_foi = len(arrivals) - 1

        beta_i = None
        min_delay = None
        # in terms of delay
        best_j = None
        _iter = 1
        start = datetime.datetime.now()
        for j in range(new_foi):
            if _iter % 5 <= 5:
                clear_last_line()
                logging.debug(f"j: {_iter} of {new_foi}")
                percentage = round(_iter / new_foi * 100)
                print(f"calculating {'#'*percentage}{'-'*(abs(100-percentage))} {percentage}%")

            beta_candidate = GPS.LoSC_Bouillard_new(arrivals, sc, weights, new_foi, j)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[new_foi], beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                # we ignore negative delay bounds as they are not reasonable
                if delay_candidate >= 0:
                    beta_i = beta_candidate
                    min_delay = delay_candidate
                    best_j = j
            _iter += 1

        write(f"j: {_iter-1} of {new_foi}")
        duration = datetime.datetime.now() - start
        print_write("total computation time: ", ":".join([str(round(float(i))).zfill(2) for i in str(duration).split(":")]))
        return beta_i, f'best_j={best_j}'

    @staticmethod
    def LoSC_Bouillard_optimizeByDelayBound_rate(arrivals, sc: ServiceCurve, weights, foi):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(weights) and weights[foi]), 'pre-conditions failed for GPS.LoSC_Bouillard_optimizeByDelayBound(...)'
        # re-indexing and sorting
        arr_foi = arrivals.pop(foi)
        weights_foi = weights.pop(foi)

        # heuristic for sorting here
        # r is the parameter we use here (descending)
        arrivals_weights = list(zip(arrivals, weights))
        arrivals_weights.sort(key=lambda x: x[0].r, reverse=True)
        arrivals_, weights_ = list(zip(*arrivals_weights))
        arrivals = list(arrivals_)
        weights = list(weights_)

        arrivals.append(arr_foi)
        weights.append(weights_foi)
        new_foi = len(arrivals) - 1

        beta_i = None
        min_delay = None
        # in terms of delay
        best_j = None
        _iter = 1
        start = datetime.datetime.now()
        for j in range(new_foi):
            if _iter % 5 <= 5:
                clear_last_line()
                logging.debug(f"j: {_iter} of {new_foi}")
                percentage = round(_iter / new_foi * 100)
                print(f"calculating {'#'*percentage}{'-'*(abs(100-percentage))} {percentage}%")

            beta_candidate = GPS.LoSC_Bouillard_new(arrivals, sc, weights, new_foi, j)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[new_foi], beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                # we ignore negative delay bounds as they are not reasonable
                if delay_candidate >= 0:
                    beta_i = beta_candidate
                    min_delay = delay_candidate
                    best_j = j
            _iter += 1

        write(f"j: {_iter-1} of {new_foi}")
        duration = datetime.datetime.now() - start
        print_write("total computation time: ", ":".join([str(round(float(i))).zfill(2) for i in str(duration).split(":")]))
        return beta_i, f'best_j={best_j}'

    @staticmethod
    def LoSC_BL_optimizeByMaxOverM(arrivals, sc: ServiceCurve, weights, foi, subsetlist):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(
                weights) and weights[
                    foi]), 'pre-conditions failed for GPS.LoSC_BL_optimizeByMaxOverM(...)'

        beta_i = RateLatency(R=0.0, T=0.0, t=1)
        for M in subsetlist:
            if len(M) == 0:
                continue
            beta_candidate = GPS.LoSC_BL_Consistent_Chang(arrivals, sc, weights, foi, M)
            if beta_candidate.rate > beta_i.rate:
                beta_i = beta_candidate
        return beta_i

    @staticmethod
    def LoSC_BL_optimizeByDelayBound(arrivals, sc: ServiceCurve, weights, foi, subsetlist):
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(
                weights) and weights[
                    foi]), 'pre-conditions failed for GPS.LoSC_BL_optimizeByDelayBound(...)'

        beta_i = None
        min_delay = None
        # in terms of delay
        best_m = None
        _iter = 1
        start = datetime.datetime.now()
        # mod = ((2 ** (len(arrivals) - 1)) // 2000) if ((2 ** (len(
        #     arrivals) - 1)) // 2000) != 0 else 1
        mod = 100000
        for M in subsetlist:
            if len(M) == 0:
                continue
            if round(_iter % mod) == 0:
                clear_last_line()
                logging.debug(f"M: {_iter} of {(2**(len(arrivals) - 1))}")
                percentage = round(_iter / (2 ** (len(arrivals) - 1)) * 100)
                print(f"calculating {'#'*percentage}{'-'*(abs(100-percentage))} {percentage}%")

            beta_candidate = GPS.LoSC_BL_Consistent_Chang(arrivals, sc, weights, foi, M)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[foi],
                                                                       beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                # we ignore negative delay bounds as they are not reasonable
                if delay_candidate >= 0:
                    beta_i = beta_candidate
                    min_delay = delay_candidate
                    best_m = M
            _iter += 1

        write(f"M: {_iter-1} of {(2**(len(arrivals) - 1))}")
        duration = datetime.datetime.now() - start
        print_write("total computation time: ",
              ":".join([str(round(float(i))).zfill(2) for i in str(duration).split(":")]))
        return beta_i, f'best_m (len)={len(best_m)}'
