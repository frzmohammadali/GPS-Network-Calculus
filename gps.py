import logging

from src.arrival_curve import ArrivalCurve, TokenBucket
from src.service_curve import ServiceCurve, RateLatency, WorkConservingLink
from src.nc import NC


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
        arr_not_in_M = [arr for ix, arr in enumerate(arrivals) if ix not in M]
        weights_in_M = [w for ix, w in enumerate(weights) if ix in M]
        arrivalsAgg = ArrivalCurve.aggregateTokenBuckets(*arr_not_in_M)
        arrAgg_deconv_beta = TokenBucket(b=arrivalsAgg.b + (arrivalsAgg.r * sc.delay),
                                         r=arrivalsAgg.r, t=1)
        sc_minus_gamma: ServiceCurve = RateLatency(R=sc.rate - arrAgg_deconv_beta.r, t=1, T=(
                                                                                                    arrAgg_deconv_beta.b + (
                                                                                                    sc.rate * sc.delay)) / (
                                                                                                    sc.rate - arrAgg_deconv_beta.r))

        return RateLatency(R=(weights[foi] / sum(weights_in_M)) * sc_minus_gamma.rate,
                           T=sc_minus_gamma.delay, t=1)

    @staticmethod
    def LoSC_Bouillard(arrivals, sc: ServiceCurve, weights, new_foi, j):
        arr_k = [arr for ix, arr in enumerate(arrivals) if ix in range(1, j + 1)]
        arrivalsAgg = ArrivalCurve.aggregateTokenBuckets(*arr_k)
        weights_k = [w for ix, w in enumerate(weights) if ix in range(j + 1, new_foi + 1)]
        sc_minus_gamma: ServiceCurve = RateLatency(R=sc.rate - arrivalsAgg.r, t=1,
                                                   T=(arrivalsAgg.b + (sc.rate * sc.delay)) / (
                                                           sc.rate - arrivalsAgg.r))

        ret = RateLatency(R=(weights[new_foi] / sum(weights_k)) * sc_minus_gamma.rate, t=1,
                          T=sc_minus_gamma.delay)
        return ret

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
        assert (arrivals and len(arrivals) and weights and len(weights) and len(arrivals) == len(
                weights) and weights[
                    foi]), 'pre-conditions failed for GPS.LoSC_Chang_optimizeByDelayBound(...)'

        beta_i = None
        min_delay = None
        # in terms of delay
        best_m = None
        _iter = 0
        for M in subsetlist:
            if len(M) == 0:
                continue
            if _iter % 100000 == 0:
                logging.debug(f"M: {_iter} of {2**len(arrivals)}")
            beta_candidate = GPS.LoSC_Chang(arrivals, sc, weights, foi, M)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[foi],
                                                                       beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                beta_i = beta_candidate
                min_delay = delay_candidate
                best_m = M
            _iter += 1

        return beta_i, f'best_m (len)={len(best_m)}'

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
        # re-indexing
        arr_foi = arrivals.pop(foi)
        arrivals.append(arr_foi)
        weights_foi = weights.pop(foi)
        weights.append(weights_foi)
        new_foi = len(arrivals) - 1

        beta_i = None
        min_delay = None
        # in terms of delay
        best_j = None
        _iter = 0
        for j in range(new_foi):
            if _iter % 100000 == 0:
                logging.debug(f"M: {_iter} of {2**len(arrivals)}")
            beta_candidate = GPS.LoSC_Bouillard(arrivals, sc, weights, new_foi, j)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[new_foi],
                                                                       beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                beta_i = beta_candidate
                min_delay = delay_candidate
                best_j = j
            _iter += 1

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
        _iter = 0
        for M in subsetlist:
            if len(M) == 0:
                continue
            if _iter % 100000 == 0:
                logging.debug(f"M: {_iter} of {2**len(arrivals)}")
            beta_candidate = GPS.LoSC_BL_Consistent_Chang(arrivals, sc, weights, foi, M)
            delay_candidate = NC.delay_bound_token_bucket_rate_latency(arrivals[foi],
                                                                       beta_candidate)
            if min_delay is None or delay_candidate <= min_delay:
                beta_i = beta_candidate
                min_delay = delay_candidate
                best_m = M
            _iter += 1

        return beta_i, f'best_m (len)={len(best_m)}'
