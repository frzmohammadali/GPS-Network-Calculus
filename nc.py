from src.arrival_curve import TokenBucket
from src.service_curve import RateLatency

class NC:
    @staticmethod
    def deconvolution(f,g):
        pass

    @staticmethod
    def delay_bound_token_bucket_rate_latency(alpha: TokenBucket, beta: RateLatency):
        assert hasattr(beta, 'R') and hasattr(beta, 'T'), "service must be RateLatency"
        assert hasattr(alpha, 'b'), "arrival must be TokenBucket"
        return round((alpha.b / beta.R) + beta.T, 4)
        # return (alpha.b / beta.R) + beta.T