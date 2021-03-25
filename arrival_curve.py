class ArrivalCurve:
    @staticmethod
    def aggregateTokenBuckets(*alphas):
        assert len([a for a in alphas if isinstance(a, TokenBucket)]) == len(
                alphas), "arrival curves are " \
                         "not token " \
                         "bucket"
        if len(alphas) == 0:
            return TokenBucket(b=0, r=0, t=1)
        sum_b = sum([alpha.b for alpha in alphas])
        sum_r = sum([alpha.r for alpha in alphas])
        return TokenBucket(b=sum_b, r=sum_r, t=alphas[0].t)


class TokenBucket(ArrivalCurve):
    def __init__(self, b=1, r=1, t=1):
        self.t = t
        self.r = r
        self.b = b

    def getCurve(self, _t):
        return (self.b + self.r * _t) if _t != 0 else 0

    def __repr__(self):
        return f'<TB r={round(self.r,2)} b={round(self.b,2)}>'

    __str__ = __repr__
