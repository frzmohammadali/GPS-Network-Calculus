class ServiceCurve:
    def __init__(self, rate, delay=0.0):
        self.rate = rate
        self.delay = delay

    def __repr__(self):
        return (f'<ServiceCurve rate={round(self.rate,4)}\n'
                f'              delay={round(self.delay,4)} />')
    __str__ = __repr__

class RateLatency(ServiceCurve):
    def __init__(self, R, T, t=1):
        self.T = T
        self.t = t
        self.R = R
        super().__init__(rate=R, delay=T)

    def getCurve(self, _t):
        return self.R * max(0, _t - self.T)

    def __repr__(self):
        server_type = "RateLatency"
        if self.T == 0.0:
            server_type = "ConstantRate"
        return f'<{server_type} R={round(self.R,4)} T={round(self.T,4)} />'

    __str__ = __repr__

class WorkConservingLink(RateLatency):

    def __init__(self, c, t=1):
        self.c = c
        super().__init__(R=self.c, T=0.0, t=t)

    def __repr__(self):
        return f'<WorkConservingLink C={round(self.c,4)} />'

    __str__ = __repr__