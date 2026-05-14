class PIController:
    def __init__(self, Kp, Ki, limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.limit = limit
        self.integrator = 0.0

    def update(self, error, dt):
        self.integrator += error * dt
        u = self.Kp * error + self.Ki * self.integrator

        if self.limit is not None:
            if u > self.limit:
                u = self.limit
            elif u < -self.limit:
                u = -self.limit

        return u
