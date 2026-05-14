import numpy as np

class PMSM:
    def __init__(self, params):
        self.Rs = params["Rs"]
        self.Ld = params["Ld"]
        self.Lq = params["Lq"]
        self.lambda_m = params["lambda_m"]
        self.p = params["p"]
        self.J = params["J"]
        self.B = params["B"]

        # Nominal parameters (for fault reset)
        self.Rs_nominal = self.Rs
        self.Ld_nominal = self.Ld
        self.Lq_nominal = self.Lq

        # States
        self.id = 0.0
        self.iq = 0.0
        self.omega = 0.0

    def step(self, vd, vq, Tl, dt):
        omega_e = self.p * self.omega

        # Electrical dynamics
        did_dt = (vd - self.Rs * self.id + omega_e * self.Lq * self.iq) / self.Ld
        diq_dt = (vq - self.Rs * self.iq - omega_e * self.Ld * self.id - omega_e * self.lambda_m) / self.Lq

        # Torque
        Te = 1.5 * self.p * (self.lambda_m * self.iq + (self.Ld - self.Lq) * self.id * self.iq)

        # Mechanical dynamics
        domega_dt = (Te - Tl - self.B * self.omega) / self.J

        # Euler integration
        self.id += did_dt * dt
        self.iq += diq_dt * dt
        self.omega += domega_dt * dt

        return self.id, self.iq, self.omega, Te
