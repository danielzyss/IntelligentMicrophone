from tools import *
from parameters import *

class Spring:

    def __init__(self, l0, p1, p2, k, d):
        self.l0 = l0
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.d = d

        if quadratic_spring:
            self.k1 = k[0]
            self.k2 = k[1]

        if quadratic_spring:
            self.d1 = d[0]
            self.d2 = d[1]

    def CalculateForce(self, particles):
        pos1 = particles[self.p1].pos
        pos2 = particles[self.p2].pos

        vel1 = particles[self.p1].vel
        vel2 = particles[self.p2].vel

        self.lt = euclidean(pos1, pos2)

        if quadratic_spring:
            F_spring = self.k1 * np.power((self.lt - self.l0), 3) + self.k2 * (self.lt - self.l0)
            F_damper = self.d1 * np.power((vel2 - vel1)*(pos2-pos1)/self.lt, 3) + self.d2*(vel2 - vel1)*(pos2-pos1)/self.lt

        else:
            F_spring = self.k * (self.lt - self.l0)
            F_damper = self.d*(vel2 - vel1)*(pos2-pos1)/self.lt

        particles[self.p1].spring_force -= -(F_spring + F_damper) * (pos2-pos1)/self.lt
        particles[self.p2].spring_force += -(F_spring + F_damper) * (pos2-pos1)/self.lt

        return particles