from tools import *
from parameters import *

class Particle:
    def __init__(self, id, position, mass, fixed, w_feed, w_in):

        self.id = id
        self.mass = mass

        self.pos = position
        self.init_pos = position
        self.spring_force = np.zeros(net_dim)
        self.vel = np.zeros(net_dim)
        self.acc = np.zeros(net_dim)
        self.vel_half = np.zeros(net_dim)
        self.prior_acc = np.zeros(net_dim)

        self.fixed = fixed
        if net_dim ==2:
            self.gravity = self.mass * np.array([0.0, -g])
        if net_dim ==3:
            self.gravity = self.mass * np.array([0.0, -g, 0.0])


        self.w_feed = w_feed
        self.w_in = w_in

    def updateAcceleration(self, ext_f, feed_f ):
        self.prior_acc = self.acc

        if not self.fixed:
            self.acc = (self.spring_force + ext_f*self.w_in + feed_f*self.w_feed + self.gravity)/self.mass
            self.spring_force = np.zeros(net_dim)

    def updateVelocity(self):
        self.vel = self.vel + self.acc*dt

    def updatePosition(self):
        self.prev_pos = self.pos
        self.pos = self.pos + self.vel*dt

    def leapfrog(self):
        self.vel_half = self.vel + self.prior_acc*dt/2
        self.pos = self.pos + self.vel_half*dt
        self.vel = self.vel_half + self.acc*dt/2

