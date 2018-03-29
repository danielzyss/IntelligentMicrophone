from tools import *
from parameters import *

class MembraneSystem():

    def __init__(self, Particles, Springs, mesh, nb_edge, dim):

        self.network_id = uuid.uuid4()
        self.net_dim = dim
        self.InitialState = self.ConstructStruct(Particles, Springs, mesh, nb_edge, self.network_id)

        self.Particles = Particles
        self.Springs = Springs
        self.Mesh = mesh
        self.nb_edge = nb_edge
        self.washout_time = 0.0

    def ConstructStruct(self, P, S, m, n, id):

        struct = {}

        particles = {'pos':[], 'mass':[], 'fixed':[], 'w_feed':[], 'w_in':[]}
        for p in P:
            particles['pos'].append(p.pos)
            particles['mass'].append(p.mass)
            particles['fixed'].append(p.fixed)
            particles['w_feed'].append(p.w_feed)
            particles['w_in'].append(p.w_in)

        struct['particles'] = particles

        springs = {'l0':[], 'p1':[], 'p2':[], 'k':[], 'd':[]}
        for s in S:
            springs['l0'].append(s.l0)
            springs['p1'].append(s.p1)
            springs['p2'].append(s.p2)
            springs['k'].append(s.k)
            springs['d'].append(s.d)
        struct['springs']=springs

        struct['mesh'] = m
        struct['nb_edge'] = n
        struct['id'] = id
        struct['net_dim'] = self.net_dim

        return struct

    def resetNetwork(self):
        for i in range(0, len(self.Particles)):
            self.Particles[i].pos = self.InitialState['particles']['pos'][i]
            self.Particles[i].vel = np.zeros(net_dim)
            self.Particles[i].acc = np.zeros(net_dim)
            self.Particles[i].force = np.zeros(net_dim)

    def SaveNetwork(self):

        self.InitialState['washout_time'] = self.washout_time

        pickle.dump(self.InitialState, open('Membranes/' + str(self.network_id)+ '.pkl', 'wb'))
        ref = open('Membranes/membranes_ref.txt', 'a')
        ref.write(str(self.network_id) + str(self.nb_edge).center(6) + str(len(self.Particles)).center(6) +
                  str(self.washout_time).center(6) + str(self.net_dim).center(6) + '\n')
        ref.close()

    def RunOpenLoop(self, M, ext_force, y, integrator='euler'):

        for i, t in enumerate(np.linspace(0, dt*y.shape[0], y.shape[0])):

            for j, s_j in enumerate(self.Springs):
                self.Particles = s_j.CalculateForce(self.Particles)
                M[j, i] = s_j.lt - s_j.l0 + np.random.normal(0.0, 0.001)


            for p in self.Particles:
                if self.net_dim ==2:
                    p.updateAcceleration(np.array([ext_force[i], 0.0]), np.array([y[i], y[i]]))
                if self.net_dim ==3:
                    p.updateAcceleration(np.array([0.0, 0.0, -ext_force[i]]), np.array([y[i], y[i], y[i]]))

                if integrator=='euler':
                    p.updateVelocity()
                    p.updatePosition()
                if integrator=='verlet':
                    p.leapfrog()

            if graph:
                if t % 0.25 < 0.001:
                    self.PlotCurrentState(t, i, M[:, :i + 1], ext_force[:i + 1])


        return M

    def RunClosedLoop(self, M, ext_force, w, integrator='euler'):

        for i, t in enumerate(np.linspace(0, dt * ext_force.shape[0], ext_force.shape[0])):

            for j, s_j in enumerate(self.Springs):
                s_j.CalculateForce(self.Particles)
                M[j, i] = s_j.lt - s_j.l0

            f = np.matmul(w, np.append(1, M[:,i]))

            for p in self.Particles:
                if self.net_dim ==2:
                    p.updateAcceleration(np.array([ext_force[i], 0.0]), np.array([f, f]))
                if self.net_dim ==3:
                    p.updateAcceleration(np.array([0.0, 0.0, -ext_force[i]]), np.array([f, f, f]))

                if integrator=='euler':
                    p.updateVelocity()
                    p.updatePosition()
                if integrator=='verlet':
                    p.leapfrog()

            if graph:
                if t % 0.25 < 0.001:
                    self.PlotCurrentState(t, i, M[:, :i + 1], ext_force[:i + 1])

        return M

    def RunLoopless(self, M, ext_force, integrator='euler'):

        for i, t in enumerate(tqdm.tqdm(np.linspace(0, M.shape[1]*dt,M.shape[1]), desc='Estimating Washout Time')):

            for j, s_j in enumerate(self.Springs):
                self.Particles = s_j.CalculateForce(self.Particles)
                M[j, i] = s_j.lt - s_j.l0

            for p in self.Particles:
                if self.net_dim ==2:
                    p.updateAcceleration(np.array([ext_force[i],0.0]), np.array([0.0, 0.0]))
                else:
                    p.updateAcceleration(np.array([0.0, 0.0, -ext_force[i]]), np.array([0.0, 0.0, 0.0]))

                if integrator == 'euler':
                    p.updateVelocity()
                    p.updatePosition()
                if integrator == 'verlet':
                    p.leapfrog()

            if graph:
                if t % 0.25 < 0.001:
                    self.PlotCurrentState(t, i, M[:, :i + 1], ext_force[:i + 1])

        return M

    def PlotCurrentState(self, t, it, X, wave):

        current_position = []
        for p in self.Particles:
            current_position.append(p.pos)
        current_position = np.array(current_position)

        ax1 = plt.subplot(1, 3, 1)

        if delaunay:
            ax1.triplot(current_position[:, 0], current_position[:, 1], self.Mesh.simplices.copy())
            for i in range(0, current_position.shape[0]):
                ax1.plot(current_position[i, 0], current_position[i, 1], 'o')
        else:
            for i, pt in enumerate(current_position):
                for j, nei in enumerate(self.Mesh[i]):
                    plt.plot([current_position[i, 0], current_position[j, 0]],
                             [current_position[i, 1], current_position[j, 1]], '-o')

        ax1.set_xlim([-x_axis_length / 2, x_axis_length + x_axis_length / 2])
        ax1.set_ylim([-y_axis_length / 2, y_axis_length + y_axis_length / 2])
        ax1.set_title('network')

        ax2 = plt.subplot(1, 3, 2)
        for m in X:
            ax2.plot(m[:it])
        ax2.set_xlim([-1, it + 2])
        ax2.set_title('spring length')

        ax3 = plt.subplot(1, 3, 3)
        ax3.plot(wave[:it])
        ax3.set_xlim([-1, it + 2])
        ax3.set_ylim([min(wave) - 1, max(wave) + 1])
        ax3.set_title('external force')

        plt.suptitle('time = ' + str(t))

        plt.pause(0.25)
        ax1.cla()
        ax2.cla()
        ax3.cla()
