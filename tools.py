from parameters import *
from Membrane import MembraneSystem
from Particle import Particle
from Spring import Spring

def CreateNetworkElements():

    if net_dim==2:
        x, y = np.linspace(0, x_axis_length, nb_row), np.linspace(0, y_axis_length, nb_col)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack((xx.flatten(), yy.flatten()), axis=-1)
        sigma_pos = ((x_axis_length + y_axis_length) / 2) / (2 * (nb_col + nb_row) / 2)
    elif net_dim==3:
        x, y, z = np.linspace(0, x_axis_length, nb_row), np.linspace(0, y_axis_length, nb_col), np.linspace(0, z_axis_length, nb_hei)
        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)
        sigma_pos = ((x_axis_length + y_axis_length + z_axis_length) / 3) / (2 * (nb_col + nb_row + nb_hei) / 3)
    else:
        raise NameError('Incorrect Network Initialization Dimensions (net_dim must be 2 or 3)')


    param_ranges = [[0.0, 0.1], [0.1, 1.0], [1.0, 10.0], [10.0, 100.0], [100.0, 1000.0], [1000.0, 10000.0]]
    param_rand_int = np.random.randint(0, len(param_ranges))
    sigma_gen = (param_ranges[param_rand_int][1] - param_ranges[param_rand_int][0]) / 4
    param_rand_int = np.random.randint(0, len(param_ranges))
    mass_gen = np.random.uniform(param_ranges[param_rand_int][0], param_ranges[param_rand_int][1])
    param_rand_int = np.random.randint(0, len(param_ranges))
    w_in_gen = np.random.uniform(param_ranges[param_rand_int][0], param_ranges[param_rand_int][1])
    param_rand_int = np.random.randint(0, len(param_ranges))
    w_feed_gen = np.random.uniform(param_ranges[param_rand_int][0], param_ranges[param_rand_int][1])
    param_rand_int = np.random.randint(0, len(param_ranges))

    feed_gen = np.random.uniform(2, grid_points.shape[0])
    feed_id = np.random.choice(np.arange(0, grid_points.shape[0]), round(feed_gen), replace=False).tolist()

    if random_input:
        in_gen = nb_col
        in_id = np.random.choice(np.arange(0, grid_points.shape[0]), round(in_gen), replace=False).tolist()

    Particles = []
    for i_p, p in enumerate(grid_points):
        fixed = IsFixed(p)
        if random_input:
            if i_p in in_id:
                if not fixed:
                    input = True
                else:
                    in_id[in_id.index(i_p)] = i_p + 1
                    input = False
            else:
                input = False
            if not fixed:
                p += np.abs(np.random.normal(0, sigma_pos, net_dim))
                if net_dim ==2:
                    p[0] = min(p[0], x_axis_length)
                    p[1] = min(p[1], y_axis_length)
                if net_dim==3:
                    p[0] = min(p[0], x_axis_length)
                    p[1] = min(p[1], y_axis_length)
                    p[2] = min(p[2], z_axis_length)
        else:
            input = IsInput(p, fixed)
            if not fixed and not input:
                p+= np.abs(np.random.normal(0, sigma_pos, net_dim))
                if net_dim ==2:
                    p[0] = min(p[0], x_axis_length)
                    p[1] = min(p[1], y_axis_length)
                if net_dim==3:
                    p[0] = min(p[0], x_axis_length)
                    p[1] = min(p[1], y_axis_length)
                    p[2] = min(p[2], z_axis_length)

        if i_p in feed_id:
            if not fixed and not input:
                w_feed = np.random.normal(0.0, 1.0) * w_feed_over_coef
            else:
                feed_id[feed_id.index(i_p)] = i_p+1
                w_feed = 0.0
        else:
            w_feed = 0.0

        if input:
            w_input = np.random.normal(0.0, 1.0) * w_input_over_coef
        else:
            w_input = 0.0

        # m = np.random.normal(mass_gen, sigma_gen)
        m = 1.0
        Particles.append(Particle(i_p, p, m, fixed, w_feed, w_input))
        grid_points[i_p] = p
    Particles = np.array(Particles)

    if delaunay:
        mesh = Delaunay(grid_points)
    else:
        mesh = RandomConnectionMesh(grid_points)

    edges, nb_edges = BuildEdges(mesh)

    if force_param:
        stiffness_gen = stiff_force
        damping_gen = damp_gen_force
        sigma_gen = sig_force
    else:
        param_rand_int = np.random.randint(0, len(param_ranges))
        stiffness_gen = np.random.uniform(param_ranges[param_rand_int][0], param_ranges[param_rand_int][1])
        param_rand_int = np.random.randint(0, len(param_ranges))
        damping_gen = np.random.uniform(param_ranges[param_rand_int][0], param_ranges[param_rand_int][1])

    Springs = []
    for j, e_j in enumerate(edges):
        l0 = euclidean(grid_points[e_j[0]], grid_points[e_j[1]])

        if quadratic_spring:
            k = np.random.normal(stiffness_gen, sigma_gen, 2)
            d = np.random.normal(damping_gen, sigma_gen, 2)
        else:
            k = np.random.normal(stiffness_gen, sigma_gen)
            d = np.random.normal(damping_gen, sigma_gen)

        new_spring = Spring(l0, e_j[0], e_j[1], k, d)
        Springs.append(new_spring)
    Springs = np.array(Springs)

    return Particles, Springs, mesh, nb_edges

def BuildEdges(tri):

    edges = []
    visitedEdges = []

    if delaunay:
        for s in tri.simplices:
            for e in list(itertools.combinations(s, 2)):
                if e not in visitedEdges:
                    edges.append(e)
                    visitedEdges.append((e[0], e[1]))
                    visitedEdges.append((e[1], e[0]))
    else:
        for i in range(0, tri.shape[0]):
            for j in tri[i]:
                if i != j and [i, j] not in visitedEdges and [j, i] not in visitedEdges:
                    edges.append(np.array([i, j]))
                    visitedEdges.append([i, j])

    edges = np.array(edges)
    nbOfEdges = edges.shape[0]

    return edges, nbOfEdges

def IsFixed(pt):

    if net_dim==2:
        if pt[1] == y_axis_length:
            return True
        elif pt[0] == x_axis_length:
            return True
        elif pt[1] == 0:
            return True
        else:
            return False
    if net_dim==3:
        if fixed_plate:
            if pt[1]==0:
                return True

        if pt[0]==0 and pt[1]==y_axis_length:
            return True
        elif pt[0]==0 and pt[2]==z_axis_length:
            return True
        elif pt[1]==y_axis_length and pt[2]==z_axis_length:
            return True
        elif pt[1]==y_axis_length and pt[2]==0:
            return True
        elif pt[0]==x_axis_length and pt[1]==y_axis_length:
            return True
        elif pt[0]==x_axis_length and pt[2]==0:
            return True
        elif pt[0]==x_axis_length and pt[1]==0:
            return True
        elif pt[1]==0 and pt[2]==0:
            return True
        elif pt[0]==0 and pt[2]==0:
            return True
        elif pt[0]==0 and pt[1]==0:
            return True
        elif pt[1]==0 and pt[2]==z_axis_length:
            return True
        elif pt[0]==x_axis_length and pt[2]==z_axis_length:
            return True
        else:
            return False

def IsInput(pt, fixed):

    if net_dim==2:
        if pt[0] == 0 and pt[1] != 0 and pt[1] != y_axis_length:
            return True
        else:
            return False
    if net_dim==3:
        if pt[2]==z_axis_length and fixed==False:
            return True
        else:
            return False

def RandomConnectionMesh(points):
    Neigh = NearestNeighbors()
    Neigh.fit(points)
    mesh = []
    for pt in points:
        r_norm = np.random.normal(0, x_axis_length / nb_row * 2)
        dist, neighbours_idx = Neigh.radius_neighbors(pt.reshape((1, net_dim)), np.abs(r_norm))
        mesh.append(neighbours_idx[0])
    return np.array(mesh)

def AssessNetwork(Membrane):
    step = np.hstack((np.zeros(20), np.ones(int(washout_max_permitted_time/dt - 20))))
    M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time/dt)))

    validated = False

    membrane_test_id = 1
    while not validated:
        print('Assessing Membrane #' + str(membrane_test_id))
        membrane_test_id+=1

        no_nan = False
        while not no_nan:
            if cplusplus:
                M = RunWithCplusplus(Membrane, 'loopless', M=M, force=step)
                if not np.isnan(M).any():
                    no_nan = True
                else:
                    Particles, Springs, mesh, nb_edges = CreateNetworkElements()
                    Membrane = MembraneSystem(Particles, Springs, mesh, nb_edges, net_dim)
                    M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time / dt)))
            else:
                try:
                    M = Membrane.RunLoopless(M, step)
                    no_nan = True
                except:
                    Particles, Springs, mesh, nb_edges = CreateNetworkElements()
                    Membrane = MembraneSystem(Particles,Springs, mesh, nb_edges, net_dim)
                    M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time / dt)))

        if net_dim ==2:
            grad_m = []
            for m in M:
                grad_m.append(np.abs(np.gradient(m, dt))/np.sqrt(x_axis_length**2 + y_axis_length**2))
        if net_dim==3:
            grad_m = []
            for m in M:
                grad_m.append(np.abs(np.gradient(m, dt)) / np.sqrt(x_axis_length ** 2 + y_axis_length ** 2 + z_axis_length**2))

        std_m = np.std(grad_m, axis=0)
        std_m_norm = (std_m - std_m.min())/(std_m.max()-std_m.min())
        washout_idx = np.where(std_m_norm > washout_criteria)[0]


        if washout_idx.shape[0]>0:
            washout_idx = max(washout_idx)
            if washout_idx<(washout_max_permitted_time/dt - 1):

                Membrane.washout_time = dt*washout_idx
                print('Membrane Validated: ' + str(Membrane.network_id) + ' with washout-time: ' + str(Membrane.washout_time))
                return Membrane
            else:
                Particles, Springs, mesh, nb_edges = CreateNetworkElements()
                Membrane = MembraneSystem(Particles, Springs, mesh, nb_edges, net_dim)
                M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time / dt)))
        else:
            Particles, Springs, mesh, nb_edges = CreateNetworkElements()
            Membrane = MembraneSystem(Particles, Springs, mesh, nb_edges, net_dim)
            M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time / dt)))

def LoadMembrane():
    InitialState = pickle.load(open('Membranes/' + membrane_id, 'rb'))

    particles = []
    springs = []

    for i_p in range(0, len(InitialState['particles']['pos'])):
        p = Particle(i_p,
                     InitialState['particles']['pos'][i_p],
                     InitialState['particles']['mass'][i_p],
                     InitialState['particles']['fixed'][i_p],
                     InitialState['particles']['w_feed'][i_p],
                     InitialState['particles']['w_in'][i_p])
        particles.append(p)

    for i_s in range(0, len(InitialState['springs']['l0'])):
        s = Spring(InitialState['springs']['l0'][i_s],
                   InitialState['springs']['p1'][i_s],
                   InitialState['springs']['p2'][i_s],
                   InitialState['springs']['k'][i_s],
                   InitialState['springs']['d'][i_s])
        springs.append(s)

    m = InitialState['mesh']
    n = InitialState['nb_edge']
    id = InitialState['id']
    w = InitialState['washout_time']
    dim = InitialState['net_dim']

    M = MembraneSystem(particles, springs, m, n, dim)
    M.network_id = id
    M.washout_time = w

    return M

def OrganizeData(X, y):
    X = X[:nb_classes]
    learning_feed = []
    for i, c in enumerate(X):
        learning_feed.append([])
        for x in c:
            learning_feed[i].append(np.full(x.shape[0],i , dtype=np.int))
            #(i + 1) * ((-1) ** (i + 1))
    learning_feed = np.array(learning_feed)

    Yset = learning_feed.reshape((learning_feed.shape[0]*learning_feed.shape[1]))
    Xset = X.reshape((X.shape[0]*X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(Xset, Yset, train_size=0.8)

    return X_train, X_test, y_train, y_test

def UploadDataset():
    X = np.load('Data/voweldataset/sounds.npy')
    y = np.load('Data/voweldataset/target.npy')
    return X,y

def PlotTraining(Membrane, M_train, w, Y_train, vowel_length):

    regressed = np.matmul(w, M_train)

    regressed_r = np.array([])
    space = 0
    for i in range(0, train_batch_size):
        mean = np.round(np.mean(regressed[space:space+vowel_length[i]]))
        regressed_r = np.append(regressed_r, np.full((vowel_length[i]), mean))
        space+=vowel_length[i]

    plt.plot(np.matmul(w, M_train), 'b', label='Linear Regression')
    plt.plot(Y_train, 'r', label='Expected Value')
    plt.plot(regressed_r, '--g', label='Running Average')
    plt.ylabel('Class')
    plt.xlabel('Time')
    plt.title('Training Regression vs. Expected Classes')
    plt.legend()
    ftitle = 'Training-' + str(nb_classes) + 'classes-' + str(train_batch_size) + 'size-' + str(Membrane.network_id) + '.png'
    # plt.savefig('figures/' + ftitle)
    plt.savefig('figures/ElitMembraneTraining.png')
    plt.close()

    return regressed, regressed_r

def PlotTesting(Membrane, M_test, w, Y_test, vowel_length):

    predicted = np.matmul(w, M_test)
    predicted_r = np.array([])
    space = 0
    for i in range(0, test_batch_size):
        mean = np.round(np.mean(predicted[space:space+vowel_length[i]]))
        predicted_r = np.append(predicted_r, np.full((vowel_length[i]), mean))
        space += vowel_length[i]

    plt.plot(predicted, 'b', label='Linear Regression')
    plt.plot(Y_test, 'r', label='Expected Value')
    plt.plot(predicted_r, '--g', label='Rounded Regression', )
    plt.ylabel('Class')
    plt.xlabel('Time')
    plt.title('Testing Regression vs. Expected Classes')
    plt.legend()
    ftitle = 'Testing-' + str(nb_classes) + 'classes-' + str(test_batch_size) + 'size-' + str(Membrane.network_id) + '.png'
    # plt.savefig('figures/' + ftitle)
    plt.savefig('figures/ElitMembraneTesting.png')
    plt.close()

    return predicted, predicted_r

def RunWithCplusplus(Membrane, action='openloop', M=np.array([]), force=np.array([]), y=np.array([]), w=np.array([])):

    command = "./fastmembrane "
    command += action + " "
    command += str(len(Membrane.Particles)) + " "
    command += str(len(Membrane.Springs)) + " "
    command += str(Membrane.washout_time) + " "
    command += "euler" + " "
    command += str(M.shape[0]) + " "
    command += str(M.shape[1]) + " "
    command += str(dt) + " "
    command += str(M.shape[1]*dt) + " "
    command += str(Membrane.net_dim) + " "

    if action=='openloop':
        np.savetxt("Cplusplus/openloop_force.txt", force, delimiter="\n", fmt='%1.6f')
        np.savetxt("Cplusplus/openloop_y.txt", y, delimiter="\n", fmt='%1.6f')
        command+= "openloop_force.txt" + " "
        command+= "openloop_y.txt" + " "
    elif action=='loopless':
        np.savetxt("Cplusplus/loopless_force.txt", force, delimiter="\n", fmt='%1.6f')
        command += "loopless_force.txt" + " "
    elif action=='closedloop':
        np.savetxt("Cplusplus/closedloop_force.txt", force, delimiter="\n", fmt='%1.6f')
        np.savetxt("Cplusplus/closedloop_w.txt", w, delimiter="\n", fmt='%1.6f')
        command += "closedloop_force.txt" + " "
        command += "closedloop_w.txt" + " "

    for p in Membrane.Particles:
        command+= str(p.pos[0]) + " "
    for p in Membrane.Particles:
        command+= str(p.pos[1]) + " "
    for p in Membrane.Particles:
        command+= str(p.mass) + " "
    for p in Membrane.Particles:
        command+= str(p.fixed) + " "
    for p in Membrane.Particles:
        command+= str(p.w_feed) + " "
    for p in Membrane.Particles:
        command+= str(p.w_in) + " "
    for s in Membrane.Springs:
        command+= str(s.l0) + " "
    for s in Membrane.Springs:
        command+= str(s.p1) + " "
    for s in Membrane.Springs:
        command+= str(s.p2) + " "
    for s in Membrane.Springs:
        if quadratic_spring:
            command+= str(s.k1) + " "
        else:
            command += str(0.0) + " "
    for s in Membrane.Springs:
        if quadratic_spring:
            command+= str(s.k2) + " "
        else:
            command+= str(s.k) + " "
    for s in Membrane.Springs:
        if quadratic_spring:
            command+= str(s.d1) + " "
        else:
            command+= str(0.0) + " "
    for s in Membrane.Springs:
        if quadratic_spring:
            command+= str(s.d2) + " "
        else:
            command+= str(s.d) + " "
    if Membrane.net_dim==3:
        for p in Membrane.Particles:
            command+= str(p.pos[2]) + " "


    owd = os.getcwd()
    os.chdir('Cplusplus')
    out = subprocess.check_output(command.split())
    out = out[:len(out)-1]
    M = np.array([float(i) for i in out.decode("utf-8").split("*")]).reshape(M.shape)
    os.chdir(owd)


    if action=='openloop':
        os.remove("Cplusplus/openloop_force.txt")
        os.remove("Cplusplus/openloop_y.txt")
    elif action=='loopless':
        os.remove("Cplusplus/loopless_force.txt")
    elif action=='closedloop':
        os.remove("Cplusplus/closedloop_force.txt")
        os.remove("Cplusplus/closedloop_w.txt")


    return M

def PlotInitialNetwork(Membrane):
    if net_dim==3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, s in enumerate(Membrane.Springs):
            ax.plot([Membrane.Particles[s.p1].pos[0], Membrane.Particles[s.p2].pos[0]],
                     [Membrane.Particles[s.p1].pos[1], Membrane.Particles[s.p2].pos[1]],
                    [Membrane.Particles[s.p1].pos[2], Membrane.Particles[s.p2].pos[2]], 'grey', zorder=1)
            if i == 0:
                ax.plot([Membrane.Particles[s.p1].pos[0], Membrane.Particles[s.p2].pos[0]],
                         [Membrane.Particles[s.p1].pos[1], Membrane.Particles[s.p2].pos[1]],
                        [Membrane.Particles[s.p1].pos[2], Membrane.Particles[s.p2].pos[2]],'grey', label='spring', zorder=1)
        i1 = 0
        i2 = 0
        i3 = 0
        i4 = 0
        for p in Membrane.Particles:
            if p.fixed:
                ax.scatter(p.pos[0], p.pos[1],p.pos[2], c='r', alpha=1.0, zorder=2)
                if i1 ==0:
                    ax.scatter(p.pos[0], p.pos[1],p.pos[2], c='r', alpha=1.0, label='fixed', zorder=2)
                    i1 =1
            elif p.w_in!=0.0:
                ax.scatter(p.pos[0], p.pos[1],p.pos[2], c='g', zorder=2)
                if i2 ==0:
                    ax.scatter(p.pos[0], p.pos[1],p.pos[2], c='g', alpha=1.0, label='input', zorder=2)
                    i2 =1
            elif p.w_feed!=0.0:
                ax.scatter(p.pos[0], p.pos[1],p.pos[2], c='purple', alpha=1.0, zorder=2)
                if i3 ==0:
                    plt.scatter(p.pos[0], p.pos[1],p.pos[2], c='purple', alpha=1.0, label='feedback', zorder=2)
                    i3 =1
            else:
                ax.scatter(p.pos[0], p.pos[1],p.pos[2], c='b', zorder=2)
                if i4 ==0:
                    ax.scatter(p.pos[0], p.pos[1],p.pos[2], c='b', alpha=1.0, label='regular', zorder=2)
                    i4 =1


        plt.legend(loc='best')
        plt.xlim([-2, 70])
        plt.ylim([-2, 70])
        plt.axis('off')
        plt.savefig('figures/morphology3d.png', transparent=True)
    else:

        for i, s in enumerate(Membrane.Springs):
            plt.plot([Membrane.Particles[s.p1].pos[0], Membrane.Particles[s.p2].pos[0]],
                     [Membrane.Particles[s.p1].pos[1], Membrane.Particles[s.p2].pos[1]], 'grey', zorder=1)
            if i == 0:
                plt.plot([Membrane.Particles[s.p1].pos[0], Membrane.Particles[s.p2].pos[0]],
                         [Membrane.Particles[s.p1].pos[1], Membrane.Particles[s.p2].pos[1]], 'grey', label='spring', zorder=1)
        i1 = 0
        i2 = 0
        i3 = 0
        i4 = 0
        for p in Membrane.Particles:
            if p.fixed:
                plt.scatter(p.pos[0], p.pos[1], c='r', alpha=1.0, zorder=2)
                if i1 ==0:
                    plt.scatter(p.pos[0], p.pos[1], c='r', alpha=1.0, label='fixed', zorder=2)
                    i1 =1
            elif p.w_in!=0.0:
                plt.scatter(p.pos[0], p.pos[1], c='g', zorder=2)
                if i2 ==0:
                    plt.scatter(p.pos[0], p.pos[1], c='g', alpha=1.0, label='input', zorder=2)
                    i2 =1
            elif p.w_feed!=0.0:
                plt.scatter(p.pos[0], p.pos[1], c='purple', alpha=1.0, zorder=2)
                if i3 ==0:
                    plt.scatter(p.pos[0], p.pos[1], c='purple', alpha=1.0, label='feedback', zorder=2)
                    i3 =1
            else:
                plt.scatter(p.pos[0], p.pos[1], c='b', zorder=2)
                if i4 ==0:
                    plt.scatter(p.pos[0], p.pos[1], c='b', alpha=1.0, label='regular', zorder=2)
                    i4 =1


        plt.legend(loc='best')
        plt.xlim([-2, 80])
        plt.ylim([-2, 80])
        plt.axis('off')
        plt.savefig('figures/morphology2d.png')

def PlotMembranesCharactristics():
    elitDNA = ['0001101101101001111001000111110101100', '0001000001101011111010100111110101100',
               '0001100101111011111001000111110101100', '0001000001101011111000100111110101100',
               '0001000001101011111000000111110101100', '1101100101111100111011001001111000010']
    elitDNAcol = ['r', 'g', 'b', 'k', 'y', 'purple']
    Membranes = []
    from GeneticGenerator import ConstructMembraneFromDNA
    for m in elitDNA:
        M, _ = ConstructMembraneFromDNA(m)
        Membranes.append(M)

    plots = []
    for i, M in enumerate(Membranes):
        plt.subplot(2, 2, 1)
        l = plt.scatter([i.w_in for i in M.Particles], [i.w_feed for i in M.Particles], c=elitDNAcol[i],
                        label='$e_' + str(i) + '$')
        plots.append(l)
        plt.xlabel('$w_{in}$')
        plt.ylabel('$w_{feed}$')
        plt.title('particle coefficients')

        plt.subplot(2, 2, 2)
        plt.scatter([s.k1 for s in M.Springs], [s.k2 for s in M.Springs], c=elitDNAcol[i])
        plt.xlabel('$k_1$')
        plt.ylabel('$k_2$')
        plt.title('springs stiffness')

        plt.subplot(2, 2, 3)
        plt.scatter([s.d1 for s in M.Springs], [s.d2 for s in M.Springs], c=elitDNAcol[i])
        plt.xlabel('$d_1$')
        plt.ylabel('$d_2$')
        plt.title('springs damper')

    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.legend(handles=plots, loc='center')
    plt.tight_layout()
    # plt.savefig('figures/optimizedcharacteristics.png')
    plt.show()

def PlotFourierTransform(xb, Mb, Membrane, w):
    xb_fft = np.fft.fft(xb)
    xb_fr = np.fft.fftfreq(xb.shape[-1])

    plt.subplot(1,3,1)
    plt.plot(xb_fr, abs(xb_fft) ** 2)
    plt.title('input signal in frequency space', fontsize=10)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('squared modulus of Fourrier Transform')

    for i, m in enumerate(Mb):
        m_fft = np.fft.fft(m)
        m_fr = np.fft.fftfreq(m.shape[-1])
        plt.subplot(1, 3, 2)
        plt.plot(m_fr, abs(m_fft)**2)
    plt.title('spring length in frequency space', fontsize=10)
    plt.xlabel('frequency (Hz)')
    # plt.ylabel('squared modulus of Fourrier Transform')

    Mb = sm.add_constant(Mb.T, has_constant='add').T
    regressed = np.matmul(w, Mb)
    r_fft = np.fft.fft(regressed)
    r_fr = np.fft.fftfreq(regressed.shape[-1])
    plt.subplot(1,3,3)
    plt.plot(r_fr, abs(r_fft)**2)
    plt.title('regressed signal in frequency space', fontsize=10)
    plt.xlabel('frequency (Hz)',)
    # plt.ylabel('squared modulus of Fourrier Transform')

    plt.tight_layout()
    plt.savefig('figures/ElitMembraneFourrier.png')
    plt.close()

def PlotFourierCoherence(xb, Mb, Membrane, w):
    M_test = sm.add_constant(Mb.T, has_constant='add').T
    regressed = np.matmul(w, M_test)

    f, Cxy = signal.coherence(xb, regressed)
    plt.semilogy(f, Cxy)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('coherence')
    plt.title('coherence in frequency space of input and regressed output')
    plt.savefig('figures/coherenceinputoutput.png')

    plt.close()
    for i, m in enumerate(Mb):
        if not Membrane.Particles[Membrane.Springs[i].p1].fixed or not Membrane.Particles[Membrane.Springs[i].p2].fixed:
            if Membrane.Particles[Membrane.Springs[i].p1].w_in ==0.0 and Membrane.Particles[Membrane.Springs[i].p2].w_in==0.0:
                if Membrane.Particles[Membrane.Springs[i].p1].w_feed == 0.0 or Membrane.Particles[
                    Membrane.Springs[i].p2].w_feed == 0.0:
                    f, Cxy = signal.coherence(xb, m)
                    plt.semilogy(f, Cxy)

    plt.xlabel('frequency (Hz)')
    plt.ylabel('coherence')
    plt.title('coherence in frequency space of input and regular springs length')
    plt.savefig('figures/coherenceinputspringlength.png')
    plt.close()

def PlotSringLength(Mb, Membrane, y_test, b):
    check1=0
    check2=0

    ##########


    if y_test[b][0] == 0 and check1 == 0:
        for i, m in enumerate(Mb):
            if not Membrane.Particles[Membrane.Springs[i].p1].fixed and not Membrane.Particles[
                Membrane.Springs[i].p2].fixed:
                if Membrane.Particles[Membrane.Springs[i].p1].w_in == 0.0 and Membrane.Particles[
                    Membrane.Springs[i].p2].w_in == 0.0:
                    if Membrane.Particles[Membrane.Springs[i].p1].w_feed == 0.0 or Membrane.Particles[
                        Membrane.Springs[i].p2].w_feed == 0.0:
                        x = np.arange(0, m.shape[0])
                        if check1 == 0:
                            plt.plot(x[:int(Membrane.washout_time / dt)], m[:int(Membrane.washout_time / dt)], 'r',
                                     label='washout')
                            plt.plot(x[int(Membrane.washout_time / dt):], m[int(Membrane.washout_time / dt):])
                            check1 = 1
                        else:
                            plt.plot(x[:int(Membrane.washout_time / dt)], m[:int(Membrane.washout_time / dt)], 'r')
                            plt.plot(x[int(Membrane.washout_time / dt):], m[int(Membrane.washout_time / dt):], )
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('spring length')
        plt.title('spring length as a function of time for class ' + str(y_test[b][0]))
        plt.savefig('figures/springsteadystate' + str(y_test[b][0]) + '.png')
        plt.close()

    if y_test[b][0] == 1 and check2 == 0:
        for i, m in enumerate(Mb):
            if not Membrane.Particles[Membrane.Springs[i].p1].fixed and not Membrane.Particles[
                Membrane.Springs[i].p2].fixed:
                if Membrane.Particles[Membrane.Springs[i].p1].w_in == 0.0 and Membrane.Particles[
                    Membrane.Springs[i].p2].w_in == 0.0:
                    if Membrane.Particles[Membrane.Springs[i].p1].w_feed == 0.0 or Membrane.Particles[
                        Membrane.Springs[i].p2].w_feed == 0.0:
                        x = np.arange(0, m.shape[0])
                        if check2 == 0:
                            plt.plot(x[:int(Membrane.washout_time / dt)], m[:int(Membrane.washout_time / dt)], 'r',
                                     label='washout')
                            plt.plot(x[int(Membrane.washout_time / dt):], m[int(Membrane.washout_time / dt):])
                            check2 = 1
                        else:
                            plt.plot(x[:int(Membrane.washout_time / dt)], m[:int(Membrane.washout_time / dt)], 'r')
                            plt.plot(x[int(Membrane.washout_time / dt):], m[int(Membrane.washout_time / dt):], )
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('spring length')
        plt.title('spring length as a function of time for class ' + str(y_test[b][0]))
        plt.savefig('figures/springsteadystate' + str(y_test[b][0]) + '.png')
        plt.close()

def PlotPhasePortrait(Mb, Membrane, y_test, b):
    check1=0
    check2=0

    #########

    reg_idx = []
    for i, m in enumerate(Mb[:, int(Membrane.washout_time / dt):]):
        if not Membrane.Particles[Membrane.Springs[i].p1].fixed and not Membrane.Particles[
            Membrane.Springs[i].p2].fixed:
            if Membrane.Particles[Membrane.Springs[i].p1].w_in == 0.0 and Membrane.Particles[
                Membrane.Springs[i].p2].w_in == 0.0:
                if Membrane.Particles[Membrane.Springs[i].p1].w_feed == 0.0 or Membrane.Particles[
                    Membrane.Springs[i].p2].w_feed == 0.0:
                    reg_idx.append(i)

    if y_test[b][0] == 0 and check1 == 0:
        for i, m in enumerate(Mb[reg_idx, int(Membrane.washout_time / dt):]):
            plt.subplot(3, 3, i + 1)
            if i == 0:
                l1 = plt.plot(np.gradient(m, dt), np.gradient(np.gradient(m, dt), dt), 'b', zorder=1)
            else:
                plt.plot(np.gradient(m, dt), np.gradient(np.gradient(m, dt), dt), 'b', zorder=1)
            plt.xlabel(r'$\nabla l_t$')
            if i % 3 == 0:
                plt.ylabel(r'$\nabla^2 l_t$')
        check1 = 1
    if y_test[b][0] == 1 and check2 == 0:
        for i, m in enumerate(Mb[reg_idx, int(Membrane.washout_time / dt):]):
            plt.subplot(3, 3, i + 1)
            if i == 0:
                l2 = plt.plot(np.gradient(m, dt), np.gradient(np.gradient(m, dt), dt), 'r', zorder=2)
            else:
                plt.plot(np.gradient(m, dt), np.gradient(np.gradient(m, dt), dt), 'r', zorder=2)
            plt.xlabel(r'$\nabla l_t$')
            if i % 3 == 0:
                plt.ylabel(r'$\nabla^2 l_t$')
        check2 = 1

    if check1 + check2 == 2:
        plt.suptitle('phase portrait')
        plt.figlegend((l1[0], l2[0]), ('class 0', 'class 1'), 'lower center')
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('figures/ElitPhasePortrait.png')
        sys.exit()