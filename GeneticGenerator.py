from parameters import *
from tools import RunWithCplusplus
from Particle import Particle
from Spring import Spring
from Membrane import MembraneSystem

def CreateNetworkElementsGEN(genparam):

    if dna_net_dim ==2:
        x, y = np.linspace(0, genparam['xlength'], genparam['row']), np.linspace(0, genparam['ylength'], genparam['col'])
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack((xx.flatten(), yy.flatten()), axis=-1)
        sigma_pos = ((genparam['xlength'] + genparam['ylength']) / 2) / (2 * (genparam['col'] +genparam['row']) / 2)
    elif dna_net_dim==3:
        x, y, z = np.linspace(0, genparam['xlength'], genparam['row']), np.linspace(0, genparam['ylength'],genparam['col']),\
                    np.linspace(0, genparam['xlength'], genparam['row'])
        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)
        sigma_pos = ((genparam['xlength'] + genparam['ylength']) / 2) / (2 * (genparam['col'] + genparam['row']) / 2)
    else:
        raise NameError('Incorrect Network Initialization Dimensions (net_dim must be 2 or 3)')


    feed_gen = np.random.uniform(2, grid_points.shape[0])
    feed_id = np.random.choice(np.arange(0, grid_points.shape[0]), round(feed_gen), replace=False).tolist()

    if random_input:
        in_gen = nb_col
        in_id = np.random.choice(np.arange(0, grid_points.shape[0]), round(in_gen), replace=False).tolist()

    Particles = []
    for i_p, p in enumerate(grid_points):
        fixed = IsFixedGEN(p, genparam)
        input = IsInputGEN(p, fixed, genparam)

        if not fixed and not input:
            p= np.abs(p+ np.random.normal(0, sigma_pos, dna_net_dim))

        if i_p in feed_id:
            if not fixed and not input:
                # w_feed = np.random.normal(genparam['feedgen'], genparam['sig']) * genparam['overfeed']
                w_feed = np.random.normal(0.0, 1.0) * genparam['overfeed']
            else:
                feed_id[feed_id.index(i_p)] = i_p+1
                w_feed = 0.0
        else:
            w_feed = 0.0

        if random_input:
            if i_p in in_id:
                if not fixed:
                    w_input = np.random.normal(0.0, 1.0) * genparam['overin']
                else:
                    in_id[in_id.index(i_p)] = i_p + 1
                    w_input = 0.0
            else:
                w_input = 0.0
        else:
            if input:
                    # w_input = np.random.normal(genparam['ingen'], genparam['sig']) * genparam['overin']
                    w_input = np.random.normal(0.0, 1.0) * genparam['overin']
            else:
                w_input = 0.0

        # m = np.random.normal(genparam['massgen'], genparam['sig'])
        m = 1.0
        Particles.append(Particle(i_p, p, m, fixed, w_feed, w_input))
        grid_points[i_p] = p
    Particles = np.array(Particles)

    if genparam['del']:
        mesh = Delaunay(grid_points)
    else:
        mesh = RandomConnectionMeshGEN(grid_points, genparam)

    edges, nb_edges = BuildEdgesGEN(mesh, genparam['del'])

    Springs = []

    for j, e_j in enumerate(edges):
        l0 = euclidean(grid_points[e_j[0]], grid_points[e_j[1]])

        if quadratic_spring:
            k = np.abs(np.random.normal(genparam['stiffgen'], genparam['sig'], 2))
            d = np.abs(np.random.normal(genparam['dampgen'], genparam['sig'], 2))
        else:
            k = np.abs(np.random.normal(genparam['stiffgen'], genparam['sig']))

            d = np.abs(np.random.normal(genparam['dampgen'], genparam['sig']))

        new_spring = Spring(l0, e_j[0], e_j[1], k, d)
        Springs.append(new_spring)
    Springs = np.array(Springs)

    return Particles, Springs, mesh, nb_edges

def BuildEdgesGEN(tri, delo):

    edges = []
    visitedEdges = []

    if delo:
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

def IsFixedGEN(pt, gennparam):

    if dna_net_dim==2:
        if pt[1] == gennparam['ylength']:
            return True
        elif pt[0] == gennparam['xlength']:
            return True
        elif pt[1] == 0:
            return True
        else:
            return False
    if dna_net_dim:
        if fixed_plate:
            if pt[1] == 0:
                return True

        if pt[0] == 0 and pt[1] == gennparam['ylength']:
            return True
        elif pt[0] == 0 and pt[2] == gennparam['xlength']:
            return True
        elif pt[1] == gennparam['ylength'] and pt[2] == gennparam['xlength']:
            return True
        elif pt[1] == gennparam['ylength'] and pt[2] == 0:
            return True
        elif pt[0] == gennparam['xlength'] and pt[1] == gennparam['ylength']:
            return True
        elif pt[0] == gennparam['xlength'] and pt[2] == 0:
            return True
        elif pt[0] == gennparam['xlength'] and pt[1] == 0:
            return True
        elif pt[1] == 0 and pt[2] == 0:
            return True
        elif pt[0] == 0 and pt[2] == 0:
            return True
        elif pt[0] == 0 and pt[1] == 0:
            return True
        elif pt[1] == 0 and pt[2] == gennparam['xlength']:
            return True
        elif pt[0] == gennparam['xlength'] and pt[2] == gennparam['xlength']:
            return True
        else:
            return False

def IsInputGEN(pt, fixed, genparam):

    if dna_net_dim==2:
        if pt[0] == 0 and pt[1] != 0 and pt[1] != genparam['ylength']:
            return True
        else:
            return False
    if dna_net_dim==3:
        if pt[2]==genparam['xlength'] and fixed==False:
            return True
        else:
            return False

def RandomConnectionMeshGEN(points, genparam):

    Neigh = NearestNeighbors()
    Neigh.fit(points)
    mesh = []
    for pt in points:
        r_norm = np.random.normal(0, ((genparam['xlength']+genparam['ylength'])/2)/((genparam['row']+genparam['col'])/2))
        dist, neighbours_idx = Neigh.radius_neighbors(pt.reshape((1, dna_net_dim)), np.abs(r_norm))
        mesh.append(neighbours_idx[0])

    return np.array(mesh)

def AssessNetworkGEN(Membrane, dna, genparam):
    n_it =0
    step = np.hstack((np.zeros(20), np.ones(int(washout_max_permitted_time/dt) - 20)))/genparam['overin']
    M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time/dt)))

    validated = False

    while not validated:

        no_nan = False
        while not no_nan:

            if cplusplus:
                M = RunWithCplusplus(Membrane, 'loopless', M=M, force=step)
                if not np.isnan(M).any():
                    no_nan = True
                else:
                    n_it+=1
                    Membrane, genparam= ConstructMembraneFromDNA(dna)
                    M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time / dt)))
                    step = np.hstack((np.zeros(20), np.ones(washout_max_permitted_time / dt - 20))) / genparam[
                        'overin']

            else:
                try:
                    M = Membrane.RunLoopless(M, step)
                    no_nan = True
                except:
                    n_it += 1
                    Membrane, genparam = ConstructMembraneFromDNA(dna)
                    M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time / dt)))
                    step = np.hstack((np.zeros(20), np.ones(washout_max_permitted_time / dt - 20))) / genparam[
                        'overin']

        grad_m = []
        for m in M:
            delta = np.abs(np.gradient(m, dt))
            grad_m.append(delta/np.sqrt(genparam['xlength']**2+ genparam['ylength']**2))

        std_m = np.std(grad_m, axis=0)
        std_m_norm = (std_m-std_m.min())/(std_m.max()-std_m.min())
        washout_idx = np.where(std_m_norm > washout_criteria)[0]

        if washout_idx.shape[0]>0 and np.max(grad_m)<1:
            washout_idx = max(washout_idx)
            if washout_idx<(washout_max_permitted_time/dt - 1):
                Membrane.washout_time = dt*washout_idx
                return Membrane, dna, True
            else:
                n_it += 1
                Membrane, genparam = ConstructMembraneFromDNA(dna)
                M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time / dt)))
                step = np.hstack((np.zeros(20), np.ones(washout_max_permitted_time / dt - 20))) / genparam['overin']
        else:
            n_it += 1
            Membrane, genparam= ConstructMembraneFromDNA(dna)
            M = np.zeros((Membrane.nb_edge, int(washout_max_permitted_time / dt)))
            step = np.hstack((np.zeros(20), np.ones(washout_max_permitted_time / dt - 20))) / genparam['overin']

def ConstructMembraneFromDNA(dna):
    gene_cuts = [[0, 4], [4, 8], [8, 10], [10, 12],
                 16, [13, 16], [16, 19], [19, 22],
                 [22, 25], [25, 28], [28, 31], [31, 34],
                 [34, 37]]

    gene_names = ['row', 'col', 'xlength', 'ylength', 'del', 'overfeed',
                  'overin', 'massgen', 'feedgen', 'ingen', 'stiffgen',
                  'dampgen', 'sig']

    max_row = 11
    max_col = 11
    min_row = 3
    min_col = 3
    maxminval = [[max_row, min_row], [max_col, min_col], 1, 1]

    generation_param ={}
    for i, gene in enumerate(gene_names):
        if i<2:
            generation_param[gene] = min(maxminval[i][0],
                                         int(dna[gene_cuts[i][0]:gene_cuts[i][1]], 2)+ maxminval[i][1])
        elif i>=2 and i<4:
            generation_param[gene] = max(maxminval[i],
                                         10**int(dna[gene_cuts[i][0]:gene_cuts[i][1]], 2))
        elif i==4:
            generation_param[gene] = bool(int(dna[gene_cuts[i]]))
        elif i>4:
            generation_param[gene] = 10**([-1,1][int(dna[gene_cuts[i][0]])]*int(dna[gene_cuts[i][0]+1:gene_cuts[i][1]],2))

    Particles, Springs, mesh, nb_edges = CreateNetworkElementsGEN(generation_param)
    Membrane = MembraneSystem(Particles, Springs, mesh, nb_edges, dna_net_dim)
    Membrane, dna, success = AssessNetworkGEN(Membrane,dna, generation_param)
    Membrane.SaveNetwork()

    return Membrane, generation_param

