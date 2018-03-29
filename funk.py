from tools import *
from parameters import *

def GenerateMembrane():

    Particles, Springs, mesh, nb_edges = CreateNetworkElements()
    Membrane = AssessNetwork(MembraneSystem(Particles,Springs, mesh, nb_edges, net_dim))
    Membrane.SaveNetwork()

    return Membrane

def Learning(Membrane,X_train,y_train):

    vowel_length = []

    M_train = []
    Y_train = []
    for b in tqdm.tqdm(range(0, train_batch_size), desc='Learning'):
        Membrane.resetNetwork()
        xb = X_train[b]
        vowel_length.append(xb.shape[0] - int(Membrane.washout_time/dt))
        yb = y_train[b]
        Mb = np.zeros((Membrane.nb_edge, yb.shape[0]))

        if cplusplus:
            Mb = RunWithCplusplus(Membrane, 'openloop', M=Mb, force=xb, y=yb)
        else:
            Mb = Membrane.RunOpenLoop(Mb, xb, yb)

        if gradientMatrix:
            grad_m = []
            for m in Mb:
                grad_m.append(np.gradient(m, dt))
            grad_m = np.array(grad_m)
            M_train.append(grad_m[:, int(Membrane.washout_time/dt):])
        else:
            Mb += np.random.normal(0, 0.0001, Mb.shape)
            M_train.append(Mb[:, int(Membrane.washout_time/dt):])

        Y_train.append(y_train[b][int(Membrane.washout_time/dt):])

    M_train = np.hstack(M_train)
    Y_train = np.hstack(Y_train)

    M_train = sm.add_constant(M_train.T , has_constant='add').T

    model = sm.OLS(Y_train, M_train.T)
    results = model.fit()
    w = results.params

    regressed, regressed_r = PlotTraining(Membrane, M_train, w, Y_train, vowel_length)

    training_accuracy = (1 - (np.count_nonzero(Y_train - regressed_r)/Y_train.shape[0]))*100
    print('training accuracy: ' + str(training_accuracy) + '%')

    return w

def Testing(Membrane,w,  X_test, y_test):

    vowel_length =[]

    M_test = []
    Y_test = []
    for b in tqdm.tqdm(range(0, test_batch_size), desc='testing'):
        Membrane.resetNetwork()
        xb = X_test[b]
        vowel_length.append(xb.shape[0] - int(Membrane.washout_time/dt))
        Mb = np.zeros((Membrane.nb_edge, xb.shape[0]))
        if cplusplus:
            Mb = RunWithCplusplus(Membrane, 'closedloop', M=Mb, force=xb, w=w)
        else:
            Mb = Membrane.RunClosedLoop(Mb, xb, w)

        if gradientMatrix:
            grad_m = []
            for m in Mb:
                grad_m.append(np.gradient(m, dt))
            grad_m = np.array(grad_m)
            M_test.append(grad_m[:, int(Membrane.washout_time/dt):])
        else:
            M_test.append(Mb[:, int(Membrane.washout_time/dt):])

        Y_test.append(y_test[b][int(Membrane.washout_time / dt):])

    M_test = np.hstack(M_test)
    Y_test = np.hstack(Y_test)

    M_test = sm.add_constant(M_test.T, has_constant='add').T

    predicted, predicted_r = PlotTesting(Membrane, M_test, w, Y_test, vowel_length)
    testing_accuracy = (1 - (np.count_nonzero(Y_test - predicted_r) / Y_test.shape[0])) * 100
    print('testing accuracy: ' + str(testing_accuracy) + '%')

def LearningMC(Membrane, X_train, y_train):

    vowel_length = []

    M_train = []
    Y_train = []
    wMC = []
    for c in range(0,nb_classes):
        wMC.append([])
        Y_train.append([])
        M_train.append([])

    ground_truth = []
    for b in tqdm.tqdm(range(0, train_batch_size), desc='Learning'):
        xb = X_train[b]
        vowel_length.append(xb.shape[0])
        ground_truth.append(y_train[b])
        for i in range(0, 2):
            if i==0:
                yb = np.full(y_train[b].shape, 1, dtype=np.int)
            else:
                yb = np.full(y_train[b].shape, 0, dtype=np.int)

            Membrane.resetNetwork()
            Mb = np.zeros((Membrane.nb_edge, yb.shape[0]))

            if cplusplus:
                Mb = RunWithCplusplus(Membrane, 'openloop', M=Mb, force=xb, y=yb)
            else:
                Mb = Membrane.RunOpenLoop(Mb, xb, yb)

            if gradientMatrix:
                grad_m = []
                for m in Mb:
                    grad_m.append(np.gradient(m, dt))
                Mb = np.array(grad_m)
                Mb += np.random.normal(0, 0.0001, Mb.shape)
            else:
                Mb += np.random.normal(0, 0.001, Mb.shape)

            if i==0:
                M_train[int(y_train[b][0])].append(Mb)
                Y_train[int(y_train[b][0])].append(yb)
            else:
                for c in range(0, nb_classes):
                    if c!=y_train[b][0]:
                        M_train[c].append(Mb)
                        Y_train[c].append(yb)

    M_train = [sm.add_constant(np.hstack(M).T, has_constant='add').T for M in M_train]
    Y_train = [np.hstack(Y) for Y in Y_train]
    ground_truth = np.hstack(ground_truth)

    regressed_rMC = []
    for c in range(0, nb_classes):
        model = sm.OLS(Y_train[c], M_train[c].T)
        results = model.fit()
        wMC[c]=results.params
        regressed = np.matmul(wMC[c], M_train[c])
        regressed_rMC.append(regressed)
    regressed_rMC = np.array(regressed_rMC)
    prediction = np.argmax(regressed_rMC, axis=0)

    space = 0
    regressed_r = []
    for i in range(0, train_batch_size):
        v = prediction[space:space + vowel_length[i]]
        counts = np.bincount(v)
        val = np.argmax(counts)
        regressed_r = np.append(regressed_r, np.full((vowel_length[i]), val))
        space += vowel_length[i]

    plt.plot(ground_truth, 'r', label='ground truth')
    plt.plot(regressed_r, '--g', label='rounded learned signal')
    plt.xlabel('time')
    plt.ylabel('class')
    plt.title('training regression vs. ground truth')
    plt.legend()
    plt.savefig('figures/mctraining3d12.png')
    plt.close()

    training_accuracy = (1 - (np.count_nonzero(ground_truth-regressed_r) / ground_truth.shape[0])) * 100
    print('training accuracy: ' + str(training_accuracy) + '%')

    return wMC

def TestingMC(Membrane, wMC, X_test, y_test):
    vowel_length = []

    M_test = []
    Y_test = []
    for c in range(0, nb_classes):
        Y_test.append([])
        M_test.append([])

    ground_truth = []
    for b in tqdm.tqdm(range(0, test_batch_size), desc='testing'):
        xb = X_test[b]
        vowel_length.append(xb.shape[0])
        ground_truth.append(y_test[b])
        for c in range(0, nb_classes):
            Membrane.resetNetwork()

            Mb = np.zeros((Membrane.nb_edge, xb.shape[0]))
            if cplusplus:
                Mb = RunWithCplusplus(Membrane, 'closedloop', M=Mb, force=xb, w=wMC[c])
            else:
                Mb = Membrane.RunClosedLoop(Mb, xb, wMC[c])

            if gradientMatrix:
                grad_m = []
                for m in Mb:
                    grad_m.append(np.gradient(m, dt))
                grad_m = np.array(grad_m)
                M_test[c].append(grad_m[:, int(Membrane.washout_time / dt):])
            else:
                M_test[c].append(Mb)


    M_test = [sm.add_constant(np.hstack(M).T, has_constant='add').T for M in M_test]
    ground_truth = np.hstack(ground_truth)

    regressed_rMC = []
    for c in range(0, nb_classes):
        regressed = np.matmul(wMC[c], M_test[c])
        regressed_rMC.append(regressed)
        plt.plot(regressed, label='classifier #' + str(c))

    plt.xlabel('time')
    plt.ylabel('regressed value')
    plt.title('One-versus-all Classification for 12 different classifier')
    plt.savefig('figures/oneversusall.png')
    plt.close()

    regressed_rMC = np.array(regressed_rMC)
    prediction = np.argmax(regressed_rMC, axis=0)
    space = 0
    regressed_r = []

    for i in range(0, test_batch_size):
        v = prediction[space:space + vowel_length[i]]
        counts = np.bincount(v)
        val = np.argmax(counts)
        regressed_r = np.append(regressed_r, np.full((vowel_length[i]), val))
        space += vowel_length[i]

    plt.plot(ground_truth, 'r', label='ground truth')
    plt.plot(regressed_r, '--g', label='rounded predicted signal')
    plt.xlabel('time')
    plt.ylabel('class')
    plt.title('testing regression vs. ground truth')
    plt.legend()
    plt.savefig('figures/mctesting3d12.png')

    testing_accuracy = (1 - (np.count_nonzero(ground_truth - regressed_r) / ground_truth.shape[0])) * 100
    print('training accuracy: ' + str(testing_accuracy) + '%')
