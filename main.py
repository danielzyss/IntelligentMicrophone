from parameters import *
from funk import *
from GeneticGenerator import ConstructMembraneFromDNA

def main():


    if read_from_dna:
        Membrane, _ = ConstructMembraneFromDNA(input_dna)
    elif new:
        Membrane = GenerateMembrane()
    else:
        Membrane = LoadMembrane()

    X, y = UploadDataset()
    X_train, X_test, y_train, y_test = OrganizeData(X, y)

    if nb_classes==2:
        w = Learning(Membrane, X_train, y_train)
        Testing(Membrane, w, X_test, y_test)
    else:
        w = LearningMC(Membrane, X_train, y_train)
        TestingMC(Membrane, w, X_test, y_test)



if __name__ =="__main__":
    main()