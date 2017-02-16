import math

class Node:

    def __init__(self, listVecs, par, undrThr):
        self.vectors = listVecs
        self.parent = par
        self.isLeft = undrThr
        self.pair = None
        self.leftCh = None
        self.rightCh = None


    def __str__(self):
        s = str(len(self.vectors))+" vectors -- "
        if self.pair is None:
            s += "Predict "+str(int(self.vectors[0][-1]))
        else:
            s += "Is x"+str(self.pair[0] + 1)+" <= "+str(round(self.pair[1], 3))+"?"

        if self.parent is not None:
            s += " -- ( "
            if self.isLeft:
                s += "Left "
            else:
                s += "Right "

            s += "child of ("+str(self.parent)+") )"

        return s


    # used in id3 and main
    def notPure(self):
        labelsOfVecs = {}
        numLabels = 0

        for v in self.vectors:
            label = v[-1]
            if label in labelsOfVecs:
                labelsOfVecs[label] += 1
            else:
                labelsOfVecs[label] = 1
                numLabels += 1
                if numLabels > 1:
                    return True

        return False


    # Used in getEntropy
    def getDistLabels(self, vectors):
        labelsOfVecs = {}
        numVectors = len(vectors)

        for v in vectors:
            label = v[-1]
            if label in labelsOfVecs:
                labelsOfVecs[label] += 1
            else:
                labelsOfVecs[label] = 1

        dist = [(l, (labelsOfVecs[l]/numVectors)) for l in labelsOfVecs.keys()]

        #print(str(dist))

        return dist


    # Used in getEntropy
    def getLnProd(self, prob):
        prod = 0

        if prob != 0:
            prod = math.log(prob)*prob*(-1)

        return prod


    # Used in getCondEntr and getInfoGain
    def getEntropy(self, vectors):
        distOfLabels = self.getDistLabels(vectors)
        logProducts = [self.getLnProd(x[1]) for x in distOfLabels]
        entropy = sum(logProducts)

        #print(str(entropy))

        return entropy


    # Used in getCondEntr and id3
    def splitVecs(self, vectors, ftr, thr, vecsZ0=[], vecsZ1=[]):
        for v in vectors:
            if v[ftr] <= thr:
                vecsZ1.append(v)
            else:
                vecsZ0.append(v)


    # Used in getInfoGain
    def getCondEntr(self, vectors, ftThPair):
        numVectors = len(vectors)
        ftr = ftThPair[0]
        thr = ftThPair[1]
        vecsZ0 = []
        vecsZ1 = []

        self.splitVecs(vectors, ftr, thr, vecsZ0, vecsZ1)

        prZ0 = len(vecsZ0)/numVectors
        prZ1 = len(vecsZ1)/numVectors
        entrXZ0 = self.getEntropy(vecsZ0)
        entrXZ1 = self.getEntropy(vecsZ1)
        condEnt = prZ0*entrXZ0 + prZ1*entrXZ1

        #print(str(condEnt))

        return condEnt


    # Used in id3
    def getInfoGain(self, vectors, pair):
        entropy = self.getEntropy(vectors)
        condEntr = self.getCondEntr(vectors, pair)
        ig = entropy - condEntr

        return ig


    # Used in id3
    def getFtrThrshPairs(self, vectors, numFeatures):
        listOfValueLists = []
        listOfThrshLists = []
        listOfPairs = []

        for i in range(numFeatures):
            values = set()

            for j in range(len(vectors)):
                values.add(vectors[j][i])

            listOfValueLists.append(sorted(tuple(values)))

        for l in listOfValueLists:
            thresholds = set()

            for x in range(len(l) - 1):
                thresh = (l[x] + l[x+1])/2.0
                thresholds.add(thresh)

            listOfThrshLists.append((sorted(tuple(thresholds))))

        for x in range(numFeatures):
            l = listOfThrshLists[x]

            for t in l:
                listOfPairs.append((x, t))

        return listOfPairs


    # Recursive function, used in main
    def id3(self, numFeatures=4):
        vecsZ0 = []
        vecsZ1 = []

        ftrThrshPairs = self.getFtrThrshPairs(self.vectors, numFeatures)
        infoGains = {pair: self.getInfoGain(self.vectors, pair) for pair in ftrThrshPairs}
        pair = max(infoGains, key=infoGains.get)

        #print(str(p)+" : "+str(infoGains[p]))

        self.splitVecs(self.vectors, pair[0], pair[1], vecsZ0, vecsZ1)

        self.pair = pair
        self.leftCh = Node(vecsZ1, self, True)
        self.rightCh = Node(vecsZ0, self, False)

        if self.leftCh.notPure():
            self.leftCh.id3()

        if self.rightCh.notPure():
            self.rightCh.id3()


def loadData(filename):
    listOfVectors = []

    with open(filename, encoding='utf-8') as f:
        for line in f:

            vec = []

            for num in line.split():
                n = float(num)
                vec.append(n)

            listOfVectors.append(tuple(vec))

    return tuple(listOfVectors)


# Recursive function to print out tree
def recPrint(root):

    print(str(root))

    if root.leftCh is not None:
        recPrint(root.leftCh)

    if root.rightCh is not None:
        recPrint(root.rightCh)


# hardcoded version of tree
def predictLabel(v):
    x2 = v[1]
    x3 = v[2]
    x4 = v[3]

    if x3 <= 2.6:
        return 1
    else:
        if x3 <= 4.95:
            if x4 <= 1.65:
                return 2
            else:
                if x2 <= 3:
                    return 3
                else:
                    return 2
        else:
            return 3


def main():
    trainData = loadData("hw3train.txt")
    testData = loadData("hw3test.txt")
    root = Node(trainData, None, None)

    if root.notPure():
        root.id3()

    print("\nTree: \n")

    recPrint(root)

    errorCount = 0
    vecCount = 0

    for v in testData:
        vecCount += 1
        if predictLabel(v) != v[-1]:
            errorCount += 1

    error = errorCount/vecCount

    print("\nError on test data: "+str(100*round(error, 4))+"%")


main()