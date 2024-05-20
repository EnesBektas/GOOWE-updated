import itertools
import numpy as np
import sympy as sp


class ProbabilityCalculation:
    def __init__(self, numOfClasses):
        self.p_array = np.zeros(numOfClasses - 1)
        self.p_total = np.zeros(numOfClasses - 1)
        self.curDim = 1
        self.prevDim = self.curDim

    def calculateProbabilities(self, voteMatrix):
        curMatrix = np.zeros((voteMatrix.shape[0], voteMatrix.shape[1]))

        self.curDim = 1
        self.prevDim = 1

        #print("voteMatrix: ", voteMatrix)
        #print("empty curMatrix: ", curMatrix)

        for i in range(voteMatrix.shape[0]):

            curMatrix[i] = voteMatrix[i]
            if i == 0:
                continue
            #print(i + 1, " dim curMartix: ", curMatrix)

            self.prevDim = self.curDim
            self.curDim = np.linalg.matrix_rank(curMatrix)

            #print("rank of curMatrix: ", self.curDim)

            if self.curDim == self.prevDim:
                self.p_array[self.curDim - 1] += 1
            self.p_total[self.prevDim - 1] += 1

            if self.curDim == len(self.p_array) + 1:
                break

        return self.p_array

    def calculateNumberOfClassifiers(self, p_arr,  p_tot , probabilityLimit=0.999):
        for i in range(len(p_arr)):
            if p_tot[i] == 0:
                p_arr[i] = 1
            else:
                p_arr[i] = p_arr[i] / p_tot[i]
        print("in calculation, p arr: ", p_arr)
        productConst = 1
        for element in p_arr:
            productConst *= 1 - element

        if productConst == 0:
            print("It is not possible to have ideal weights")
            return np.infty

        prob = productConst
        print("product const: ", prob)
        sum = 0
        k = 0
        while prob < probabilityLimit:
            if k == 0:
                sum = 1
            else:
                x_array = np.array(list(self.partitions(k, len(p_arr))))
                #print("xarr", x_array)
                for row in x_array:
                    product = 1
                    for i in range(len(row)):
                        product *= pow(p_arr[i], row[i])
                    sum += product
                print("Sum: ", sum)
            k += 1
            prob = productConst * sum
            print("prob: ", prob)

        print("prob: ", prob)
        if k > 0:
            k -= 1
        return k + len(p_arr) + 1
    
    def optimized_calculation(self, p_arr, p_tot, probabilityLimit = 0.999):
        # Vectorized zero handling
        for i in range(len(p_arr)):
            if p_tot[i] == 0:
                p_arr[i] = 1
            else:
                p_arr[i] = p_arr[i] / p_tot[i]
        print("in calculation, p arr: ", p_arr)

        productConst = 1
        for element in p_arr:
            print("elem ", element)
            productConst *= 1 - element

        if productConst == 0:
            print("It is not possible to have ideal weights")
            return np.infty

        prob = productConst
        print("product const: ", prob)

        sum = 0
        k = 0

        while prob < probabilityLimit:
            if k == 0:
                sum = 1
            else:
                x_array = np.array(list(self.partitions(k, len(p_arr))))
                power_matrix = np.power(np.tile(p_arr, (x_array.shape[0], 1)), x_array)
                sum += np.prod(power_matrix, axis=1).sum()
                print("Sum: ", sum)

            k += 1
            prob = productConst * sum
            print("prob: ", prob)
            print("k: ", k)

        print("prob: ", prob)
        if k > 0:
            k -= 1
        return k + len(p_arr) + 1
    

    def partitions(self, n, b):
        masks = np.identity(b, dtype=int)
        for c in itertools.combinations_with_replacement(masks, n):
            yield sum(c)

    ## average of p values
    def calculateNumberOfClassifiers_withNaiveAssumption(self, p_arr,  p_tot , probabilityLimit=0.999):
        for i in range(len(p_arr)):
            if p_tot[i] == 0:
                p_arr[i] = 1
            else:
                p_arr[i] = p_arr[i] / p_tot[i]
                
        print("in calculation, p arr: ", p_arr)

        avg_p = sum(p_arr) / len(p_arr)
        num_of_classes= len(p_arr) + 1

        return calculateWithDerivativeFormula(num_of_classes, avg_p, probabilityLimit)
    
    ## sum(p) / sum(p_tot)
    def calculateNumberOfClassifiers_withNaiveAssumption_2(self, p_arr,  p_tot , probabilityLimit=0.999):

                
        print("in calculation, p arr: ", p_arr)

        avg_p = sum(p_arr) / sum(p_tot)
        num_of_classes= len(p_arr) + 1

        return calculateWithDerivativeFormula(num_of_classes, avg_p, probabilityLimit)


def factorial(m):
    result = 1
    for i in range(2, m+1):
        result *= i
    return result

def calculateWithDerivativeFormula(m, p_value, threshold):
    print(p_value)
    n = m
    while True:
        p = sp.Symbol('p')
        f = pow(p, m - 2) * ( 1 - pow(p , n - m + 1) ) / ( 1 - p )
        derivative_f = sp.diff(f, p, m - 2)
        fnc = derivative_f * pow( (1 - p), m - 1 ) / factorial( m - 2 )

        result = fnc.subs( p, p_value )
        print("\n iter ", n-m)
        print(derivative_f)
        print(fnc)
        print(result)

        if result >= threshold:
            break
        n += 1

    return n

