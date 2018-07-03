# Matrix Script ---------------------------------------
# Date created: 2018-06-14
# Date modified: 2018-06-14
# Author: Ainsleigh Hill
# Email: ainsleigh.hill@cern.ch
#
# Purpose: Matrix algebra functions
#------------------------------------------------------

# vector is a list
# matrix is a list of vectors
# scalar is an element of R

def MatVec(matrix,vector):
    vector_product = []
    for i in range(len(matrix)):
        adder = 0
        for j in range(len(matrix[0])):
            adder = adder + matrix[i][j]*vector[j]
        vector_product.append(adder)
    return vector_product

def DotProduct(vector1, vector2):
    product = 0
    for i in range(len(vector1)):
        product = product + vector1[i]*vector2[i]
    return product

def Column(matrix, column):
    c = []
    for i in range(len(matrix)):
        c.append(matrix[i][column])
    return c

def Row(matrix, row):
    return matrix[row]

def VecScalar(vector, scalar):
    scalar_product = [i*scalar for i in vector]
    return scalar_product

def VecAdd(vector1, vector2):
    vector = []
    for i in range(len(vector1)):
        vector.append(vector1[i]+vector2[i])
    return vector
def VecSub(vector1, vector2):
    vector = []
    for i in range(len(vector1)):
        vector.append(vector1[i]-vector2[i])
    return vector