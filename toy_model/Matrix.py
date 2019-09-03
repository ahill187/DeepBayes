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
    """Given a matrix and a vector, performs matrix multiplication

    @type matrix: array
    @param matrix: A matrix of real numbers
    @type vector: array
    @param vector: A vector of real numbers
    @rtype: array
    @returns: A vector of real numbers
    """
    
    vector_product = []
    for i in range(len(matrix)):
        adder = 0
        for j in range(len(matrix[0])):
            adder = adder + matrix[i][j]*vector[j]
        vector_product.append(adder)
    return vector_product

def DotProduct(vector1, vector2):
    """Given two vectors, computes the dot product

    @type vector1: array
    @param vector1: A vector of real numbers
    @type vector2: array
    @param vector2: A vector of real numbers
    @rtype: array
    @returns: A positive real number
    """
    
    product = 0
    for i in range(len(vector1)):
        product = product + vector1[i]*vector2[i]
    return product

def Column(matrix, column):
    """Given a matrix and a column number, return the column from that matrix

    @type matrix: array
    @param matrix: A matrix of real numbers
    @type column: int
    @param column: The index of the column
    @rtype: array
    @returns: An array column
    """
    
    c = []
    for i in range(len(matrix)):
        c.append(matrix[i][column])
    return c

def Row(matrix, row):
    """Given a matrix and a row number, return the row from that matrix

    @type matrix: array
    @param matrix: A matrix of real numbers
    @type row: int
    @param row: The index of the row
    @rtype: array
    @returns: An array row
    """
    return matrix[row]

def VecScalar(vector, scalar):
    """Given a vector and a scalar, perform scalar multiplication
    @type vector: array
    @param vector: A vector of real numbers
    @type scalar: float
    @param scalar: A real number
    @rtype: array
    @returns: A vector of real numbers
    """
    scalar_product = [i*scalar for i in vector]
    return scalar_product

def VecAdd(vector1, vector2):
    """Given two vectors, perform vector addition

    @type vector1: array
    @param vector1: A vector of real numbers
    @type vector2: array
    @param vector2: A vector of real numbers
    @rtype: array
    @returns: An vector of real numbers
    """
    
    vector = []
    for i in range(len(vector1)):
        vector.append(vector1[i]+vector2[i])
    return vector

def VecSub(vector1, vector2):
    """Given two vectors, perform vector subtraction

    @type vector1: array
    @param vector1: A vector of real numbers
    @type vector2: array
    @param vector2: A vector of real numbers
    @rtype: array
    @returns: An vector of real numbers
    """
    
    vector = []
    for i in range(len(vector1)):
        vector.append(vector1[i]-vector2[i])
    return vector

def VecDivide(vector1, vector2):
    """Given two vectors, perform component-wise division

    @type vector1: array
    @param vector1: A vector of real numbers
    @type vector2: array
    @param vector2: A vector of real numbers
    @rtype: array
    @returns: A vector of real numbers
    """
    vector = []
    for i in range(len(vector1)):
        vector.append(vector1[i]/vector2[i])
    return vector

def VecMult(vector1, vector2):
    """Given two vectors, perform component-wise multiplication

    @type vector1: array
    @param vector1: A vector of real numbers
    @type vector2: array
    @param vector2: A vector of real numbers
    @rtype: array
    @returns: A vector of real numbers
    """
    vector = []
    for i in range(len(vector1)):
        vector.append(vector1[i]*vector2[i])
    return vector

def MatScalar(matrix, scalar):
    """Given a matrix and a scalar, perform scalar multiplication

    @type matrix: array
    @param matrix: A matrix of real numbers
    @type scalar: float
    @param scalar: A real number
    @rtype: array
    @returns: A matrix of real numbers
    """
    mat = []
    for i in range(len(matrix)):
        mat.append(VecScalar(matrix[i], scalar))

    return mat
