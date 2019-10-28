import numpy as np

## Functions:

def get_valid_sequences(n=4,k=4,j=2):
    options = range(k+1)
    block = np.array([1]).reshape((1,1))
    # fill in 2 to n-2 digits
    for i in range(1,n-2): # index
        if i+1 > block.shape[1]:
            hblocks = []
            for r in range(block.shape[0]): # row index
                for opt in options:
                    if block[r,-1] != opt:
                        hblock = np.hstack((block[r,:],opt))
                        hblocks.append(hblock)
            block = np.array(hblocks)
    
    # the n-1 digit
    hblocks = []
    for r in range(block.shape[0]):
        for opt in options:
            if block[r,-1] != opt and j != opt:
                hblock = np.hstack((block[r,:],opt))
                hblocks.append(hblock)
    block = np.array(hblocks)
    
    # add last digit = j
    lastcolumn = j*np.ones((block.shape[0],1))
    block = np.hstack((block,lastcolumn))
    block = np.array(block,dtype=np.int)
    return(block)

def count_combinations(n=4,k=4,j=2):
    sum = 0
    sign = 1
    for i in range(n-2,-1,-1):
        sum += sign*k**i
        sign *= -1
    if j==1: sum += sign
    print('n = {}, k = {}, j = {}'.format(n,k,j))
    print(sum)
    return(sum)
    
def count_combinations_m(n=4,k=4,j=2,modulo=10**10+7):
    sum = 0
    sign = 1
    for i in range(n-2,-1,-1):
        sum += sign*pow(k% modulo,i,modulo) % modulo 
        sign *= -1
    if j==1: sum += sign
    sum = sum % modulo
    print('n = {}, k = {}, j = {}'.format(n,k,j))
    print(sum)
    return(sum)
    
## Questions:
modulo = 10**10+7

n = 4
k = 4
j = 2 #  1<=j<=k
count_combinations(n,k,j)

n = 4
k = 100
j = 1 #  1<=j<=k
count_combinations(n,k,j)

n = 100
k = 100
j = 1 #  1<=j<=k
count_combinations_m(n,k,j,modulo)

n = 347
k = 2281
j = 829 #  1<=j<=k
count_combinations_m(n,k,j,modulo)

n = int(1.26*10**6)
k = int(4.17*10**6)
j = 1 #  1<=j<=k
count_combinations_m(n,k,j,modulo)

n = int(10**7)
k = int(10**12)
j = 829 #  1<=j<=k
count_combinations_m(n,k,j,modulo)
