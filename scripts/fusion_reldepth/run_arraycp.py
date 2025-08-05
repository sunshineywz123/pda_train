import numpy as np

def rot_matrix(a):
    n = a.shape[0]
    m = n // 2
    
    for i in range(1, m + 1):
        # 4 directions
        # 1. j, k -> j-1, k+1
        # 2. j, k -> j-1, k-1
        # 3. j, k -> j+1, k-1
        # 4. j, k -> j+1, k+1
        last = None 
        for ii in range(i):
            j = m + i - ii
            k = m + ii
            new_j, new_k = j - 1, k + 1
            last = a[new_k, new_j]
            a[new_k, new_j] = a[k, j]
            
        for ii in range(i):
            j = m + i - ii
            k = m - ii
            new_j, new_k = j - 1, k - 1
            import ipdb; ipdb.set_trace()
            last, a[new_k, new_j] = a[new_k, new_j], last
            
        for ii in range(i):
            j = m - i + ii
            k = m - ii
            new_j, new_k = j + 1, k - 1
            last, a[new_k, new_j] = a[new_k, new_j], last
            
        for ii in range(i):
            j = m + i - ii
            k = m - ii
            new_j, new_k = j + 1, k + 1 
            import ipdb; ipdb.set_trace()
            if ii == i:
                a[new_k, new_j] = last
            else:
                last, a[new_k, new_j] = a[new_k, new_j], last
            
    return a


def rot2_matrix(a):
    n = a.shape[0]
    # i, j -> n-j, i
    
    for i in range(n):
        for j in range(n):
            a[i, j] = a[n-j-1, i]

if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(rot_matrix(a))
    try:
        print(rot_matrix(a))
    except Exception as e:
        print(a)
    