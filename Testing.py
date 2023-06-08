
from Parking import *

def test_circular(n, m):
    
    d = {}
    park = Park([1] * m, n, circular = True)

    for i in range(n**m):
        filling = tuple(park.lot)
        if filling in d:
            d[filling] += 1
        else:
            d[filling] = 1
        park.next()

    return d

def print_by_conjugacy_class():
    d = test_circular(4, 4)

    print("Partition = 1,1,1,1")
    print(d[(1,2,3,4)])
    print()
    print("Partition = 2,1,1")
    print(d[(2,1,3,4)], d[(1,3,2,4)], d[(1,2,4,3)], d[(4,2,3,1)])
    print(d[(3,2,1,4)], d[(1,4,3,2)])
    print()
    print("Partition = 3,1")
    print(d[(3,1,2,4)], d[1,4,2,3], d[(4,2,1,3)], d[(4,1,3,2)])
    print(d[(2,3,1,4)], d[1,3,4,2], d[(3,2,4,1)], d[(2,4,3,1)])
    print()
    print("Partition = 2,2")
    print(d[(2,1,4,3)], d[(4,3,2,1)], d[(3,4,1,2)])
    print()
    print("Partition: 4")
    print(d[(2,3,4,1)], d[(4,1,2,3)])
    print(d[(2,4,1,3)], d[(3,1,4,2)])
    print(d[(3,4,2,1)], d[(4,3,1,2)])

def print_by_cosets_of_cyclic_group():
    d = test_circular(4, 4)

    print("coset representative: 1,2,3,4. ")
    print("e, (1234), (13)(24), (1432)")
    print(d[(1,2,3,4)], d[(2,3,4,1)], d[(3,4,1,2)], d[(4,1,2,3)])
    print()
    print("coset representative: 1,2,4,3")
    print("(34), (124), (1423), (132)")
    print(d[(1,2,4,3)], d[(2,4,3,1)], d[(4,3,1,2)], d[(3,1,2,4)])
    print()
    print("coset representative: 1,3,2,4")
    print("(23), (134), (1243), (142)")
    print(d[(1,3,2,4)], d[(3,2,4,1)], d[(2,4,1,3)], d[(4,1,3,2)])
    print()
    print("coset representative: 1,3,4,2")
    print("(234), (1324), (143), (12)")
    print(d[(1,3,4,2)], d[(3,4,2,1)], d[(4,2,1,3)], d[(2,1,3,4)])
    print()
    print("coset representative: 1,4,2,3")
    print("(243), (14), (123), (1342)")
    print(d[(1,4,2,3)], d[(4,2,3,1)], d[(2,3,1,4)], d[(3,1,4,2)])
    print()
    print("coset representative: 1,4,3,2")
    print("(24), (14)(23), (13), (12)(34)")
    print(d[(1,4,3,2)], d[(4,3,2,1)], d[(3,2,1,4)], d[(2,1,4,3)])
    