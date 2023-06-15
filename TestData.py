import numpy as np

pref_at_i_n8 = [[1062882, 800738, 683089, 601441, 531441, 461441, 379793, 262144], [697089, 959233, 932387, 887601, 844873, 808123, 783931, 800738], [277397, 277397, 421892, 494264, 541262, 583122, 632262, 697089], [54622, 54622, 54622, 108684, 160728, 207166, 248242, 277397], [4992, 4992, 4992, 4992, 18678, 35078, 48497, 54622], [169, 169, 169, 169, 169, 2221, 4299, 4992], [1, 1, 1, 1, 1, 1, 128, 169], [0, 0, 0, 0, 0, 0, 0, 1]]
pref_at_i_n7 = [[65536, 48729, 40953, 35328, 30208, 24583, 16807], [37883, 54690, 53435, 51008, 48808, 47433, 48729], [12373, 12373, 21404, 26326, 29938, 33563, 37883], [1763, 1763, 1763, 4893, 7942, 10573, 12373], [93, 93, 93, 93, 752, 1433, 1763], [1, 1, 1, 1, 1, 64, 93], [0, 0, 0, 0, 0, 0, 1]]
pref_at_i_n6 = [[4802, 3506, 2881, 2401, 1921, 1296], [2341, 3637, 3591, 3461, 3381, 3506], [581, 581, 1252, 1656, 1991, 2341], [51, 51, 51, 257, 451, 581], [1, 1, 1, 1, 32, 51], [0, 0, 0, 0, 0, 1]]
pref_at_i_n5 = [[432, 307, 243, 189, 125], [165, 290, 293, 291, 307], [27, 27, 88, 129, 165], [1, 1, 1, 16, 27], [0, 0, 0, 0, 1]]
pref_at_i_n4 = [[50, 34, 25, 16], [13, 29, 31, 34], [1, 1, 8, 13], [0, 0, 0, 1]]
pref_at_i_n3 = [[8, 5, 3], [1, 4, 5], [0, 0, 1]]
pref_at_i_n2 = [[2, 1], [0, 1]]
pref_at_i_n1 = [[1]]

num_lucky_n7 = [[720, 9792, 44176, 87296, 81136, 33984, 5040], [720, 11640, 59464, 125054, 111124, 33984, 0], [600, 9394, 44021, 76699, 43146, 0, 0], [384, 4996, 17736, 17954, 0, 0, 0], [162, 1448, 2710, 0, 0, 0, 0], [32, 130, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]
num_lucky_n6 = [[120, 1318, 4553, 6388, 3708, 720], [120, 1578, 6063, 8448, 3708, 0], [96, 1170, 3766, 3370, 0, 0], [54, 487, 901, 0, 0, 0], [16, 71, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
num_lucky_n5 = [[24, 204, 504, 444, 120], [24, 244, 634, 444, 0], [18, 155, 263, 0, 0], [8, 38, 0, 0, 0], [1, 0, 0, 0, 0]]
num_lucky_n4 = [[6, 37, 58, 24], [6, 43, 58, 0], [4, 19, 0, 0], [1, 0, 0, 0]]


# June 12th
#Varibales from computations by hand
disp0 = np.array([1,1,2, 2,2,3, 2,3,2,  2,2,3, 2,1,1, 3,2,2,  2,3,2, 3,2,2, 1,2,1])
disp1 = np.array([1,2,1, 0,1,0, 1,0,0,  0,1,0, 1,1,2, 0,0,1,  1,0,0, 0,0,1, 2,1,1])
disp2 = np.array([3] * 27) - disp0 - disp1
total_displacement = 1 * disp1 + 2 * disp2
max_displacement = np.array([(0 if disp1[i] == 0 else 1) if disp2[i] == 0 else 2 for i in range(27)])

# June 14th
# Change of basis computed by hand from character basis to new basis
character_to_new_basis_3 = np.zeros((27,27))
character_to_new_basis_3[0][0] = 1
character_to_new_basis_3[1][1] = 1
character_to_new_basis_3[2][1] = -1
character_to_new_basis_3[4][2] = 1
character_to_new_basis_3[5][2] = -1
character_to_new_basis_3[2][3] = 1
character_to_new_basis_3[3][3] = -1
character_to_new_basis_3[9][4] = 1
character_to_new_basis_3[13][5] = 1
character_to_new_basis_3[5][6] = 1
character_to_new_basis_3[6][6] = -1
character_to_new_basis_3[10][7] = 1
character_to_new_basis_3[11][7] = -1
character_to_new_basis_3[12][7] = -1
character_to_new_basis_3[13][7] = -1
character_to_new_basis_3[18][8] = 1
character_to_new_basis_3[3][9] = 1
character_to_new_basis_3[8][10] = 1
character_to_new_basis_3[9][10] = -1
character_to_new_basis_3[12][11] = 1
character_to_new_basis_3[14][11] = -1
character_to_new_basis_3[7][12] = 1
character_to_new_basis_3[8][12] = -1
character_to_new_basis_3[19][13] = 1
character_to_new_basis_3[20][14] = 1
character_to_new_basis_3[21][14] = -1
character_to_new_basis_3[14][15] = 1
character_to_new_basis_3[21][16] = 1
character_to_new_basis_3[22][16] = -1
character_to_new_basis_3[25][17] = 1
character_to_new_basis_3[6][18] = 1
character_to_new_basis_3[15][19] = 1
character_to_new_basis_3[17][20] = 1
character_to_new_basis_3[18][20] = -1
character_to_new_basis_3[11][21] = 1
character_to_new_basis_3[15][21] = -1
character_to_new_basis_3[22][22] = 1
character_to_new_basis_3[24][23] = 1
character_to_new_basis_3[25][23] = -1
character_to_new_basis_3[16][24] = 1
character_to_new_basis_3[17][24] = -1
character_to_new_basis_3[23][25] = 1
character_to_new_basis_3[24][25] = -1
character_to_new_basis_3[26][26] = 1

character_to_new_basis_3[: , [1,9]] = character_to_new_basis_3[: , [9,1]]
character_to_new_basis_3[: , [2,18]] = character_to_new_basis_3[: , [18,2]]
character_to_new_basis_3[: , [4,12]] = character_to_new_basis_3[: , [12,4]]
character_to_new_basis_3[: , [5,21]] = character_to_new_basis_3[: , [21,5]]
character_to_new_basis_3[: , [7,15]] = character_to_new_basis_3[: , [15,7]]
character_to_new_basis_3[: , [8,24]] = character_to_new_basis_3[: , [24,8]]
character_to_new_basis_3[: , [11,19]] = character_to_new_basis_3[: , [19,11]]
character_to_new_basis_3[: , [14,22]] = character_to_new_basis_3[: , [22,14]]
character_to_new_basis_3[: , [17,25]] = character_to_new_basis_3[: , [25,17]]


basis_labels_3 = ["000", 
                  "S_3*100", "S_2*001", "001", 
                  "S_3*200", "S_2*002", "002",
                  "S_3*011", "S_2*110", "110",
                  "S_3*012", "S_2*012", "S_2*102", "012", "120", "201",
                  "S_3*220", "S_2*220", "220", 
                  "111",
                  "S_3*112", "S_2*112", "112", 
                  "S_3*221", "S_2*221", "221", 
                  "222"]