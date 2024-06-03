import sys
import warnings
import numpy as np
import random

def min_zero_row(zero_mat, mark_zero):
	
	'''
	The function can be splitted into two steps:
	#1 The function is used to find the row which containing the fewest 0.
	#2 Select the zero number on the row, and then marked the element corresponding row and column as False
	'''

	#Find the row
	min_row = [99999, -1]

	for row_num in range(zero_mat.shape[0]): 
		if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
			min_row = [np.sum(zero_mat[row_num] == True), row_num]

	# Marked the specific row and column as False
	zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
	mark_zero.append((min_row[1], zero_index))
	zero_mat[min_row[1], :] = False
	zero_mat[:, zero_index] = False

def mark_matrix(mat):

	'''
	Finding the returning possible solutions for LAP problem.
	'''

	#Transform the matrix to boolean matrix(0 = True, others = False)
	cur_mat = mat
	zero_bool_mat = (cur_mat == 0)
	zero_bool_mat_copy = zero_bool_mat.copy()

	#Recording possible answer positions by marked_zero
	marked_zero = []
	while (True in zero_bool_mat_copy):
		min_zero_row(zero_bool_mat_copy, marked_zero)
	
	#Recording the row and column positions seperately.
	marked_zero_row = []
	marked_zero_col = []
	for i in range(len(marked_zero)):
		marked_zero_row.append(marked_zero[i][0])
		marked_zero_col.append(marked_zero[i][1])

	#Step 2-2-1
	non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))
	
	marked_cols = []
	check_switch = True
	while check_switch:
		check_switch = False
		for i in range(len(non_marked_row)):
			row_array = zero_bool_mat[non_marked_row[i], :]
			for j in range(row_array.shape[0]):
				#Step 2-2-2
				if row_array[j] == True and j not in marked_cols:
					#Step 2-2-3
					marked_cols.append(j)
					check_switch = True

		for row_num, col_num in marked_zero:
			#Step 2-2-4
			if row_num not in non_marked_row and col_num in marked_cols:
				#Step 2-2-5
				non_marked_row.append(row_num)
				check_switch = True
	#Step 2-2-6
	marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

	return(marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
	cur_mat = mat
	non_zero_element = []

	#Step 4-1
	for row in range(len(cur_mat)):
		if row not in cover_rows:
			for i in range(len(cur_mat[row])):
				if i not in cover_cols:
					non_zero_element.append(cur_mat[row][i])
	min_num = min(non_zero_element)

	#Step 4-2
	for row in range(len(cur_mat)):
		if row not in cover_rows:
			for i in range(len(cur_mat[row])):
				if i not in cover_cols:
					cur_mat[row, i] = cur_mat[row, i] - min_num
	#Step 4-3
	for row in range(len(cover_rows)):  
		for col in range(len(cover_cols)):
			cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num
	return cur_mat

def hungarian_algorithm(mat): 
	dim = mat.shape[0]
	cur_mat = mat

	#Step 1 - Every column and every row subtract its internal minimum
	for row_num in range(mat.shape[0]): 
		with warnings.catch_warnings(record=True) as w:	
			cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])
			if len(w) > 0:
				sys.exit("No Solution")
	
	for col_num in range(mat.shape[1]): 
		cur_mat[:,col_num] = cur_mat[:,col_num] - np.min(cur_mat[:,col_num])
	zero_count = 0
	while zero_count < dim:
		#Step 2 & 3
		ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
		zero_count = len(marked_rows) + len(marked_cols)

		if zero_count < dim:
			cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

	return ans_pos

def ans_calculation(mat, pos):
  total = 0
  ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
  ans_mat = np.where(ans_mat==0, -1, ans_mat)
  for i in range(len(pos)):
    total += mat[pos[i][0], pos[i][1]]
    ans_mat[pos[i][0], pos[i][1]] = mat[pos[i][0], pos[i][1]]
  return total, ans_mat

def check_zero(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    # Initialize row and column flags
    row_has_zero = [False] * rows
    col_has_zero = [False] * cols

    # Iterate through the matrix and set flags
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                row_has_zero[i] = True
                col_has_zero[j] = True

    # Check if any row or column does not have zero
    for i in range(rows):
        if not row_has_zero[i]:
            return False
    
    for j in range(cols):
        if not col_has_zero[j]:
            return False

    return True

def main():
	n = 5
	value_set = list(range(1,n+1))
	
	m = []
	for _ in range(n):
		row = random.sample(value_set,n)
		m.append(row) #Preferance matrix of men
	w = []
	for _ in range(n):
		row = random.sample(value_set,n)
		w.append(row) #Preferance matrix of women
	print("Preference matrix of men: ")
	print(m)
	print("Preference matrix of women: ")
	print(w)

	transposed = []
	for i in range(len(w[0])):
		sublist = []
		for j in range(len(w)):
			sublist.append(w[j][i])
		transposed.append(sublist)


	mat = abs(m-np.array(transposed))
	print()
	print("Auxillary Matrix |Pref(m,w) - Pref(w,m)|:")
	print(mat)
	
	c = 0
	mat1 = mat
	while not check_zero(mat1):
		'''for i in range(n):
			currentMin = mat1[i][0]  # Assume the first element of the row as the current minimum
			for j in range(1, len(mat1[i])):
				if mat1[i][j] < currentMin:
					currentMin = mat1[i][j]
					if currentMin > c:
						mat1[i][j] = 10000 #Essentially infinity
						c = currentMin'''
		c = c + 1
		mat1 = mat - c
		mat1 = np.where(mat1<0, 0, mat1)

	print()
	print("Reduced Matrix: ")
	print(mat1)
	print(f"Matching at a biasness of: {c}") #degree of biaseness allowed
	
	mat1 = np.where(mat1>c, float('inf'), mat)
	
	print("Hungarian Matrix:")
	print(mat1) #Cost matrix for each match
	ans_pos = hungarian_algorithm(mat1.copy())#Get the element position.
	ans, ans_mat = ans_calculation(mat1, ans_pos)#Get the minimum value and corresponding matrix.
	'''n = len(ans_mat)
	c = ans_mat[0][0]
	for i in range(n):
		for j in range(n):
			if ans_mat[i][j] > c:
				c = ans_mat[i][j]
	print(f"Matching at a biasness of: {int(c)}")'''
	print()
	#Show the result
	print(f"{ans_mat}")
	print(f"Matching at a total cost of: {ans:.0f}")

if __name__ == "__main__":
    main()