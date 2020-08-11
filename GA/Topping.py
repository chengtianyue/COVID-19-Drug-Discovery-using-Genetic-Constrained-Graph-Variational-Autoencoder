def cut(input_list,reps,number):
	maxi = len(input_list)-1
	assert maxi >= (reps * number)
	output_list = []
	for a in range(reps * number, reps * (number+1)):
		output_list.append(input_list[a])
		if a == maxi:
			break
	return output_list

