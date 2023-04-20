def print_function(input_string, **kwargs):
    print(input_string + '...')
    
    for key, arg in kwargs.items():
        len_key = len(key)
        len_arg = len(str(arg))
        n_dots = 50 - len_key - len_arg
        print(2*'-', key, n_dots*'.', arg)

    print(50*'_')