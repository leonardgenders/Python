# CHALLENGE #1: Implement a function called count_characters that takes in two arguments: a string representing the full_text and a second string representing the characters to be counted.
# The function should return a dictionary containing the character as the key and the number of occurrences found in the full_text as the value.
# NOTE: You should return the number of characters for only the characters found in string characters argument. Additionally, you should count uppercase and lowercase letters as the same letter.

def count_characters(full_text, characters):
    """
    A function that takes in some text and counts the specified characters.
    
    Parameters
    ----------
    full_text: The text to search
    characters: The characters to look for
    
    Returns
    -------
    count_dict: A dictionary containing the character as the key and the 
                 count as the value
    """
    
    # Initialize the dictionary to return
    count_dict = {}
    
    
    # for each character in characters.lower(): make case insensitive
    for crctr in characters.lower():
    
    # for each item in full_text.lower(): make case insensitive
        for i in full_text.lower():
        
        # if item is equivalent to the character:
            if i == crctr:
            
            # if i is already in the count dictioanry
                if i in count_dict:
                
                # add an additional count 
                    count_dict[i] += 1
                
                # otherwise (else) add to the dictionary with i as the key
                else:
                    count_dict[i] = 1
        
    return count_dict


# CHALLENGE #2: Implement a function called find_top_3_rows that takes in a pandas DataFrame (i.e., the argument named the_data), and a specified column name from that DataFrame (argument named the_col)
# and returns a DataFrame called ret_df that contains the top 3 rows with the largest values from the passed in column, the_col.

def find_top_3_rows(the_data, the_col):
    """
    A function that receives a DataFrame and the name of a column in that DataFrame.
    It should return a new DataFrame with the top 3 rows with the largest values
    from the column specified.
    
    Parameters:
    -----------
    the_data: a DataFrame
    
    the_col: the name of the column in theData that you are looking for largest values of
    
    Returns:
    --------
    ret_df: a DataFrame containing the top three rows of the passed in theData based on 
           the column name passed in, theCol
    """
    
    # YOUR CODE HERE
    # import the pandas package
    import pandas as pd
    
    # sort the data frame by the_col, desending order, and head of three
    ret_df = the_data.sort_values(by=the_col, ascending = False).head(3)
    
    return ret_df


# CHALLENGE #3: There is file named restaurant.csv in the subfolder data. Create a function called get_rows_as_list that takes in the name of the subdirectory (i.e., the argument sub_dir) and the name of the file 
# (i.e., the argument named the_file), and returns a list called ret_list that returns each row of the file as an element of the list.

def get_rows_as_list(sub_dir, the_file):
    '''
    A function that receives the name of a subfolder/subdirectory and the name 
    of a file in that subfolder. The function returns a list where each element
    of the list is a row in the file.
    
    Parameters:
    -----------
    sub_dir: the name of the subfolder/subdirectory
    
    the_file: the name of the file that you want to read
    
    Returns:
    --------
    ret_list: a list containing each row of the file as an element of the list
    '''
    # YOUR CODE HERE
    # combine two strings for the file location and name of the file
    with open ('./' + sub_dir + '/' + the_file, 'r') as f:
    
    # use .readlines() to return a list
        ret_list = f.readlines()
    
    # for each row in ret_list print the row
        for row in ret_list:
            print(row)
    
    return ret_list
