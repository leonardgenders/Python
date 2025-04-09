# Processing textual data and building/interpreting Keras models for language classification.
"""
M4 Lab-0 Write Your Own Tokenizer
We will process the text file zillow_description.txt available in the file tree to the left
You should feel free to click open and inspect the content of the file.

Read the text file's entire content into a new string variable called zillow.
Make all of the words in the zillow variable lowercase.
**Q4-0-0 Print the content of the updated zillow variable.
"""

# Read the text file's entire content into a new string variable called zillow.
fh = open("zillow_description.txt", "r")
zillow_original = fh.read()
print(zillow_original)

# Make all of the words in the zillow variable lowercase.
zillow = zillow_original.lower()
# **Q4-0-0 Print the content of the updated zillow variable.
print(zillow)

'''
We will process the updated zillow variable into a list of words.
First let's remove punctuation symbols.
We will use the string library for this task.
Import the string library.
The string library has a pre-built constant of a list of punctuation symbols called punctuation.
**Q4-0-1 Print the content of the punctuation constant
'''
# First let's remove punctuation symbols
# import the string library
import string

# **Q4-0-1 Print the content of the punctuation constant
print(string.punctuation)

'''
Using a for loop, go through each element of the zillow string variable, and 
remove the element from the zillow variable if it is on the punctuation list you just printed in the previous step.
**Q4-0-2 Print the content of the updated zillow variable.
'''

# Using a for loop, go through each element of the zillow string variable, and 
# remove the element from the zillow variable if it is on the punctuation list you just printed in the previous step
for element in zillow:
  if element in string.punctuation:
    zillow = zillow.replace(element, "") # replaces with nothing

# **Q4-0-2 Print the content of the updated zillow variable.
print(zillow)
'''
Split the zillow variable into a Python list of words (i.e. splitting by the white space).
You have now completed "word tokenization" which is an important first step in Natural Language Processing.
**Q4-0-3 Print the Python list of words.
'''

# Split the zillow variable into a Python list of words (i.e. splitting by the white space)
zillow_words = zillow.split(" ") # split by white space

# **Q4-0-3 Print the Python list of words.
print(zillow_words)

'''
Next, our goal is to count the number of positive words in the zillow description.
The positive words we will count are stored in this list given in the starter code.
First, you should initiate a Python dictionary (Reminder: Python dictionary stores key-value pairs).
Then loop through the positive words list with each positive word as a key in your dictionary, with an initial value of 0.
**Q4-0-4 Print your dictionary - make sure all values are zeros.
'''

# FREEZE CODE BEGIN
positive_words=['happy', 'free', 'easy', 'gorgeous', 'generous', 'perfect', 'luxurious', 'good', 'wonderful', 'fantastic', 'amazing', 'lovely']
# FREEZE CODE END

# Initiate a Python dictionary
pos_dict = {}

# Loop through the positive words list with each positive word as a key in your dictionary, with an initial value of 0
for word in positive_words:
  pos_dict[word] = 0 # initial value of 0

# **Q4-0-4 Print your dictionary - make sure all values are zeros.
print(pos_dict)

'''
Next, 
Loop through the zillow word list from earlier.
Count the number of positive words in your word list that are listed in the positive_words variable here.
You should track each positive word's count as its value in the dictionary.
The dictionary should contain key-value pairs with positive words as keys and their counts as values.
**Q4-0-5 Print your updated dictionary (now with final count of each positive word).
'''

# Loop through the zillow word list from earlier, combined the for loop for pos_count
pos_count = 0
for word in zillow_words:
  # count the number of positive words in your word list that are listed in the positive_words variable here
  if word in positive_words: # if word in zillow_words is also in positive_words:
    pos_dict[word] +=1
    pos_count += 1 # track each positive word's count as its value in the dictionary

# **Q4-0-5 Print your updated dictionary (now with final count of each positive word)
print(pos_dict)

'''
**Q4-0-6 Print "There is a total of x positive words in the zillow description." - Replace x with the proper value.
'''

# **Q4-0-6 Print "There is a total of x positive words in the zillow description." - Replace x with the proper value. 
print(f'There is a total of {pos_count} positive words in the zillow description.')



"""
M4 Lab-1 Tokenization with Keras
We will now process the text file zillow_description.txt with Keras's text preprocessing module.
Again you should feel free to click open/download and inspect the content of the file.

First, import Tokenizer and text_to_word_sequence functions from the keras.preprocessing.text module.

Read the text file's entire content into a new string variable called zillow.
**Q4-1-0 Print the content of entire file in its original form.
"""

# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings
# FREEZE CODE END

# import Tokenizer and text_to_word_sequence functions from the kearas.preprocessing.text module
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

# Read the text file's entire content into a new string variable called zillow.
fh = open("zillow_description.txt", "r")
zillow = fh.read()

# **Q4-1-0 Print the content of entire file in its original form.
print(zillow)


'''
Process the zillow text into a list of words using the text_to_word_sequence function.
**Q4-1-1 Print the text_to_word_sequence function output.
Read the function's documentation and review the output here.
'''
# Process the zillow text into a list of words using the text_to_word_sequence function
result = text_to_word_sequence(zillow)

# **Q4-1-1 Print the text_to_word_sequence function output
print(result)

'''
Create a Keras Tokenizer() object.
Consider our zillow file as a single document.
We will use the Tokenizer's fit_on_texts function to fit the Tokenizer().
The Tokenizer() expects to receive a list.
So first turn the zillow string variable into a list with just one element.
Then run the Tokenizer's fit_on_texts() function on the newly created zillow list.
This function creates a dictionary based on the zillow text.
Use the Tokenizer's methods to perform the following:
**Q4-1-2 Print the Tokenizer's dictionary of words and their counts.
**Q4-1-3 Print the Tokenizer's dictionary of words and their uniquely assigned integer indices.
**Q4-1-4 Print "There is/are xxx document(s) in this Tokenizer" - Replace xxx with the number of documents that have been fitted with this tokenizer.
Read the Tokenizer's documentation and review the output here.
'''

# Create a Keras Tokenizer() object
t = Tokenizer()

# Consider our zillow file as a single document
document = [zillow]

# We will use the Tokenizer's fit_on_texts function to fit the Tokenizer(), tokenizer expects to receive a list
t.fit_on_texts(document)

# **Q4-1-2 Print the Tokenizer's dictionary of words and their counts
print(t.word_counts)

# **Q4-1-3 Print the Tokenizer's dictionary of words and their uniquely assigned integer indices.
print(t.word_index)

# **Q4-1-4 Print "There is/are xxx document(s) in this Tokenizer" - Replace xxx with the number of documents that have been fitted with this tokenizer
print(f'There is/are {t.document_count} document(s) in this Tokenizer')

'''
Use the Tokenizer's texts_to_matrix function to create a matrix of each word in the zillow document and each word's count.
**Q4-1-5 Print this matrix.
'''
# Use the Tokenizer's texts_to_matrix function to create a matrix of each word in the zillow document and each word's count
encoded_docs = t.texts_to_matrix(document, mode='count')
# **Q4-1-5 Print this matrix.
print(encoded_docs)
