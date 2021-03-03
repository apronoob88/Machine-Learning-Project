#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

# In[2]:


def load_train(train_path):
    f = open(train_path, encoding="utf8")
    lines = []
    for line in f:
        if line != '\n':
            line = line.strip('\n').split(' ')
            lines.append(line)
    df= pd.DataFrame(lines, columns = ['word', 'state'])
    return df
ans= load_train('SG/train')


# In[3]:


def load_test(test_path):
    f = open(test_path, encoding='utf8')
    lines = []
    for line in f:
        if line != '\n':       
            line = line.strip('\n')
            lines.append(line)
    df= pd.DataFrame(lines, columns = ['word'])
    return df


# In[4]:


def gen_emission_param_table(file_path):
    #input_data = load_train('EN/train')
    input_data = load_train(file_path)
    input_data.columns = ['word', 'state']
    unique_word_list = input_data.word.unique()
    unique_state_list = input_data.state.unique()
    data = {word:(np.zeros(len(unique_state_list))) for word in unique_word_list}  # each word as a column, each column length is the number of unique y state, all entry are 0
                                                                                   # https://chrisalbon.com/python/data_wrangling/pandas_list_unique_values_in_column/
    emission_count_table = pd.DataFrame(data, index = unique_state_list)           # transform the dictionary into colums with index as name of each state(y),  
                                                                                   # columns as each word of x, all entries are 0
    
    y_count_dic = {state:0 for state in unique_state_list}                         # y_count_dic stores Count(y) in a dictionary
    emission_param_table = pd.DataFrame(data, index = unique_state_list)           # emission_count_table stores all Count(y -> x) in a dataframe
                                                                                   # emission_param_table stores all the emission parameters in a dataframe
    print("updating emission_count_table and y_count_dic")
    for index, row in input_data.iterrows():
        word = row['word']
        state = row['state']
        #print(index, word, state)
        #print(index)
        emission_count_table[word][state] += 1
        y_count_dic[state]+=1
    
    print("updating emission_param_table")
    for index, row in input_data.iterrows():
        word = row['word']
        state = row['state']
        emission_param_table[word][state] = emission_count_table[word][state] / y_count_dic[state]
    return emission_param_table, emission_count_table, y_count_dic


# In[5]:


def get_emission_parameter(emission_param_table,x, y):
    if x not in emission_param_table.columns:
        #print(f"word {x} is not found in the data set")
        return 
    result = emission_param_table[x][y]
    return result




# ### Introducing #UNK# to the function

# In[8]:


def gen_emission_param_table_UNK(training_data_path, test_data_path):
    training_data = load_train(training_data_path)
    unique_word_list_training = training_data.word.unique()
    unique_state_list_training = training_data.state.unique()
    
    
    
    test_data = load_test(test_data_path)
    unique_word_list_test = test_data.word.unique()
    
    unk_list = np.setdiff1d(unique_word_list_test, unique_word_list_training) # return the list of words in test data but not in training data
    #non_unk_list_test = np.setdiff1d(unique_word_list_test, unk_list) # return the list of non UNK words in test data
    
    data = {word:(np.zeros(len(unique_state_list_training))) for word in unique_word_list_training}
    data["UNK"] = np.zeros(len(unique_state_list_training))    # add a UNK column to the table
    
    emission_count_table = pd.DataFrame(data, index = unique_state_list_training)           # transform the dictionary into colums with index as name of each state(y),  
                                                                                   # columns as each word of x, all entries are 0
    
    y_count_dic = {state:0 for state in unique_state_list_training}                         # y_count_dic stores Count(y) in a dictionary
    emission_param_table = pd.DataFrame(data, index = unique_state_list_training)           # emission_count_table stores all Count(y -> x) in a dataframe
                                                                                   # emission_param_table stores all the emission parameters in a dataframe
    
    print("updating emission_count_table and y_count_dic")
    for index, row in training_data.iterrows():
        word = row['word']
        state = row['state']
        #print(index, word, state)
        #print(index)
        y_count_dic[state]+=1
        if word not in unk_list:
            emission_count_table[word][state] += 1
        
    
    print("updating emission_param_table")
    k = 0.5
    for index, row in training_data.iterrows():
        word = row['word']
        state = row['state']
        #print(index)
        if word not in unk_list:
            emission_param_table[word][state] = emission_count_table[word][state] / (y_count_dic[state] + k)    
    for state in unique_state_list_training:
        emission_param_table['UNK'][state] = k/(y_count_dic[state] + k)    # compute the UNK value for each state y

    
    #print("unl_list is: ",unk_list)
    #print("y_count_dic is: ", y_count_dic)
    return emission_param_table, unk_list


# In[9]:


def get_emission_parameter_UNK(emission_param_table, unk_list, x, y):

    if x in unk_list:
        result = emission_param_table['UNK'][y]
        #print(f"{x} is tagged as UNK and the this e('UNK'|{y}) is {result}" )
        return result
    elif x not in emission_param_table.columns:
        #print(f"word {x} is not found in the test set")
        return 
    result = emission_param_table[x][y]
    return result


# In[10]:


def get_argmax_y(emission_param_table, unk_list, x):
    if x in unk_list:
        arg_max_y = emission_param_table['UNK'].idxmax()
        #print(f"{x} is tagged as UNK and the this arg_max_y e({x}|y) is {arg_max_y}" )
        return arg_max_y
    
    arg_max_y = emission_param_table[x].idxmax()
    return arg_max_y


# In[11]:


def gen_state(input_path, output_path, emission_param_table, unk_list ):
    with open(input_path, "r", encoding="utf8") as f1, open(output_path, 'w', encoding="utf8") as f2:
        test_list = f1.readlines()
        for word in test_list:
            if word == '\n':
                #print("new sentence")
                f2.write(word)
                continue
            word = word.strip()
            arg_max_y =  get_argmax_y(emission_param_table, unk_list, word)
            output = f'{word} {arg_max_y}\n'
            f2.write(output)




print("Generating output for EN....")
emission_param_table_UNK_test , unk_list = gen_emission_param_table_UNK('EN/train', 'EN/dev.in')
gen_state('EN/dev.in','EN/dev.p2.out', emission_param_table_UNK_test, unk_list)


print("Generating output for CN.....")
emission_param_table_UNK_test_CN , unk_list_CN = gen_emission_param_table_UNK('CN/train', 'CN/dev.in')


gen_state('CN/dev.in','CN/dev.p2.out', emission_param_table_UNK_test_CN, unk_list_CN)

print("Generating output for SG.....")
emission_param_table_UNK_test_SG , unk_list_SG = gen_emission_param_table_UNK('SG/train', 'SG/dev.in')
gen_state('SG/dev.in','SG/dev.p2.out', emission_param_table_UNK_test_SG, unk_list_SG)


print("Evaluation result for EN is: ")
os.system('python EvalScript/evalResult.py EN/dev.out EN/dev.p2.out')

print("Evaluation result for CN is: ")
os.system("python EvalScript/evalResult.py SG/dev.out SG/dev.p2.out")

print("Evaluation result for SG is: ")
os.system("python EvalScript/evalResult.py SG/dev.out SG/dev.p2.out")