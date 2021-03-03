#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

# In[2]:


################### generate emission parameters ##############################
# function to load training set to generate emission table
def load_train(train_path):
    f = open(train_path, encoding="utf8")
    lines = []
    for line in f:
        if line != '\n':
            line = line.strip('\n').split(' ')
            lines.append(line)
    df= pd.DataFrame(lines, columns = ['word', 'state'])
    return df

# function to load dev.in to generate emission table
def load_test(test_path):
    f = open(test_path, encoding='utf8')
    lines = []
    for line in f:
        if line != '\n':       
            line = line.strip('\n')
            lines.append(line)
    df= pd.DataFrame(lines, columns = ['word'])
    return df


def gen_emission_param_table_UNK(training_data_path, test_data_path):
    training_data = load_train(training_data_path)
    unique_word_list_training = training_data.word.unique()
    unique_state_list_training = training_data.state.unique()
    
    
    
    test_data = load_test(test_data_path)
    unique_word_list_test = test_data.word.unique()
    
    unk_list = np.setdiff1d(unique_word_list_test, unique_word_list_training) # return the list of words in test data but not in training data
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

    
    print("unl_list is: ",unk_list)
    print("y_count_dic is: ", y_count_dic)
    return emission_param_table, unk_list


# In[3]:


def get_emission_parameter_UNK(emission_param_table, unk_list, x, y):

    if x in unk_list:
        result = emission_param_table['UNK'][y]
        return result
    elif x not in emission_param_table.columns:
        return 
    result = emission_param_table[x][y]
    return result


# In[4]:


################### generate transition parameters ##############################

# load_train here is different from part 2
# read the train data line by line, replace each '\n' with STOP and START
# so that it is clear when each sentence end and when each sentence started
def load_train_transition(train_path):
    f = open(train_path, encoding="utf8")
    lines = []
    # add a START to before first sentence
    lines.append(['START'])
    for line in f:
        if line == '\n':
            lines.append(['STOP'])
            lines.append(['START'])
        
        else:
            line = line.strip('\n').split(' ')
            del line[0]  # keep the states only
            lines.append(line)
        
    df= pd.DataFrame(lines, columns = ['state'])
    return df

def create_transition_count_table(df_train):
    unique_state_list = df_train.state.unique()
    
    data = {state:(np.zeros(len(unique_state_list))) for state in unique_state_list}
    transition_count_table = pd.DataFrame(data, index = unique_state_list)   # there will one extra STOP row and a START column
    transition_count_table = transition_count_table.drop('STOP')  #drop the extra STOP column
    transition_count_table = transition_count_table.drop(columns=['START']) #drop the extra START column
    return transition_count_table

def create_y_count_dic(df_train):
    unique_state_list = df_train.state.unique()
    y_count_dic = {state:0 for state in unique_state_list}
    y_count_dic.pop('STOP', None)  # remove the extra STOP state since we are inclusing count(STOP) when computing the transition paraameter 
    return y_count_dic

def gen_transition_param_table(train_path):
    input_data = load_train_transition(train_path)
    transition_count_table = create_transition_count_table(input_data)
    transition_param_table = transition_count_table.copy(deep=True) # create a empty transition_param_table. Rows and columns of transition_param_table same as transition_count_table
    y_count_dic = create_y_count_dic(input_data)
    print('Generating transition parameter table')
    for i in range(len(input_data) -1):          # len(input_data) -1 coz we iterating from y_i = 0 to y_i = n-1
        y_i = input_data['state'][i]
        y_i_p1 = input_data['state'][i+1]        # y_i_p1 stands for yi_+1 (y_i plus 1)
        if y_i != 'STOP':                        # we do not count the transition from STOP to some other state
            transition_count_table[y_i_p1][y_i] += 1
            y_count_dic[y_i] += 1
            
    cols_list = transition_count_table.columns.values.tolist()      
    index_list = transition_count_table.index.values.tolist()
    for index in index_list:
        for col in cols_list:
            transition_param_table[col][index] = transition_count_table[col][index]/y_count_dic[index]  # a(y_i, y_i+1) = Count(y_i, y_i+1) / Count(y_i)
    return transition_param_table


# In[5]:


# load dev.in, identify the sentences. store each sentence as a list
# return a nested list sentence_list stroing all sentences (list)
def get_sentence_list(test_path):
    f = open(test_path, encoding="utf8")
    sentence_list = [] # a list of all sentences
    sentence = []   # a list storing 1 sentence
    for word in f:
        if word != '\n':
            word = word.strip('\n')
            sentence.append(word)
        
        else:
            sentence_list.append(sentence)
            sentence = []
    return sentence_list

#sentence_list = get_sentence_list('SG/dev.in')

def get_state_list(train_path):
    f = open(train_path, encoding="utf8")
    state_list = []
    # add a START to before first sentence
    for line in f:
        if line != '\n':
            line = line.strip('\n').split(' ')
            state = line[1]
            if state not in state_list:
                state_list.append(state)
    return state_list


def viterbi(em_param_table, tr_param_table, unk_list, sentence, state_list):
    # create the 2-D pi(j,u) table, initialize all values as 0 at beginning
    # sentence: a list storing all words in a sentence as its element
    # state_list : a list storing all the states except START and STOP
    
    s = (len(state_list),len(sentence))
    pi_table = pd.DataFrame(np.zeros(s), columns=sentence, index=state_list)
    start_node = 1 # Add a START node in pi table, pi(0,START)
    stop_node = 0  # Add a STOP node in the pi table, pi(n+1,STOP)
    
    
    
    sentence = pi_table.columns.values.tolist().copy()
    states = pi_table.index.values.tolist()
    
    index_column = [i for i in range(len(sentence))]  #replace the column name with position number ie. i = 0, ...n
    #rint(index_column)
    
    pi_table.columns = index_column
    for state, row in pi_table[[0]].iterrows():
        pi_table[0][state] = start_node * tr_param_table[state]['START'] * get_emission_parameter_UNK(em_param_table,unk_list, sentence[0], state)
    
    #this represents pi(i,u) for all state u at position i
    temp_result = [] # a temp table storing the pi values at each position
    for i in range(0, len(sentence) - 1):
        for state_next, row_next in pi_table[[i+1]].iterrows():
           #print("#####################")
            for state, row in pi_table[[i]].iterrows():                   
                value = pi_table[i][state] * tr_param_table[state_next][state] * get_emission_parameter_UNK(em_param_table,unk_list, sentence[i+1], state_next)
                temp_result.append(value)
            pi_table[i+1][state_next] = max(temp_result)
            temp_result = []
    
    # from last position to STOP node
    n = len(sentence) - 1
    for state, row in pi_table[[n]].iterrows():
        value = pi_table[n][state] * tr_param_table['STOP'][state]
        temp_result.append(value)    

    stop_node = max(temp_result)
    temp_result = []
    
    optimum_path_np = []
    
    temp_arr = np.array([])
    
    for state, row in pi_table.iterrows():
        pi_value = row[n]
        value = pi_value * tr_param_table['STOP'][state]
        temp_arr = np.append(temp_arr, value)
    max_y_index = np.argmax(temp_arr)
    arg_max_y_np = states[max_y_index]
    optimum_path_np.insert(0,arg_max_y_np )

    for i in range (len(sentence)-2, -1,-1):
        temp_arr = np.array([])
        for state, row in pi_table.iterrows():
            pi_value = row[i]
            value = pi_value * tr_param_table[optimum_path_np[0]][state]   # pi (i, u) * a (u, y*_i+1)
            temp_arr = np.append(temp_arr,value)
        max_y_index = np.argmax(temp_arr)
        
        arg_max_y_np = states[max_y_index]
        optimum_path_np.insert(0, arg_max_y_np)
    return optimum_path_np


# In[9]:


def gen_state(train_path, input_path, output_path, em_param_table, tr_param_table, unk_list ):
    sentence_list = get_sentence_list(input_path)
    state_list = get_state_list(train_path)
    #viterbi(em_param_table, tr_param_table, unk_list, sentence, state_list):
    with open(output_path, 'w', encoding="utf8") as f:
        #for sentence in sentence_list:
        print("No. of sentence: ",len(sentence_list))
        for i in range(len(sentence_list)):
            sentence = sentence_list[i]
            print(i)
            #pi_table = create_pi_table(sentence, state_list)
            optimum_state = viterbi( em_param_table, tr_param_table, unk_list, sentence,state_list)
            for i in range (len(sentence)):
                #print("sentence: ", sentence)
                #print("optimum path: ", optimum_state)
                output = sentence[i] + ' ' + optimum_state[i]
                f.write(output)
                f.write('\n')
            f.write('\n')





print("Generating output for EN....")
emission_param_table_UNK_test , unk_list = gen_emission_param_table_UNK('EN/train', 'EN/dev.in')
tr_param_table_en = gen_transition_param_table('EN/train')
gen_state('EN/train','EN/dev.in', 'EN/dev.p3.out', emission_param_table_UNK_test, tr_param_table_en, unk_list )


print("Generating output for CN.....")
emission_param_table_UNK_test_CN , unk_list_CN = gen_emission_param_table_UNK('CN/train', 'CN/dev.in')
tr_param_table_cn = gen_transition_param_table('CN/train')
gen_state('CN/train','CN/dev.in', 'CN/dev.p3.out', emission_param_table_UNK_test_CN, tr_param_table_cn, unk_list )



print("Generating output for SG.....")
emission_param_table_UNK_test_SG , unk_list_SG = gen_emission_param_table_UNK('SG/train', 'SG/dev.in')
tr_param_table_sg = gen_transition_param_table('SG/train')
gen_state('SG/train','SG/dev.in', 'SG/dev.p3.out', emission_param_table_UNK_test_SG, tr_param_table_sg, unk_list )


print("Evaluation result for EN is: ")
os.system('python EvalScript/evalResult.py EN/dev.out EN/dev.p3.out')

print("Evaluation result for CN is: ")
os.system("python EvalScript/evalResult.py SG/dev.out SG/dev.p3.out")

print("Evaluation result for SG is: ")
os.system("python EvalScript/evalResult.py SG/dev.out SG/dev.p3.out")



