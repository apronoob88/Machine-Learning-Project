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

    
    print("unl_list is: ",unk_list)
    print("y_count_dic is: ", y_count_dic)
    return emission_param_table, unk_list


# In[3]:


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
    #transition_count_table,  y_count_dic = create_TransitionCountTable_and_YCountDic(input_data)
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

sentence_list = get_sentence_list('SG/dev.in')
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
    
    start_node = 1
    end_node = np.array([0,0,0])
    # len(sentence)-1 becault first layer is actually a scalar
    s = (len(state_list),len(sentence)-1,3)
    
    # a 3d matrix with each row as the states, column as the word, 
    # each element is a 1x3 array to store top 3 positions 
    # pi_table.shape[0] -> rows (states)
    # pi_table.shape[1] -> columns (words)
    # pi_table.shape[2] -> elements (top 3 paths)
    pi_table = np.zeros(s)
    
    # Moving forward

    # from start to position 1 
    layer_1 = np.array([])
    # find values for the nodes at position 1 (first word)
    for i in range(len(state_list)):
        value = start_node * tr_param_table[state_list[i]]['START'] * get_emission_parameter_UNK(em_param_table,unk_list, sentence[0], state_list[i])
        layer_1 = np.append(layer_1,value)

    # find top 3 values for the nodes at position 2 (second word)
    # note the first column of pi_table corresponds to the second word since for the first word, it only has top 1 path
    
    for next_layer_index in range(len(state_list)):
        temp_result = np.array([])
        for curr_layer_index in range(len(state_list)):
            value = layer_1[curr_layer_index] * tr_param_table[state_list[next_layer_index]][state_list[curr_layer_index]] * get_emission_parameter_UNK(em_param_table,unk_list, sentence[1], state_list[next_layer_index])
            temp_result = np.append(temp_result,value)

        temp_result = temp_result[np.argsort(temp_result)[-3:]]
        temp_result = np.sort(temp_result)[::-1]
        pi_table[next_layer_index][0] = temp_result

    
    for i in range(pi_table.shape[1]-2):
        word_curr = sentence[i+1]
        word_next = sentence[i+2]
        
        for next_layer_index in range(len(state_list)):
            temp_result = np.array([])
            for curr_layer_index in range(len(state_list)):

                #for each 3 pi scores for each node
                for j in range(pi_table.shape[2]):
                    pi_value = pi_table[curr_layer_index][i][j]
                    value = pi_value * tr_param_table[state_list[next_layer_index]][state_list[curr_layer_index]] * get_emission_parameter_UNK(em_param_table,unk_list, word_next, state_list[next_layer_index])
                    temp_result = np.append(temp_result,value)

            temp_result = temp_result[np.argsort(temp_result)[-3:]]
            temp_result = np.sort(temp_result)[::-1]

            pi_table[next_layer_index][i+1] = temp_result

    
    
    # from n node to STOP
    temp_result = temp_result = np.array([])

    for curr_layer_index in range(len(state_list)):
        for j in range(pi_table.shape[2]):
            pi_value = pi_table[curr_layer_index][pi_table.shape[1]-1][j]
            value = pi_value * tr_param_table['STOP'][state_list[curr_layer_index]]
            temp_result = np.append(temp_result, value)
    
    temp_result = temp_result[np.argsort(temp_result)[-3:]]
    temp_result = np.sort(temp_result)[::-1]
    stop_node = temp_result.copy()
    
    
    optimum_path = []  # this is actually the 3rd best path we are keeping track of 
    
    # Moving backward
    
    #the final value at STOP for the third optimal path
    path_3_value = np.min(stop_node)

    value_path = 0
    
    found_path = False
    for curr_layer_index in range(len(state_list)):
        if found_path:
            break
        for j in range(pi_table.shape[2]):
            pi_value = pi_table[curr_layer_index][pi_table.shape[1]-1][j]
            value = pi_value * tr_param_table['STOP'][state_list[curr_layer_index]]
            if value == path_3_value:
                value_path = pi_value
                node = state_list[curr_layer_index]
                optimum_path.insert(0,node)
                found_path = True
                break

    # moving back from n-1 to 3rd word
    for i in range(pi_table.shape[1]-2, -1, -1):
        word_next = sentence[i+2]
        for curr_layer_index in range(len(state_list)):
            path_found_temp = False
            for j in range(pi_table.shape[2]):
                pi_value = pi_table[curr_layer_index][i][j]
                value = pi_value * tr_param_table[optimum_path[0]][state_list[curr_layer_index]] * get_emission_parameter_UNK(em_param_table,unk_list, word_next, optimum_path[0])
                if value == value_path:
                    node = state_list[curr_layer_index]
                    optimum_path.insert(0,node)
                    value_path = pi_value
                    path_found_temp = True
                    break
            if path_found_temp:
                break
    
    #from 2nd word to 1st word
    
    for i in range(len(layer_1)):
        pi_value = layer_1[i]
        value = pi_value * tr_param_table[optimum_path[0]][state_list[i]] * get_emission_parameter_UNK(em_param_table,unk_list, sentence[1], optimum_path[0])
        if value == value_path:
            node = state_list[i]
            optimum_path.insert(0,node)
            
    return optimum_path # its called optimum_path but it is actually the 3rd best path
    


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


print("Generating output for EN")
em_param_table_UNK , unk_list = gen_emission_param_table_UNK('EN/train', 'EN/dev.in')
tr_param_table = gen_transition_param_table('EN/train')
gen_state('EN/train','EN/dev.in', 'EN/dev.p4.out', em_param_table_UNK, tr_param_table, unk_list )

os.system('python EvalScript/evalResult.py EN/dev.out EN/dev.p4.out')