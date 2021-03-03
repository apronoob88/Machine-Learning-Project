import pandas as pd
import numpy as np

################### generate emission parameters ##############################
# function to load training set to generate emission table


def load_train(train_path):
    f = open(train_path, encoding="utf8")
    lines = []
    for line in f:
        if line != '\n':
            line = line.strip('\n').split(' ')
            lines.append(line)
    df = pd.DataFrame(lines, columns=['word', 'state'])
    return df

# function to load dev.in to generate emission table


def load_test(test_path):
    f = open(test_path, encoding='utf8')
    lines = []
    for line in f:
        if line != '\n':
            line = line.strip('\n')
            lines.append(line)
    df = pd.DataFrame(lines, columns=['word'])
    return df


def gen_emission_param_table_UNK(training_data_path, test_data_path):
    training_data = load_train(training_data_path)
    unique_word_list_training = training_data.word.unique()
    unique_state_list_training = training_data.state.unique()

    test_data = load_test(test_data_path)
    unique_word_list_test = test_data.word.unique()

    # return the list of words in test data but not in training data
    unk_list = np.setdiff1d(unique_word_list_test, unique_word_list_training)
    # non_unk_list_test = np.setdiff1d(unique_word_list_test, unk_list) # return the list of non UNK words in test data

    data = {word: (np.zeros(len(unique_state_list_training)))
            for word in unique_word_list_training}
    # add a UNK column to the table
    data["UNK"] = np.zeros(len(unique_state_list_training))

    # transform the dictionary into colums with index as name of each state(y),
    emission_count_table = pd.DataFrame(data, index=unique_state_list_training)
    # columns as each word of x, all entries are 0

    # y_count_dic stores Count(y) in a dictionary
    y_count_dic = {state: 0 for state in unique_state_list_training}
    # emission_count_table stores all Count(y -> x) in a dataframe
    emission_param_table = pd.DataFrame(data, index=unique_state_list_training)
    # emission_param_table stores all the emission parameters in a dataframe

    print("updating emission_count_table and y_count_dic")
    for index, row in training_data.iterrows():
        word = row['word']
        state = row['state']
        #print(index, word, state)
        # print(index)
        y_count_dic[state] += 1
        if word not in unk_list:
            emission_count_table[word][state] += 1

    print("updating emission_param_table")
    k = 0.5
    for index, row in training_data.iterrows():
        word = row['word']
        state = row['state']
        # print(index)
        if word not in unk_list:
            emission_param_table[word][state] = emission_count_table[word][state] / (
                y_count_dic[state] + k)
    for state in unique_state_list_training:
        # compute the UNK value for each state y
        emission_param_table['UNK'][state] = k/(y_count_dic[state] + k)

    print("unl_list is: ", unk_list)
    print("y_count_dic is: ", y_count_dic)
    return emission_param_table, unk_list


em_param_table_UNK, unk_list = gen_emission_param_table_UNK('EN/train', 'EN/dev.in')
print(unk_list)
#em_param_table_UNK


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

    df = pd.DataFrame(lines, columns=['state'])
    return df


def create_transition_count_table(df_train):
    unique_state_list = df_train.state.unique()

    data = {state: (np.zeros(len(unique_state_list)))
            for state in unique_state_list}
    # there will one extra STOP row and a START column
    transition_count_table = pd.DataFrame(data, index=unique_state_list)
    transition_count_table = transition_count_table.drop(
        'STOP')  # drop the extra STOP column
    transition_count_table = transition_count_table.drop(
        columns=['START'])  # drop the extra START column
    return transition_count_table


def create_y_count_dic(df_train):
    unique_state_list = df_train.state.unique()
    y_count_dic = {state: 0 for state in unique_state_list}
    # remove the extra STOP state since we are inclusing count(STOP) when computing the transition paraameter
    y_count_dic.pop('STOP', None)
    return y_count_dic


def gen_transition_param_table(train_path):
    input_data = load_train_transition(train_path)
    #transition_count_table,  y_count_dic = create_TransitionCountTable_and_YCountDic(input_data)
    transition_count_table = create_transition_count_table(input_data)
    # create a empty transition_param_table. Rows and columns of transition_param_table same as transition_count_table
    transition_param_table = transition_count_table.copy(deep=True)
    y_count_dic = create_y_count_dic(input_data)
    print('Generating transition parameter table')
    # len(input_data) -1 coz we iterating from y_i = 0 to y_i = n-1
    for i in range(len(input_data) - 1):
        y_i = input_data['state'][i]
        # y_i_p1 stands for yi_+1 (y_i plus 1)
        y_i_p1 = input_data['state'][i+1]
        if y_i != 'STOP':                        # we do not count the transition from STOP to some other state
            transition_count_table[y_i_p1][y_i] += 1
            y_count_dic[y_i] += 1

    cols_list = transition_count_table.columns.values.tolist()
    index_list = transition_count_table.index.values.tolist()
    for index in index_list:
        for col in cols_list:
            # a(y_i, y_i+1) = Count(y_i, y_i+1) / Count(y_i)
            transition_param_table[col][index] = transition_count_table[col][index] / \
                y_count_dic[index]
    return transition_param_table


tr_param_table = gen_transition_param_table('EN/train')
tr_param_table


# load dev.in, identify the sentences. store each sentence as a list
# return a nested list sentence_list stroing all sentences (list)
def get_sentence_list(test_path):
    f = open(test_path, encoding="utf8")
    sentence_list = []  # a list of all sentences
    sentence = []   # a list storing 1 sentence
    for word in f:
        if word != '\n':
            word = word.strip('\n')
            sentence.append(word)

        else:
            sentence_list.append(sentence)
            sentence = []
    return sentence_list


sentence_list = get_sentence_list('EN/dev.in')
# for i in range(5):
#     print('##########################################')
#     display(sentence_list[i])


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


state_list = get_state_list('EN/train')
state_list

sentence = sentence_list[0]
sentence

# create the 2-D pi(j,u) table, initialize all values as 0 at beginning
# inputs
# sentence: a list storing all words in a sentence as its element
# state_list : a list storing all the states except START and STOP


def create_pi_table(sentence, state_list):
    # copy the sentence, so that insert will not affect the actual sentence
    sentence_copy = sentence.copy()
    state_list_copy = state_list.copy()

    s = (len(state_list_copy), len(sentence_copy))
    #table = np.zeros(s)
    # print(table)
   # df = pd.DataFrame(np.random.randn(3, 3), columns=list('ABC'), index=[1, 2, 3])

    #sentence_copy.insert(0, 'START')
    # sentence.append('STOP')
    # print(sentence_copy)
    # state_list.insert(0,'START')
    # state_list.append('STOP')
    #data = {word:(np.zeros(len(state_list))) for word in sentence_copy}

    #pi_table = pd.DataFrame(data, index = state_list)
    pi_table = pd.DataFrame(
        np.zeros(s), columns=sentence_copy, index=state_list_copy)
    return pi_table
#state_list = ["O","B-neutral", "I-neutral", "B-positive", "I-positive", "B-negative", "I-negative"]
# pi_table = create_pi_table(sentence,state_list)
# pi_table


def viterbi(em_param_table, tr_param_table, unk_list, sentence, state_list):
    # create the 2-D pi(j,u) table, initialize all values as 0 at beginning
    # sentence: a list storing all words in a sentence as its element
    # state_list : a list storing all the states except START and STOP

    s = (len(state_list), len(sentence))
    pi_table = pd.DataFrame(np.zeros(s), columns=sentence, index=state_list)
   # display(pi_table)

    start_node = 1  # Add a START node in pi table, pi(0,START)
    stop_node = 0  # Add a STOP node in the pi table, pi(n+1,STOP)

    sentence = pi_table.columns.values.tolist().copy()
    states = pi_table.index.values.tolist()

    # replace the column name with position number ie. i = 0, ...n
    index_column = [i for i in range(len(sentence))]
    print(index_column)

    pi_table.columns = index_column
   # display(pi_table)
    # rint("forward")
#     start = time.time()
    # Initialize the nodes at position 1
    for state, row in pi_table[[0]].iterrows():
        pi_table[0][state] = start_node * tr_param_table[state]['START'] * \
            get_emission_parameter_UNK(
                em_param_table, unk_list, sentence[0], state)
   # display(pi_table)
#     # forward moving

    # this represents pi(i,u) for all state u at position i
    temp_result = []  # a temp table storing the pi values at each position
    temp2_result = []
    for i in range(0, len(sentence) - 1):
        for state_next, row_next in pi_table[[i+1]].iterrows():
           # print("#####################")
            for state, row in pi_table[[i]].iterrows():

                value = pi_table[i][state] * tr_param_table[state_next][state] * \
                    get_emission_parameter_UNK(
                        em_param_table, unk_list, sentence[i+1], state_next)

               #print('state: ', state)
               #print('word: ',sentence[i] )
                #alue = pi_table[i][state]

                temp_result.append(value)
            # print(max(temp_result))
            #print(sentence[i+1],state_next, i+1)
            pi_table[i+1][state_next] = max(temp_result)
            temp_result = []
     ######################PART 5##############
    for i in range(0, len(sentence) - 2):

        for state_next, row_next in pi_table[[i+2]].iterrows():
            for state, row in pi_table[[i]].iterrows():
                value2 = pi_table[i][state] * tr_param_table[state_next][state] * \
                    get_emission_parameter_UNK(
                        em_param_table, unk_list, sentence[i+2], state_next)

                temp2_result.append(value2)

            pi_table[i+2][state_next] = max(temp2_result)
            temp2_result = []

     ########################################################

# display(pi_table)

    # from last position to STOP node
    n = len(sentence) - 1
    for state, row in pi_table[[n]].iterrows():
        # temp_result.append(row[sentence[i]])

        #value = pi_table[sentence[i]][state] * tr_param_table[state_next][state] * em_param_table[sentence[i+1]][state_next]
        value = pi_table[n][state] * tr_param_table['STOP'][state]

        temp_result.append(value)
        # temp2_result.append(value)
    stop_node = max(temp_result, temp2_result)
    temp_result = []
    temp2_result = []
    # rint(stop_node)
#     end = time.time()
#     print(end - start)

#     print("backward")
#     start = time.time()
#     # back ward tracking
    # display(pi_table)

    optimum_path_np = []

    # from STOP to n

    temp_arr = np.array([])

    for state, row in pi_table.iterrows():
        pi_value = row[n]
        value = pi_value * tr_param_table['STOP'][state]
        temp_arr = np.append(temp_arr, value)
    max_y_index = np.argmax(temp_arr)
    arg_max_y_np = states[max_y_index]
    optimum_path_np.insert(0, arg_max_y_np)
    #temp_arr = np.array([])
    #print('np argmax: ', arg_max_y_np)
    # print(optimum_path_np)

    # print(states)
    # from n-1 to 0
    for i in range(len(sentence)-2, -1, -1):
        temp_arr = np.array([])
        for state, row in pi_table.iterrows():
            pi_value = row[i]
            # pi (i, u) * a (u, y*_i+1)
            value = pi_value * tr_param_table[optimum_path_np[0]][state]
            temp_arr = np.append(temp_arr, value)
        # print(temp_arr)
        max_y_index = np.argmax(temp_arr)
        #print(temp_arr, max_y_index, states[max_y_index])

        # print(states[max_y_index])
        arg_max_y_np = states[max_y_index]
        optimum_path_np.insert(0, arg_max_y_np)


#     optimum_path = []
#     temp_table = pi_table[[n]]
#     #print(temp_table)
#     for state, row in temp_table.iterrows():

#         pi_value = row[n]
#         temp_table[n][state] = pi_value * tr_param_table['STOP'][state]
#     arg_max_y = pi_table.idxmax(axis = 0)
#     #print("df argmax:", arg_max_y)
#     optimum_path.insert(0,arg_max_y[n])
#     #print(optimum_path)


#     # from n-1 to 0
#     for i in range (len(sentence)-2, -1,-1):
#         temp_table = pi_table[[i]]
#         for state, row in temp_table.iterrows():
#             pi_value = row[i]
#             temp_table[i][state] = pi_value * tr_param_table[optimum_path[0]][state]   # pi (i, u) * a (u, y*_i+1)

#         arg_max_y = pi_table.idxmax(axis = 0)
#         print('##############################')
#         print(temp_table[i],arg_max_y[i])
#         #print(arg_max_y[i])
#         optimum_path.insert(0,arg_max_y[i])
#        #print(arg_max_y)
# #     end = time.time()
# #     print(end - start)
#     #display(pi_table)
    return optimum_path_np


optimum_path_np = viterbi(em_param_table_UNK, tr_param_table, unk_list, sentence, state_list)
#optimum_path_np


def gen_state(train_path, input_path, output_path, em_param_table, tr_param_table, unk_list):
    sentence_list = get_sentence_list(input_path)
    state_list = get_state_list(train_path)
    # viterbi(em_param_table, tr_param_table, unk_list, sentence, state_list):
    with open(output_path, 'w', encoding="utf8") as f:
        # for sentence in sentence_list:
        print("No. of sentence: ", len(sentence_list))
        for i in range(len(sentence_list)):
            sentence = sentence_list[i]
            print(i)
            #pi_table = create_pi_table(sentence, state_list)
            optimum_state = viterbi(
                em_param_table, tr_param_table, unk_list, sentence, state_list)
            for i in range(len(sentence)):
                #print("sentence: ", sentence)
                #print("optimum path: ", optimum_state)
                output = sentence[i] + ' ' + optimum_state[i]
                f.write(output)
                f.write('\n')
            f.write('\n')
            # f.write(output)


gen_state('EN/train', 'EN/dev.in', 'EN/dev.p5.out',em_param_table_UNK, tr_param_table, unk_list)




em_param_table_UNK_test, unk_list = gen_emission_param_table_UNK('EN/train', 'EN/test.in')
gen_state('EN/train', 'EN/test.in', 'EN/test.p5.out',em_param_table_UNK, tr_param_table, unk_list)

os.system('python EvalScript/evalResult.py EN/dev.out EN/dev.p5.out')