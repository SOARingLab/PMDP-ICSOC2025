import numpy as np
import os
from collections import Counter
from tqdm import tqdm

def read_matrix(file_path):
    """
    Reads a matrix from a given file path.

    Args:
    file_path (str): The path to the matrix file.

    Returns:
    np.ndarray: The matrix read from the file.
    """
    try:
        matrix = np.loadtxt(file_path)
        return matrix
    except Exception as e:
        print(f"An error occurred while reading the file {file_path}: {e}")
        return None



def round_matrix_blocks(matrix_blocks, decimals=1):
    """
    Round each matrix block to 1 decimal place according to normal rounding rules.

    Args:
    matrix_blocks (list of np.ndarray): List of matrix blocks.
    decimals (int): Number of decimal places to be kept (default 1).

    Returns:
    list of np.ndarray: List of matrix blocks after rounding.
    """
    rounded_blocks = []
    for block in matrix_blocks:
        rounded_block = np.round(block, decimals=decimals)
        rounded_blocks.append(rounded_block)
    return rounded_blocks

def split_matrix(matrix, n, decimal=1):
    """
    Splits a matrix into n blocks along the columns, with the remainder added to the last block.

    Args:
    matrix (np.ndarray): The matrix to be split.
    n (int): The number of blocks to split the matrix into.

    Returns:
    list of np.ndarray: A list of n blocks.
    """

    
    # Get the number of columns in the matrix
    num_cols = matrix.shape[1]
    
    # Calculate the size of each block
    block_size = num_cols // n
    remainder = num_cols % n
    
    blocks = []
    start_col = 0
    
    for i in range(n):
        if i == n - 1:
            # Add the remainder to the last block
            end_col = num_cols
        else:
            end_col = start_col + block_size
        
        blocks.append(matrix[:, start_col:end_col])
        start_col = end_col
    
    matrix_blocks_rounded = round_matrix_blocks(blocks, decimals=decimal)
    

    return matrix_blocks_rounded


def get_unique_values(matrix_blocks):
    """
    Get all unique values from the list of matrix blocks.

    Args:
    matrix_blocks (list of np.ndarray): The list of matrix blocks.

    Returns:
    np.ndarray: An array of unique values.
    """
    # Concatenate all blocks into a single array
    concatenated_matrix = np.hstack(matrix_blocks)
    
    # Get unique values
    unique_values = np.unique(concatenated_matrix)
    
    return unique_values

############### handle the uncertainty sampling ####################


def calculate_frequency(value_list):
    """
    Calculate the frequency of occurrence of each value group.

    Parameters:
    - value_list (list of list): the value list of the variable, each element is a value group (list).

    Return:
    - frequency_dict (dict): value groups (tuple) and their corresponding frequencies.
    """
    tuple_list = [tuple(v) for v in value_list]
    frequency_dict = Counter(tuple_list)
    return frequency_dict

def calculate_probabilities(frequency_dict):
    """
    Compute sampling probabilities from a frequency dictionary.

    Parameters:
    - frequency_dict (dict): tuple of values and their corresponding frequencies.

    Returns:
    - probabilities (list): list of sampling probabilities corresponding to the order of the value groups.
    - unique_values (list of list): list of value groups.
    """
    total = sum(frequency_dict.values())
    unique_values = [list(k) for k in frequency_dict.keys()]
    probabilities = [count / total for count in frequency_dict.values()]
    return probabilities, unique_values

def probability_sampling(probabilities, unique_values, sample_size=1, replace=True):
    """
    Sampling based on precomputed probabilities.

    Parameters:
    - probabilities (list): a list of sampling probabilities corresponding to the order of the value groups.
    - unique_values (list of list): a list of value groups.
    - sample_size (int): the number of samples.
    - replace (bool): whether to sample with replacement.

    Returns:
    - sampled_values (list of list): a list of value groups obtained by sampling.
    """
    sampled_indices = np.random.choice(len(unique_values), size=sample_size, replace=replace, p=probabilities)
    sampled_values = [unique_values[i] for i in sampled_indices]
    return sampled_values

def remove_duplicate_sublists(list_of_lists):
    """
    Remove duplicate sublists from a list, preserving the original order.

    Parameters:
    - list_of_lists (list of list): the main list containing the sublists.

    Returns:
    - unique_list (list of list): the new list with duplicate sublists removed.
    """
    seen = set()
    unique_list = []
    for sublist in list_of_lists:
        # Convert sublists to tuples for hashing
        tpl = tuple(sublist)
        if tpl not in seen:
            seen.add(tpl)
            unique_list.append(sublist)
    return unique_list


def compute_frequency_probabilities_domain(n, k = 1160, mode_str = 'Reading Dataset for Each Activity: ', decimal = 0.25):
    """
    Transpose each block in rt and tp matrices and calculate the frequency and probability of each concrete_service.
    This function will be called twice, first time is to get the domain of the abstract service, second time is to get the frequency and probability of the abstract service

    Parameters:
    - n (int): The number of blocks to split. that is the candidate number of abstract services
    - k (int): The number of concrete services, that is the number of records (part or all) in the dataset will be used in this calling
    - decimal (int): The number of decimal places to round to.

    Returns:
    - frequency_prob_list (list of list of tuples): (probability list, unique value list) for each block in each matrix type.
    - frequency_prob_list[0]: frequency and probability list of rt matrix.
    - frequency_prob_list[1]: frequency and probability list of tp matrix.
    """
    base_path = os.path.dirname(__file__)
    rt_matrix_path = os.path.join(base_path, 'rtMatrix.txt')
    tp_matrix_path = os.path.join(base_path, 'tpMatrix.txt')

    rt_matrix = read_matrix(rt_matrix_path)[:, :k]
    tp_matrix = read_matrix(tp_matrix_path)[:, :k]

    # e.g.,  original is 339x k, that will be splited as a matrix into n blocks along the columns, with the remainder added to the last block.
    rt_matrix_round_blocks = split_matrix(rt_matrix, n, decimal=decimal)
    tp_matrix_round_blocks = split_matrix(tp_matrix, n, decimal=decimal)

    frequency_prob_dict = {}
    domain_frequency_prob_dict = {}

    rt_tp_matrix_domain_list = []
    rt_matrix_domain_list = []
    tp_matrix_domain_list = []  

    processed_transpose_rt_matrix_round_blocks_number = []
    # 'Ui' is the uncontrollable variable of abstract service, '~j' is the concrete service
    for i, item in enumerate(tqdm(rt_matrix_round_blocks, desc=mode_str)):
        frequency_prob_dict[f'U{i+1}'] = {}
        domain_frequency_prob_dict[f'U{i+1}_rt'] = {}
        domain_frequency_prob_dict[f'U{i+1}_tp'] = {}

        # original is 339x194, after transpose is 194x339
        transpose_rt_matrix_round_blocks = rt_matrix_round_blocks[i].T
        transpose_tp_matrix_round_blocks = tp_matrix_round_blocks[i].T

        processed_transpose_rt_matrix_round_blocks_number.append([])
    
        
        block_freq_probs = {}
        # it converts from ‘339x194’ to ‘194x339’, that is, each row is a concrete service (e.g., having 194 concrete service), the columns are the records (e.g., each record has 339 records)
        # e.g., 'transpose_rt_matrix_round_blocks' is 194x339, each row is a concrete service, the columns are the records
        var_index = 0
        for j in range(len(transpose_rt_matrix_round_blocks)):
            
            combine_record = []
            rt_recrod = []
            tp_record = []
            for k in range(len(transpose_rt_matrix_round_blocks[j])):
                rt_value = transpose_rt_matrix_round_blocks[j][k]
                tp_value = transpose_tp_matrix_round_blocks[j][k]
                # e.g., 'rt_matrix_round_blocks[i][j][k]' is the value of the concrete service
                if not np.isnan(rt_value) and not np.isnan(tp_value) and rt_value != -1 and tp_value != -1:
                    if isinstance(rt_value, (int, float)) and isinstance(tp_value, (int, float)):
                        
                        
                        ### The statisc information of 10 Abstract Service CSSC-MDP Example code ###
                        #{'U1': 1.778894472361809, 'U2': 1.6013400335008374, 'U3': 2.8006700167504186, 
                        # 'U4': 6.1373534338358455, 'U5': 2.4489112227805694, 'U6': 1.2529313232830819, 
                        # 'U7': 1.9698492462311556, 'U8': 2.4355108877721947, 'U9': 1.4271356783919598, 
                        # 'U10': 1.1474036850921272}

                        '''
                        if i == 0 :
                            if rt_value <= 1.778894472361809*0.85:
                                rt_value = 1.778894472361809*0.85
                            elif rt_value <= 1.778894472361809:
                                rt_value = 1.778894472361809
                            elif rt_value > 1.778894472361809:
                        '''


                        '''
                        # Reduce the variable domain according to the constraints 'TP>13', 
                        # so directly classify the value of 'TP' into two categories, 12 and 14 means TP<=13 and TP>13, respectively
                        if tp_value > 13:
                            tp_value = 14
                        else:
                            tp_value = 12

                        '''
                        combine_record.append([rt_value, tp_value])
                        rt_recrod.append([rt_value])
                        tp_record.append([tp_value])

        
            if len(combine_record) == 0:
                continue
            frequency_dict = calculate_frequency(combine_record)
            probabilities, unique_values = calculate_probabilities(frequency_dict)

            rt_frequency_dict = calculate_frequency(rt_recrod)
            rt_probabilities, rt_unique_values = calculate_probabilities(rt_frequency_dict)

            tp_frequency_dict = calculate_frequency(tp_record)
            tp_probabilities, tp_unique_values = calculate_probabilities(tp_frequency_dict)
            
            #print(len(probabilities), len(unique_values))

            # This diction 0 dimension is the abstract service, 1 dimension is the concrete service, and the value 'frequency_prob_dict[f'rt_tp_{i+1}'][f'~{j+1}']' is the frequency and probability about the concrete service
            frequency_prob_dict[f'U{i+1}'][f'~{var_index+1}'] = [probabilities, unique_values]

            domain_frequency_prob_dict[f'U{i+1}_rt'][f'~{var_index+1}'] = [rt_probabilities, rt_unique_values]
            domain_frequency_prob_dict[f'U{i+1}_tp'][f'~{var_index+1}'] = [tp_probabilities, tp_unique_values]

            # don't use 'j' because some concrete services are been removed when the value is -1
            var_index += 1
        processed_transpose_rt_matrix_round_blocks_number[i] = var_index


        domain = []
        #### get domain of the abstract service ####    
        for key, concrete_service in frequency_prob_dict[f'U{i+1}'].items():
            domain.extend(concrete_service[1])
        rt_tp_matrix_domain_list.append(remove_duplicate_sublists(domain))

        rt_domain = []
        tp_domain = []
        for key, concrete_service in domain_frequency_prob_dict[f'U{i+1}_rt'].items():
            rt_domain.extend(concrete_service[1])
        for key, concrete_service in domain_frequency_prob_dict[f'U{i+1}_tp'].items():
            tp_domain.extend(concrete_service[1])

        rt_matrix_domain_list.append(remove_duplicate_sublists(rt_domain))
        tp_matrix_domain_list.append(remove_duplicate_sublists(tp_domain))



    ##### If need to write the domain of the abstract service to the file ####
    #rt_matrix_domain_path = os.path.join(base_path, f'CSSC_rt_AbstractServices_{n}_Eech_has_about_{len(rt_matrix_round_blocks[0][0])}_ConcreteServices.txt')
    #with open(rt_matrix_domain_path, 'w') as file:
    activity_variable_doamin_string_list = []
    
    # # 'Ci' is the controllable variable of abstract service
    for i, item in enumerate(rt_matrix_round_blocks):
        # original is 339x194, after transpose is 194x339
        transpose_rt_matrix_round_blocks = rt_matrix_round_blocks[i].T

        tmp_domain_str ='C={' + f'C{i+1}:' + '{'
        for j in range(processed_transpose_rt_matrix_round_blocks_number[i]):
            tmp_domain_str += f'[{j+1}],'
        tmp_domain_str = tmp_domain_str[:-1] + '}}'
        tmp_domain_str = tmp_domain_str + '||'


        '''
        tmp_domain_str += 'U={' + f'U{i+1}:' + '{'
        for _, values in enumerate(rt_tp_matrix_domain_list[i]):
            tmp_domain_str += f'({values},'+ 'unknown),'
        tmp_domain_str = tmp_domain_str[:-1] + '}}'

        activity_variable_doamin_string_list.append(tmp_domain_str)
        '''

        tmp_domain_str += 'U={' + f'rt{i+1}:' + '{'
        for _, values in enumerate(rt_matrix_domain_list[i]):
            tmp_domain_str += f'({values},'+ 'unknown),'
        tmp_domain_str = tmp_domain_str[:-1] + '}|'

        tmp_domain_str += f'tp{i+1}:' + '{'
        for _, values in enumerate(tp_matrix_domain_list[i]):
            tmp_domain_str += f'({values},'+ 'unknown),'
        tmp_domain_str = tmp_domain_str[:-1] + '}}'

        activity_variable_doamin_string_list.append(tmp_domain_str)
        
        ##### If need to write the domain of the abstract service to the file ####
        #file.write(f'The rt{i+1} domain of Concrete Services for Abstract Service {i+1}: \n{tmp_domain_str}\n')


    return frequency_prob_dict, activity_variable_doamin_string_list



def new_compute_frequency_probabilities_domain(n, k = 1160, mode_str = 'Reading Dataset for Each Activity: ', decimal = 1):
    """
    Transpose each block in the rt and tp matrices and calculate the frequency and probability of each concrete_service.

    Parameters:
    - n (int): number of blocks to split.
    - decimal (int): number of decimal places to round to.

    Returns:
    - frequency_prob_list (list of list of tuples): (probability list, unique value list) for each block in each matrix type.
    - frequency_prob_list[0]: frequency and probability list for the rt matrix.
    - frequency_prob_list[1]: frequency and probability list for the tp matrix.
    """
    base_path = os.path.dirname(__file__)
    rt_matrix_path = os.path.join(base_path, 'rtMatrix.txt')
    tp_matrix_path = os.path.join(base_path, 'tpMatrix.txt')

    rt_matrix = read_matrix(rt_matrix_path)[:, :k]
    tp_matrix = read_matrix(tp_matrix_path)[:, :k]

    # e.g.,  original is 339x1160, that will be splited as a matrix into n blocks along the columns, with the remainder added to the last block.
    rt_matrix_round_blocks = split_matrix(rt_matrix, n, decimal=decimal)
    tp_matrix_round_blocks = split_matrix(tp_matrix, n, decimal=decimal)

    frequency_prob_dict = {}
    rt_tp_matrix_domain_list = []  

    processed_transpose_rt_matrix_round_blocks_number = []
    # 'Ui' is the uncontrollable variable of abstract service, '~j' is the concrete service
    for i, item in enumerate(tqdm(rt_matrix_round_blocks, desc=mode_str)):
        frequency_prob_dict[f'U{i+1}'] = {}
        # original is 339x194, after transpose is 194x339
        transpose_rt_matrix_round_blocks = rt_matrix_round_blocks[i].T
        transpose_tp_matrix_round_blocks = tp_matrix_round_blocks[i].T

        processed_transpose_rt_matrix_round_blocks_number.append([])
    
        
        block_freq_probs = {}
        # it converts from ‘339x194’ to ‘194x339’, that is, each row is a concrete service (e.g., having 194 concrete service), the columns are the records (e.g., each record has 339 records)
        # e.g., 'transpose_rt_matrix_round_blocks' is 194x339, each row is a concrete service, the columns are the records
        var_index = 0
        for j in range(len(transpose_rt_matrix_round_blocks)):
            
            combine_record = []
            for k in range(len(transpose_rt_matrix_round_blocks[j])):
                rt_value = transpose_rt_matrix_round_blocks[j][k]
                tp_value = transpose_tp_matrix_round_blocks[j][k]
                # e.g., 'rt_matrix_round_blocks[i][j][k]' is the value of the concrete service
                if not np.isnan(rt_value) and not np.isnan(tp_value) and rt_value != -1 and tp_value != -1:
                    if isinstance(rt_value, (int, float)) and isinstance(tp_value, (int, float)):
                        combine_record.append([rt_value, tp_value])
        
            if len(combine_record) == 0:
                continue
            frequency_dict = calculate_frequency(combine_record)
            probabilities, unique_values = calculate_probabilities(frequency_dict)
            
            #print(len(probabilities), len(unique_values))

            # This diction 0 dimension is the abstract service, 1 dimension is the concrete service, and the value 'frequency_prob_dict[f'rt_tp_{i+1}'][f'~{j+1}']' is the frequency and probability about the concrete service
            frequency_prob_dict[f'U{i+1}'][f'~{var_index+1}'] = [probabilities, unique_values]
            # don't use 'j' because some concrete services are been removed when the value is -1
            var_index += 1
        processed_transpose_rt_matrix_round_blocks_number[i] = var_index


        domain = []
        #### get domain of the abstract service ####    
        for key, concrete_service in frequency_prob_dict[f'U{i+1}'].items():
            domain.extend(concrete_service[1])
        rt_tp_matrix_domain_list.append(remove_duplicate_sublists(domain))

    ##### If need to write the domain of the abstract service to the file ####
    #rt_matrix_domain_path = os.path.join(base_path, f'CSSC_rt_AbstractServices_{n}_Eech_has_about_{len(rt_matrix_round_blocks[0][0])}_ConcreteServices.txt')
    #with open(rt_matrix_domain_path, 'w') as file:
    activity_variable_doamin_string_list = []
    
    # # 'Ci' is the controllable variable of abstract service
    for i, item in enumerate(rt_matrix_round_blocks):
        # original is 339x194, after transpose is 194x339
        transpose_rt_matrix_round_blocks = rt_matrix_round_blocks[i].T

        tmp_domain_str ='C={' + f'C{i+1}:' + '{'
        for j in range(processed_transpose_rt_matrix_round_blocks_number[i]):
            tmp_domain_str += f'[{j+1}],'
        tmp_domain_str = tmp_domain_str[:-1] + '}}'
        tmp_domain_str = tmp_domain_str + '||'
            
        tmp_domain_str += 'U={' + f'U{i+1}:' + '{'
        for _, values in enumerate(rt_tp_matrix_domain_list[i]):
            tmp_domain_str += f'({values},'+ 'unknown),'
        tmp_domain_str = tmp_domain_str[:-1] + '}}'

        activity_variable_doamin_string_list.append(tmp_domain_str)
        
        ##### If need to write the domain of the abstract service to the file ####
        #file.write(f'The rt{i+1} domain of Concrete Services for Abstract Service {i+1}: \n{tmp_domain_str}\n')


    return frequency_prob_dict, activity_variable_doamin_string_list


############### handle the uncertainty sampling ####################


if __name__ == "__main__":

    # Example usage:
    n = 30  # Number of blocks, that is the number of task/activity in one BPMN model
    decimal = 1  # Number of decimal places to round to, the CSSC-MDP algorithm uses 2 decimal places
    var_name = 'rt'


    base_path = os.path.dirname(__file__)
    rt_matrix_path = os.path.join(base_path, 'rtMatrix.txt')
    tp_matrix_path = os.path.join(base_path, 'tpMatrix.txt')

    rt_matrix = read_matrix(rt_matrix_path)
    tp_matrix = read_matrix(tp_matrix_path)


    # Example usage:
    # Assuming 'matrix' is already defined and is a numpy array

    rt_matrix_round_blocks = split_matrix(rt_matrix, n, decimal=decimal)
    tp_matrix_round_blocks = split_matrix(tp_matrix, n, decimal=decimal)

    
    #### CSCC-MDP the example of mean data of state division ####
    abstract_service_values_median = []
    for i, abstract_services in enumerate(rt_matrix_round_blocks):
        abstract_services = abstract_services.T
        abstract_service_values_median.append([])
        for concrete_services in abstract_services:
            # each abstract service has 194 concrete services
            # each concrete service has 339 records
            abstract_service_values_median[i].extend([np.median(concrete_services, axis=0)])
    
    abstract_service_mean_dict = {}
    for i, value in enumerate(abstract_service_values_median):
        abstract_service_mean_dict[f'as_{var_name}_{i}'] = np.mean(value, axis=0)
    #### CSCC-MDP the example of mean data of state division ####


    rp_matrix_domain_list = []
    # Get unique values for each block in tp_matrix_blocks
    for i, block in enumerate(tp_matrix_round_blocks):
        rp_matrix_domain_list.append(get_unique_values(block))
        #print(f"Unique values in tp_matrix_blocks[{i}]: {rp_matrix_domain}")
        print(f"Number of unique values in rp_matrix_blocks[{i}]: {len(rp_matrix_domain_list[i])}")

    


    # Print the shape of the matrices to verify
    if rt_matrix is not None:
        print(f"rtMatrix shape: {rt_matrix.shape}")
    if tp_matrix is not None:
        print(f"tpMatrix shape: {tp_matrix.shape}")