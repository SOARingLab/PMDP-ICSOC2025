import builtins
import datetime
import json
import math
import string
import time
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.nn import SmoothL1Loss


# Get the project root directory
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from collections import defaultdict
from tqdm import tqdm
import numpy as np
from src.pmdp.Create_PMDP_Environment_From_Graph import PMDPEnvironment
from src.pmdp.Convert_BPMN_into_Graph import parse_bpmn, Vertex as BaseVertex, Edge as BaseEdge, WeightedDirectedGraph as BaseGraph, Variable as BaseVariable, Constraint as BaseConstraint, KPI as BaseKPI
from Datasets.TravelAgencyNFPsDataset.ReadData_ReturnProbabilityDistribution import get_travelAgency_distribution
from Datasets.wsdream.Read_Dataset import compute_frequency_probabilities_domain, read_matrix, split_matrix
from src.pmdp.KPIs_Constraints_Operators_Parse_Evaluation import OPERATOR_INFO
from typing import Dict, List, Union, Set
from farmhash import FarmHash32, FarmHash64, FarmHash128
from src.utilities.PrioritizedExperienceReplay import PrioritizedReplayBuffer
from src.utilities.PrioritizedExperienceReplay_Pytorch import PrioritizedReplayBuffer as PrioritizedReplayBuffer_Pytorch
from Datasets.QWS.process_normed_QWSData_to_distribution import get_QWS_distribution
import ctypes, statistics
import copy, os, ast, re
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.ticker import MaxNLocator
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import grad
#from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

#seed = 42
#torch.manual_seed(seed)
torch_intType = torch.int64

# Create a mapping for int16 to string conversion
int16_to_string_map = {}
existing_hashes = set()
horizon_int16_to_string_map = {}
wait_int16_to_string_map = {}
None_int16_to_string_map = {}
finish_int16_to_string_map = {}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_mappings():
    """
    Clear all mapping dictionaries and collections
    """
    global int16_to_string_map
    global existing_hashes
    global horizon_int16_to_string_map
    global wait_int16_to_string_map
    global None_int16_to_string_map
    global finish_int16_to_string_map

    int16_to_string_map.clear()
    existing_hashes.clear()
    horizon_int16_to_string_map.clear()
    wait_int16_to_string_map.clear()
    None_int16_to_string_map.clear()
    finish_int16_to_string_map.clear()

learning_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
We define subclasses in PMDPEnvironment and ensure that all methods and attributes have explicit type annotations. 
This will help compilers and editors to better infer data structures and thus provide code-completion functionality.
'''
class Vertex(BaseVertex): # Vertex (i.e., node in HPST) is the extension of the BaseVertex (i.e., vertex in graph) in HPST
    pass
class Edge(BaseEdge):
    pass
class WeightedDirectedGraph(BaseGraph):
    pass
class Variable(BaseVariable):
    pass

class Constraint(BaseConstraint):
    pass
class KPI(BaseKPI):
    pass


class ConstraintState:
    def __init__(self, constraints_list, beta=0.1):
        """
        Initializes the ConstraintState.

        :param constraint_type: A dictionary mapping constraint names to their types ('relational' or 'optimization').
        :param beta: Precision parameter for optimization constraints (default is 0.1).
        """
        self.beta = beta
        self.constraints_list = constraints_list
        self.constraint_values = self.initialize_ConstraintsState_dict()
        self.cs = torch.full((len(constraints_list),), -1, dtype=torch.float32, device=learning_device)

    def initialize_ConstraintsState_dict(self):
        """
        Initializes the constraint values based on their type.

        Relational constraints are initialized to 0 (not satisfied).
        Optimization constraints are initialized to 1.

        :return: A dictionary mapping each constraint to its initial state.
        """
        ConstraintsState_dict = {}
        for constraint in self.constraints_list:
            # -1 means the constraint is not unknown, that is, the constraint have not been evaluated yet
            if '\max' not in constraint.expression and '\min' not in constraint.expression:
                ConstraintsState_dict[constraint.name]  = torch.tensor(-1, dtype=torch.float32, device=learning_device)  # Relational constraints initialized to 0 or 1
            else:
                ConstraintsState_dict[constraint.name]  = torch.tensor(-1, dtype=torch.float32, device=learning_device) # Optimization constraints initialized to 1
        return ConstraintsState_dict
    
    def reset(self, constraints_list):
        self.constraint_values = self.initialize_ConstraintsState_dict()
        self.cs = torch.full((len(constraints_list),), -1, dtype=torch.float32, device=learning_device)


    def update_CS(self):
        """
        Updates the constraint state tensor (self.cs) based on the current values in self.constraint_values.
        """
        for i, constraint in enumerate(self.constraints_list):
            self.cs[i] = self.constraint_values[constraint.name]

    def get_constraint_state(self):
        """
        获取当前约束状态
        """
        return tuple(self.cs)
    
    def __repr__(self):

        return f"ConstraintState(cs= {self.cs}, constraint_values={self.constraint_values}, beta={self.beta}, constraints_list={self.constraints_list})"

# This only use for Table-based Q-learning
class ConstraintAwareExpectedEligibilityTraces:
    def __init__(self, nS, nA):
        """
        Initialize the matrix z(cs) and counter n(cs) of the desired eligibility trace
        - nS: state space size
        - nA: action space size
        """
        self.z = defaultdict(lambda: torch.zeros((nS, nA), dtype=torch.float32, device=learning_device))  # Expected eligibility trace
        self.n_cs = defaultdict(int)  # Number of visits to each constraint state

    def update(self, constraintName, eligibility_trace):
        """
        Update the expected eligibility trace according to the formula z(cs) = z(cs) + (e_n(cs)+1 - z(cs)) / (n(cs) + 1)
        - cs: Current constraint state
        - instantaneous_eligibility: Instantaneous eligibility trace
        """
        n = self.n_cs[constraintName]
        self.z[constraintName] += (eligibility_trace - self.z[constraintName]) / (n + 1)
        self.n_cs[constraintName] += 1

def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values



# return the ActionIndice (i.e., row index) for the Q based on the current state  
def select_action(state_indice, actionSpace_row_start, actionSpace_row_end, Q, epsilon, only_greedy=False):
    if torch.rand(1).item() > epsilon or only_greedy:  # Use torch.rand
        # Greed action
        #return torch.argmax(Q[state_indice][actionSpace_row_start:actionSpace_row_end]).item() + actionSpace_row_start  # Use torch.argmax
        
        q_values = Q[state_indice][actionSpace_row_start:actionSpace_row_end]
        max_value = torch.max(q_values)
        max_indices = torch.where(q_values == max_value)[0]  # Find all indices with the max value
        selected_index = max_indices[torch.randint(len(max_indices), (1,)).item()]  # Randomly select one of the max indices
        return selected_index.item() + actionSpace_row_start
    # Exploration action
    return torch.randint(actionSpace_row_start, actionSpace_row_end, (1,)).item()  # Use torch.randint


# Q_target is a slow updating Q-function (Q_) compared to current_Q (Q), which is used to compute the TD error
def compute_td_error(reward, gamma, Q_target, current_Q, next_state, action):
    max_action = torch.argmax(Q_target[next_state])  # Use torch.argmax
    td_target = reward + gamma * Q_target[next_state][max_action]
    return td_target - current_Q[action]

def update_q_value(Q, state, action, td_error, alpha):
    Q[state][action] += alpha * td_error


# If having log file, then calculate the bounds

def calculate_max_min_from_txt(log_file_path):
    """
    Calculate the maximum and minimum values of each variable in a given .txt file.

    Parameters:
    logfile_path (str): path to the .txt file.
    It like:
    {'a': '1021', 'b': '1208', 'c': '2669', 'ee': '24', 'd': '2208'}
    {'a': '1861', 'b': '905', 'c': '757', 'ee': '24', 'd': '2208'}

    Returns:
    tuple: two dictionaries, one for the maximum value and one for the minimum value.
    """
    data = []

    # Read the file and parse the data
    with open(log_file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(ast.literal_eval(line.strip()))

    # Extract values for each variable
    variables = data[0].keys()
    max_values = {var: float('-inf') for var in variables}
    min_values = {var: float('inf') for var in variables}

    for entry in data:
        for var in variables:
            value = int(entry[var])
            if value > max_values[var]:
                max_values[var] = value
            if value < min_values[var]:
                min_values[var] = value

    return max_values, min_values


def mdp_update_dynamic_bounds(pmdp, var_name, value, bounds_list):
    # make constraints with path-aware
    if value == 'NotSelected':
        value = 0
    if var_name not in pmdp.dynamic_bounds:
        pmdp.dynamic_bounds[var_name] = (value, value)
    else:
        min_val, max_val = bounds_list
        if value < min_val:
            min_val = value
        if value > max_val:
            max_val = value
        pmdp.dynamic_bounds[var_name] = (min_val, max_val)



##### some functions for parsing variables to construct the state space for Q-function #####

# Custom hash function
def custom_hash_16bit(data: str, existing_hashes: set) -> int:
    # Use FarmHash32 to compute the initial hash value
    hash32 = FarmHash32(data)
    hash32_int16 = int(np.array(hash32 % (2**16)).astype(np.int16))

    # Check for duplicate int16 hash values
    if hash32_int16 in existing_hashes:
        # Use FarmHash64 to compute a new hash value
        hash64 = FarmHash64(data)
        hash64_int16 = int(np.array(hash64 % (2**16)).astype(np.int16))

        # Check for duplicate int16 hash values
        if hash64_int16 in existing_hashes:
            # Calculate new hash value using FarmHash128
            hash128 = FarmHash128(data)
            hash128_int16 = int(np.array(hash128 % (2**16)).astype(np.int16))

            # Combine the three hash values and convert to int16
            combined_hash_int16 = int(np.array(hash32_int16 ^ hash64_int16 ^ hash128_int16).astype(np.int16))
            if combined_hash_int16 in existing_hashes:
                raise ValueError(f"Hash collision (when initialize Q-function) detected for {data}")
            else:
                existing_hashes.add(combined_hash_int16)
                return combined_hash_int16
        else:
            existing_hashes.add(hash64_int16)
            return hash64_int16
    else:
        existing_hashes.add(hash32_int16)
        return hash32_int16
    

def custom_hash_32bit(data: str, existing_hashes: set) -> int:
    """
    生成唯一的 int32 哈希值，避免与 existing_hashes 中的值冲突。
    
    Parameters:
    - data (str): The string data to hash.
    - existing_hashes (set of int): The set of existing hash values to check for collisions.

    Returns:
    - int: The generated unique int32 hash value.

    Exceptions:
    - ValueError: If all hash functions result in collisions, an exception is raised.
    """
    # Use FarmHash32 to compute the initial hash value and convert to int32
    hash32 = FarmHash32(data)
    hash32_int32 = int(hash32 % (2**32))

    # Check for duplicate int32 hash values
    if hash32_int32 in existing_hashes:
        # Use FarmHash64 to compute a new hash value and convert to int32
        hash64 = FarmHash64(data)
        hash64_int32 = int(hash64 % (2**32))

        # Check for duplicate int32 hash values
        if hash64_int32 in existing_hashes:
            # Use FarmHash128 to compute a new hash value and convert to int32
            hash128 = FarmHash128(data)
            hash128_int32 = int(hash128 % (2**32))

            # Combine the three hash values and convert to int32
            combined_hash_int32 = hash32_int32 ^ hash64_int32 ^ hash128_int32
            if combined_hash_int32 in existing_hashes:
                raise ValueError(f"Hash collision detected for {data} with combined hash {combined_hash_int32}")
            else:
                existing_hashes.add(combined_hash_int32)
                return combined_hash_int32
        else:
            existing_hashes.add(hash64_int32)
            return hash64_int32
    else:
        existing_hashes.add(hash32_int32)
        return hash32_int32

# Hash function for tensor 
def string_to_hash_int64(s: str):
    hash_value = FarmHash32(s)
    # Convert unsigned integer to signed integer
    hash32_signed = ctypes.c_int32(hash_value).value
    return hash32_signed

    # Convert the hash value to a string and truncate it to an 18-digit decimal integer
    #hash_str = str(hash_value)[:18]

    # Convert back to an integer
    #truncated_hash = int(hash_str)

    # Check if it is within the range of int64
    #if -9223372036854775808 <= truncated_hash <= 9223372036854775807:
    #return truncated_hash
    #else:
    #raise ValueError("Hash value exceeds int64 range")
    
def parse_variable_domain(vertex_id: str, var_name: str, var_domain: list, var_controlType: int, domainType = 'normal') -> List:
    parsed_values = []
    # Do not sort the variable domains to keep the variable order consistent with the BPMN model, and make the tensors in the Q-table not sorted by the size of the variable values
    var_domain.sort()
    if domainType == 'normal':
        for item in var_domain:
            if var_controlType == 1:
                # var is controllable variable
                if ',' not in item:
                    # If it is not a vector (i.e. a number or a character), item will be in the form of '[0]' or '[character]'. First remove the symbols: [ and ] and then concatenate it with the variable name to form a string.
                    item = item.strip('[]')

                # After processing item, if it is a vector, item will be in the form of '[0,1,2]', directly concatenate item with variable name to form a string
                # Add variable name to each value and convert to string
                full_item = f"{var_name}={item}"
                if full_item in int16_to_string_map.values():
                    # If it already exists, directly retrieve the hash value
                    hashed_value = [key for key, value in int16_to_string_map.items() if value == full_item][0]
                else:
                    # Use hash to map string to unique integer
                    hashed_value = custom_hash_32bit(full_item, existing_hashes)

                # Store the mapping from hash value to original string in the dictionary
                int16_to_string_map[hashed_value] = full_item
                # Add the hash value to the list
                parsed_values.append(hashed_value)

            elif var_controlType == 2:
                # var is uncontrollable variable
                # the item will be like [[value], probability]
                # so the item[0] can get the value, and item[1] get the probability
                item = item[0]
            
                if ',' not in item:
                    # If it is not a vector (i.e. a number or a character), item will be in the form of '[0]' or '[character]'. First remove the symbols: [ and ] and then concatenate it with the variable name to form a string.
                    item = item.strip('[]')

                # If it is a vector, item will be in the form of '[0,1,2]', directly concatenate item with variable name to form a string
                # Add variable name to each value and convert to string
                full_item = f"{var_name}={item}"
                if full_item in int16_to_string_map.values():
                    # If it already exists, directly retrieve the hash value
                    hashed_value = [key for key, value in int16_to_string_map.items() if value == full_item][0]
                else:
                    # Use hash to map string to unique integer
                    hashed_value = custom_hash_32bit(full_item, existing_hashes)

                # Store the mapping from hash value to original string in the dictionary
                int16_to_string_map[hashed_value] = full_item
                # Add the hash value to the list
                parsed_values.append(hashed_value)
            else:
                raise ValueError(f"The control type of {var_name} is not defined or correct, it should be 1 or 2")
            
    # adding 'None' and 'wait' as an value for the variable, and use f"{vertex_id}=None" or f"{vertex_id}=wait" as the state value
    elif domainType == 'wait_none':
        # for each var add 'None' as a state for the var is not assigned a value
        full_item = f"{vertex_id}=None"  
        if full_item in None_int16_to_string_map.values():
            # If the same string already exists, directly extract the hash value
            hashed_value = [key for key, value in None_int16_to_string_map.items() if value == full_item][0]
        else:
            hashed_value = custom_hash_32bit(full_item, existing_hashes)

        None_int16_to_string_map[hashed_value] = full_item
        parsed_values.append(hashed_value)


        # if 'vertex_id.startswith('empty')', the vertex has no any variables
        # the vertex_id will has the value about the vertex id
        # or in vertex is not startwith 'empty', but the controlType is 1 (controllable var), then add 'wait' as an state for the var, and use f"{vertex_id}=wait" as the state value

        #Below 1 line code is used to add 'wait' to the controlable variable and contraollable variable in the vertex_id.startswith('empty'), 
        # but now, we don't need to add 'wait' to the controlable variable, except the vertex_id.startswith('empty'), rather use 'if vertex_id.startswith('empty')' as the condition to add 'wait' action
        ############if var_controlType == 1 or vertex_id.startswith('empty'):############
        
        # 2024-10-01: Deactivate the 'wait' state for the vertices that are not the 'empty...' vertex
        # Only add 'wait' as an action for the var, and use f"{vertex_id}=wait" as the state value, the vertex is the 'empty...' vertex (i.e., for waiting synchronization)
        if vertex_id.startswith('empty'):
            full_item = f"{vertex_id}=wait"  
            if full_item in wait_int16_to_string_map.values():
                # If the same string already exists, directly extract the hash value
                hashed_value = [key for key, value in wait_int16_to_string_map.items() if value == full_item][0]
            else:
                hashed_value = custom_hash_32bit(full_item, existing_hashes)

            wait_int16_to_string_map[hashed_value] = full_item    
            parsed_values.append(hashed_value)
    
    else:
        # for the domainType = 'action'
        parsed_values.extend(parse_variable_domain(vertex_id, var_name, var_domain, var_controlType, domainType = 'normal'))
        parsed_values.extend(parse_variable_domain(vertex_id, var_name, var_domain, var_controlType, domainType = 'wait_none'))
    
    return parsed_values


def get_endEvent_vertex(graph: BaseGraph): 
    vertices : List[Vertex] = graph.vertices
    # Current assumption: although use List as type structure, but in the BPMN model, there is only one end event vertex
    endEvent_vertices = []
    for v in vertices.values():
        if not any(edge.source == v.vertex_id for edge in graph.edges) and any(edge.target == v.vertex_id for edge in graph.edges):
            # just be executed once, because there is only one end event vertex in the BPMN model
            endEvent_vertices.append(v.vertex_id)

    # if future, there are multiple end event vertices in the BPMN model, then return the list of endEvent_vertices directly
    return endEvent_vertices[0]


# The 'retrieve_Vars_from_expression' function is used to retrieve the variables from an expression (and KPI expression in constraint)
def retrieve_Vars_from_expression(env : PMDPEnvironment, expression: str) -> list:
    # The OPERATOR_INFO is a dictionary, which contains the operators and their corresponding information
    operator_keys = OPERATOR_INFO.keys()


    ### Due to the operator '\sqrt' in the expression, is like '\sqrt{...} or \sqrt[...]{...}', 
    # we need to remove the '{' and '}' or '[' and ']' in the expression, those two symbols '{','}' and '(',')' are not the operators and not the variables or KPIs
    sqrt_tag = False
    if r'\sqrt' in expression:
        sqrt_tag = True

    if 'tmpKPI' in expression:
        pass

    # Replace the operators with spaces, in case non-operators has the same part of name as operators
    for key in operator_keys:
        expression = expression.replace(key, " ")

    expression_list = expression.split()
    #remove the duplicate variables or KPIs
    expression_list = list(set(expression_list))

    if sqrt_tag:
        
        for _ in range(len(expression_list)):
            if expression_list[_].startswith('{'):
                expression_list[_] = expression_list[_].replace('{', '')
                expression_list[_] = expression_list[_].replace('}', '')
            elif expression_list[_].startswith('['):
                expression_list[_] = expression_list[_].replace('[', '')
                expression_list[_] = expression_list[_].replace(']', '')

    # get the variables or KPIs from the expression
    VoK = []
    for key in env.context.keys():
        if key in expression_list:
            if key in env.current_state:               
                VoK.append(key)
            else:
                # the key is a KPI expression, we need to retrieve the variables from the expression
                for kpi in env.graph.KPIs:
                    if kpi.name == key:
                        kpi_expression = kpi.expression
                        VoK.extend(retrieve_Vars_from_expression(env, kpi_expression))
                        break
    return VoK


def classify_vars(env : PMDPEnvironment, constraints: List[Constraint]):
    
    # if constraints share the common variables, then they belong to the same class, all variables in those constraints will be stored in the same item in the unrelated_vars_list 
    # if constraints have no common variables, then they belong to different classes
    subStateSpace_vars_list = []
    for constraint in constraints:
        constraint_vars = retrieve_Vars_from_expression(env, constraint.expression)
        if len(constraint_vars) == 0:
            raise ValueError(f"Constraint {constraint.name} has no variables, please check the constraint or KPI expression")
        #remove the duplicate variables
        constraint_vars = list(set(constraint_vars))
        found = False
        for i in range(len(subStateSpace_vars_list)):
            c = subStateSpace_vars_list[i]
            if set(constraint_vars).intersection(set(c)):
                c.extend(constraint_vars)
                # remove the duplicate variables
                subStateSpace_vars_list[i] = list(set(c))
                found = True
                #break
        if not found:
            subStateSpace_vars_list.append(constraint_vars)

    # Sort subStateSpace_vars_list stably
    subStateSpace_vars_list = sorted(subStateSpace_vars_list, key=lambda x: (len(x), x))
    return subStateSpace_vars_list

def classify_constraints_by_substateSpace(env, subStateSpace_vars_name_list, whc, wsc):
    """
    Categorize HC and SC constraints by sub-state variable names.
    The 'subStateSpace_vars_name_list' is has the fixed order about the sub-state for each call of 'classify_constraints_by_substateSpace' function
    Parameters:
    - subStateSpace_vars_name_list: List of lists containing variable names for each sub-state.
    - graph: The graph object containing HC and SC constraints.

    Returns:
    - subState_HC_list: List of lists containing categorized HC constraints.
    - subState_SC_list: List of lists containing categorized SC constraints.
    """
    subState_HC_list = []
    subState_SC_list = []
    subState_HC_weights_list = [0] * len(subStateSpace_vars_name_list)
    subState_SC_weights_list = [0] * len(subStateSpace_vars_name_list)

    if len(env.graph.HC) == 0:
        whc = 0
        wsc = 1
    elif len(env.graph.SC) == 0:
        whc = 1
        wsc = 0
    
    for subStateVarNames in subStateSpace_vars_name_list:
        sub_HC = []
        sub_SC = []
        for hc in env.graph.HC:
            all_vars = retrieve_Vars_from_expression(env, hc.expression)
            #remove the duplicate variables
            all_vars = list(set(all_vars))
            if any([var_name in all_vars for var_name in subStateVarNames]):
                if hc.weight != 0:
                    sub_HC.append(hc)
                subState_HC_weights_list[subStateSpace_vars_name_list.index(subStateVarNames)] += hc.weight
        for sc in env.graph.SC:
            all_vars = retrieve_Vars_from_expression(env, sc.expression)
            #remove the duplicate variables
            all_vars = list(set(all_vars))
            if any([var_name in all_vars for var_name in subStateVarNames]):
                if sc.weight != 0:
                    sub_SC.append(sc)
                subState_SC_weights_list[subStateSpace_vars_name_list.index(subStateVarNames)] += sc.weight

        subState_HC_list.append(sub_HC)
        subState_SC_list.append(sub_SC)

    for i in range(len(subStateSpace_vars_name_list)):
        subState_HC_weights_list[i] *= whc
        subState_SC_weights_list[i] *= wsc

    return subState_HC_list, subState_SC_list, subState_HC_weights_list, subState_SC_weights_list, whc, wsc

def merge_intersections(lists: List[List[str]]) -> List[List[str]]:
    merged = []
    while lists:
        current = lists.pop(0)
        current_set = set(current)
        to_remove = []
        for i in range(len(lists)):
            other = lists[i]
            other_set = set(other)
            if not current_set.isdisjoint(other_set):
                current_set = current_set.union(other_set)
                to_remove.append(i)
        to_remove.reverse()
        for i in to_remove:
            lists.pop(i)
        merged.append(list(current_set))
    return merged

def retrieve_variables_from_graph(graph : WeightedDirectedGraph):
    variables_dict = {}
    for vertex in graph.vertices.values():
        for variable in vertex.C_v:
            variable.vertex_id = vertex.vertex_id
            variables_dict[variable.name] = variable
        for variable in vertex.U_v:
            variable.vertex_id = vertex.vertex_id
            variables_dict[variable.name] = variable
    return variables_dict

# get the elements in list1, but not in list2
def get_non_intersecting_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    unique_to_set1 = set1.difference(set2)
    return list(unique_to_set1)

# get state row_index for Q-table from the 2D stateTag tensor
def find_subtensor_row_index(tensor_2d, sub_tensor):
    # Compare each row with the sub-tensor
    matches = (tensor_2d == sub_tensor).all(dim=1)
    # 获取匹配的索引
    indices = torch.nonzero(matches).squeeze()
    return indices

def compute_2Dtensor_cartesian_product(tensor1, tensor2):
    """
    Compute the Cartesian product of two 2D tensors.

    Parameters:
    tensor1 (torch.Tensor): The first input 2D tensor.
    tensor2 (torch.Tensor): The second input 2D tensor.

    Returns:
    torch.Tensor: The Cartesian product as a 2D tensor.
    """
    # Get the number of rows and columns for each tensor
    rows1, cols1 = tensor1.size()
    rows2, cols2 = tensor2.size()
    
    # Repeat tensor1 for each row in tensor2
    tensor1_expanded = tensor1.unsqueeze(1).expand(-1, rows2, -1).reshape(-1, cols1)
    
    # Repeat each row in tensor2 for the number of rows in tensor1
    tensor2_expanded = tensor2.repeat(rows1, 1)
    
    # Concatenate the expanded tensors along the last dimension
    cartesian_product = torch.cat((tensor1_expanded, tensor2_expanded), dim=1)
    
    return cartesian_product

def sort_and_group_var_list(var_list):
    """
    Group the variable list by vertex_id and sort by name within each group.

    Parameters:
    var_list (list): a list containing variable information, each element is a list or tuple containing at least two elements,
    the first element is vertex_id, the second element is name.

    Returns:
    list: the sorted variable list.
    """
    # Group by vertex_id
    grouped_vars = defaultdict(list)
    for item in var_list:
        vertex_id = item[0]  # Use the first element as vertex_id
        grouped_vars[vertex_id].append(item)

    # Sort by name within each group
    sorted_var_list = []
    for vertex_id, items in grouped_vars.items():
        sorted_items = sorted(items, key=lambda x: x[1])  # Sort by name
        sorted_var_list.extend(sorted_items)  # Append the sorted results to the final list

    return sorted_var_list

def stable_2Dtensor_unique(tensor):
    """
    Remove duplicates from the given 2D tensor and keep the order consistent.

    Parameters:
    tensor (torch.Tensor): the input 2D tensor.

    Returns:
    torch.Tensor: the 2D tensor after removing duplicates.
    """
    # Get unique rows and their original index
    unique_tensor, inverse_indices = torch.unique(tensor, dim=0, return_inverse=True)

    # Get the device of the tensor
    device = tensor.device

    # Get the original order indices
    _, original_indices = torch.sort(torch.arange(inverse_indices.size(0), device=device)[inverse_indices])

    # Return unique rows according to the original order
    sorted_unique_tensor = tensor[original_indices].unique(dim=0)

    return sorted_unique_tensor

def calculate_cartesian_products_for_subWaitNnoneSpaces_subStateSpace(tensor1, tensor2, sameVertex_TagindexTensor_list):
    """
    Computes the Cartesian product of two 2D tensors based on column index ranges in sameVertex_TagindexTensor_list.

    Parameters:
    tensor1 (torch.Tensor): The first 2D tensor.
    tensor2 (torch.Tensor): The second 2D tensor.
    sameVertex_TagindexTensor_list (list): A list of column index ranges.

    Returns:
    list: A list of Cartesian product results for each index range.
    """
    results = []

    # Iterate over sameVertex_TagindexTensor_list
    for index_range in sameVertex_TagindexTensor_list:
        a, b = index_range

        # Extract column range [a, b] from tensor1
        sub_tensor1 = tensor1[:, a:b]
        print(f"sub_tensor1: {sub_tensor1}")
        unique_sub_tensor1 = stable_2Dtensor_unique(sub_tensor1)
        print(f"unique_sub_tensor1: {unique_sub_tensor1}")
 

        # Extract columns from tensor2 excluding the range [a, b]
        sub_tensor2 = torch.cat((tensor2[:, :a], tensor2[:, b:]), dim=1)
        print(f"tensor2[:, :a]: {tensor2[:, :a]}")
        print(f"tensor2[:, b:]: {tensor2[:, b:]}")
        print(f"sub_tensor2: {sub_tensor2}")
        unique_sub_tensor2 =stable_2Dtensor_unique(sub_tensor2)
        print(f"unique_sub_tensor2: {unique_sub_tensor2}")


        # Compute Cartesian product
        result = compute_2Dtensor_cartesian_product(unique_sub_tensor1, unique_sub_tensor2)
        print(f"result: {result}")

        # Add the results to a list
        results.append(result)

    # Concatenate all results along the row dimension
    concatenated_result = torch.cat(results, dim=0)
    return concatenated_result


# The 'filter_tensor' function is used to filter the tensor based on the given column ranges and the dict_map
def filter_subStateSpace_tensor(env: PMDPEnvironment, tensor: torch.Tensor, col_ranges: list, a_int16_to_string_map: dict) -> torch.Tensor:
    # Initialize a boolean index based on 'rows', initially keeping all rows
    row_mask = torch.ones(tensor.size(0), dtype=torch.bool, device=tensor.device)
    
    int16_keys = torch.tensor(list(a_int16_to_string_map.keys()), device=tensor.device)
    
    for col_range in col_ranges:
        col_start, col_end = col_range

        # Check the dimensions of a tensor
        if tensor.dim() == 1:
            sub_tensor = tensor[col_start:col_end]
        else:
            sub_tensor = tensor[:, col_start:col_end]

        # Check if any element in the given columns of each row is in int16_keys
        condition = env.isin(sub_tensor, int16_keys)

        # If the row contains at least one element in int16_keys
        has_int16_key = condition.any(dim=1) if tensor.dim() > 1 else condition.any()
        
        if has_int16_key.any():
            # Check if all elements belong to int16_keys
            all_key = condition.all(dim=1) if tensor.dim() > 1 else condition.all()

            # Check if all elements belong to int16_keys and are the same
            if tensor.dim() == 1:
                all_same_key = all_key & (sub_tensor == sub_tensor[0])
            else:
                all_same_key = all_key & (sub_tensor.eq(sub_tensor[:, 0].unsqueeze(1)).all(dim=1))
            
            # The rows that need to be deleted are those that have an int16_key but not an all_same_key
            rows_to_delete = has_int16_key & ~all_same_key

            # Update row mask to keep only the needed rows
            row_mask = row_mask & ~rows_to_delete

    # Use bool tensor as index to filter the tensor
    return tensor[row_mask] if tensor.dim() > 1 else tensor[row_mask.nonzero(as_tuple=True)[0]]

def flatten_tuple(t):
    """
    Flatten nested tuples into a comparable sequence.

    Parameters:
    t (tuple): input tuple, may contain nested tuples.

    Returns:
    list: the flattened sequence.
    """
    flat_list = []
    for item in t:
        if isinstance(item, tuple):
            flat_list.extend(flatten_tuple(item))
        else:
            flat_list.append(item)
    return flat_list

def stable_sort_tuple_set(tuple_set):
    """
    Stably sort a set of tuples.

    Parameters:
    tuple_set (set): The input set, containing tuples.

    Returns:
    list: The sorted list.
    """
    return sorted(tuple_set, key=flatten_tuple)

def filter_tensor_by_elements(tensor_a, list_b):
    """
    Filters rows in tensor_a, keeping the row if the element in the row exists in list_b.

    Parameters:
    tensor_a (torch.Tensor): Input 2D Tensor.
    list_b (list): List containing elements to check.

    Returns:
    torch.Tensor: The child Tensor that satisfies the condition.
    """
    # Initialize result list
    result = []

    # Iterate through each row in tensor_a
    for row in tensor_a:
        # Check if any element in the row exists in list_b
        if any(element.item() in list_b for element in row):
            # If the condition is met, add the row to the result list
            result.append(row)

    # Convert the result list to a new 2D Tensor
    if result:
        return torch.stack(result)
    else:
        return torch.empty((0, tensor_a.size(1)), dtype=tensor_a.dtype, device=tensor_a.device)
    
def filter_ActionTensor_by_columns(tensor, column_indices, value_dict, subAction_vertices_ids, horizion = False):
    """
    Filters the rows of a 2D tensor, keeping those rows where the values in all other column indices are equal to the values stored in the dictionary '...=None'.

    Parameters:
    tensor (torch.Tensor): The input 2D tensor.
    column_indices (list): List of column indices, e.g. [[0, 2], [2, 4]].
    value_dict (dict): Dictionary storing column values, e.g. {key: '...=None'}.
    subAction_vertices_ids: List of vertex IDs of the sub-action space corresponding to column_indices, used to identify corresponding rows of each vertex in the tensor

    Returns:
    torch.Tensor: The filtered 2D tensor, with valid rows and columns indices.
    """
    if not horizion:
        # Ensure NoneAction_dict has the same order as subAction_vertices_ids
        NoneAction_dict = {k: value_dict[k] for vertex_id in subAction_vertices_ids for k, v in value_dict.items() if v.split('=')[1] == 'None' and v.split('=')[0] == vertex_id}
    else:
        NoneAction_dict = value_dict
    # Pre-convert the keys of the dictionary to a list
    keys_set = list(NoneAction_dict.keys())
    keys_tensor = torch.tensor(keys_set, dtype=tensor.dtype, device=tensor.device)
    
    filtered_rows = []
    rowsIndex = 0
    rowsTga = []
    filtered_rowsTags = []
    
    Vertex_index = 0
    old_vertex_id = subAction_vertices_ids[Vertex_index]

    for col_range in column_indices:
        current_vertex_id = subAction_vertices_ids[Vertex_index]
        if current_vertex_id != old_vertex_id:
            if len(rowsTga) > 0:
                rowsTga.append(rowsIndex)
                filtered_rowsTags.append(rowsTga)
                rowsTga = []
                old_vertex_id = current_vertex_id
        
        start_col, end_col = col_range
        if tensor.dim() == 1:
            # If it's a 1D tensor, process it directly
            other_cols = torch.cat((tensor[:start_col], tensor[end_col:]))
            mask = torch.isin(other_cols, keys_tensor).all()
            valid_rows = tensor.unsqueeze(0) if mask else torch.empty((0,), dtype=tensor.dtype, device=tensor.device)

            # Exclude columns containing '...=None'
            if torch.isin(valid_rows[start_col:end_col], keys_tensor).any():
                valid_rows = torch.empty((0,), dtype=tensor.dtype, device=tensor.device)
        else:
            other_cols = torch.cat((tensor[:, :start_col], tensor[:, end_col:]), dim=1)
            # Check using tensor operations
            mask = torch.isin(other_cols, keys_tensor).all(dim=1)
            valid_rows = tensor[mask]

            # Filter out rows where the value in the col_range columns is in keys_tensor
            # that is, the value is '...=None' for the active vertices, which is not legal in the action space
            valid_rows = valid_rows[~torch.isin(valid_rows[:, start_col:end_col], keys_tensor).any(dim=1)]
        
        if valid_rows.size(0) > 0:
            if len(rowsTga) == 0:
                rowsTga.append(rowsIndex)
            rowsIndex += valid_rows.size(0)
            filtered_rows.append(valid_rows)
        
        old_vertex_id = current_vertex_id
        Vertex_index += 1

    rowsTga.append(rowsIndex)
    filtered_rowsTags.append(rowsTga)

    if filtered_rows:
        return torch.cat(filtered_rows) if tensor.dim() > 1 else torch.stack(filtered_rows), filtered_rowsTags
    else:
        return torch.empty((0, tensor.size(1)), dtype=tensor.dtype, device=tensor.device) if tensor.dim() > 1 else torch.empty((0,), dtype=tensor.dtype, device=tensor.device), filtered_rowsTags


def process_horizon(env, horizion, horizon_int16_to_string_map, existing_hashes, learning_device):
    """
    Process horizon and generate filtered_horizonTag_StateSpace_tensor.

    Parameters:
    horizion (list): list of time steps in PMDP.
    env (object): environment object, contains convert_horizon_item_to_list method and graph attribute.
    horizon_int16_to_string_map (dict): dictionary storing horizon hash value and string mapping.
    existing_hashes (set): set of existing hash values.
    learning_device (torch.device): device used to store tensors.

    Returns:
    torch.Tensor: filtered horizonTag state space tensor.
    """
    # Store the hash values of the horizon space, each item is a time step in PMDP
    horizion_domain = []

    # Ensure the order of active vertices in the horizon is consistent during Q-table state space row construction
    for item in horizion:
        item_list = env.convert_horizon_item_to_list(item)
        item_list = sorted(item_list)
        if any(item.startswith('empty') or (len(env.graph.vertices[item].C_v) == 0 and len(env.graph.vertices[item].U_v) == 0) for item in item_list):
            item_list_as_string = ','.join(item_list)

            if item_list_as_string in horizon_int16_to_string_map.values():
                # If the same string already exists, directly take out the hash value
                horizon_hash_value = [key for key, value in horizon_int16_to_string_map.items() if value == item_list_as_string][0]
            else:
                horizon_hash_value = custom_hash_32bit(item_list_as_string, existing_hashes)
            
            horizon_int16_to_string_map[horizon_hash_value] = item_list_as_string
            horizion_domain.append(horizon_hash_value)

    # Store the hash values of the horizon space, each item is a time step in PMDP
    sorted_horizion_domain = sorted(horizion_domain)
    filtered_horizonTag_StateSpace_tensor = torch.tensor(sorted_horizion_domain, dtype=torch_intType, device=learning_device)

    return filtered_horizonTag_StateSpace_tensor



def get_states_for_Q_rows(env : PMDPEnvironment, horizion : Set[Union[str, tuple]], graph: BaseGraph) -> torch.Tensor:

    global wait_int16_to_string_map
    global none_int16_to_string_map

    # Get the endEvent vertex in BPMN
    endEvent_vertexID = get_endEvent_vertex(graph)

    # Each item in sub_vars_list stores a series of variables that are related to each other due to constraints, forming a sub state space. 
    # Variables in the same subspace need to be combined using the Cartesian product to construct the sub state space.
    # There is no relationship between each item, and there are no duplicate variables within each item, so each item forms an independent sub state space.
    # Each sub state space is a part of the entire state space, which form a sub Q-table for the Q-function, which model state-action pair about variables that only belong to the sub state space.
    # Finally, all sub Q-tables are combined to form the entire Q-table for the Q-function.
    # The function 'classify_vars' is used to classify the variables in the constraints into different sub state spaces.
    constrainst_list = graph.HC + graph.SC
    # subStateSpace_vars_list containts the names of variables, not the objects of variables
    subStateSpace_vars_list = classify_vars(env, constrainst_list)
    subStateSpace_vars_list = merge_intersections(subStateSpace_vars_list)

    # vars_dict is a dictionary, which stores the objects of variables
    vars_dict = retrieve_variables_from_graph(graph)

    # 'all_vars_in_constraints' only contain names of variables
    all_vars_in_constraints = []
    for sub_vars in subStateSpace_vars_list:
        all_vars_in_constraints.extend(sub_vars)
    vars_not_in_constraints_list = get_non_intersecting_elements(vars_dict.keys(), all_vars_in_constraints)


    ##### The uncertainty of uncontrollable variables （in different vertex, i.e., activity） is assumed to be a 'Independent Distribution' in PMDP Environment #####
    ##### But in some senarios, in same vertex, the uncontrollable variables may have the 'Joint Distribution' relationship with the controllable variables #####
    ##### For example, in 'QoS aware service composition', not only we can select which candidate service to be composed, but the QoS of the candidate service is uncontrollable and  related to the selected service #####
    ##### So when this situation occurs, we need to identify the relationship between the controllable variables and the uncontrollable variables, and make it clear #####
    ##### You can connect it with the constraints, then we can automatically identify the relationship between the controllable variables and the uncontrollable variables #####
    ##### If those variables cannot be connected by constraints, you need use the 'format variable name rules' to model it , such as 'var1~var2' is the complete name of uncontrollable variable, where 'var1' is the name of controllable variable #####
    for var_not_in_constraints in vars_not_in_constraints_list:
        # assume that each controllbale variable can have a or multiple uncontrollable variables (those variables must be same vertex) having the joint distribution relationship with it
        # the uncontrollable variables (e.g., var2) use the name of controllable variable (var1) as the prefix, and add '~' as the separator, such as 'var1~var2' is the complete name of uncontrollable variable
        # those var in vars_not_in_constraints_list must have some variables in the constraints that have the relationship with them

        # The Table based Constraint-Aware Double Q-learning need this to construct 'subStateSpace, the Neural Network based does not need this that only one state space
        if '~' not in var_not_in_constraints:
            # it is a controllable variable
            for vars in subStateSpace_vars_list:
                
                if any(element.startswith(f"{var_not_in_constraints}~") for element in vars):
                    # find the uncontrollable variable of the controllable variable, and add the controllable variable to the corresponding sub state space
                    vars.append(var_not_in_constraints)
                    break
        else:  
            # it is a uncontrollable variable
            # get the 'left_part' of the variable name, which is the name of controllable variable
            cv_name = var_not_in_constraints.split('~', 1)[0]
            for vars in subStateSpace_vars_list:
                if cv_name in vars:
                    vars.append(var_not_in_constraints)
                    break
            
    # check if there are variables that are not related to any constraints, so they are not included in the any state space
    # 'all_vars_in_StateSpace' only contain names of variables
    all_vars_in_StateSpace = []
    for sub_vars in subStateSpace_vars_list:
        all_vars_in_StateSpace.extend(sub_vars)

    vars_without_relationship_about_Q = get_non_intersecting_elements(vars_dict.keys(), all_vars_in_StateSpace) 
    if len(vars_without_relationship_about_Q) > 0:
        # if there are variables that are not related to any constraints, then they are not included in the any state space, so they are not included in the Q-table
        print(f"Warning: the following variables are not included in the Q-table, because they are not related to any constraints or other variables: {vars_without_relationship_about_Q}")


    sub_stateTag_tensor_list = []
    sub_stateSpace_vars_sorted_list = []
    

    for subStateSpace_vars in subStateSpace_vars_list:
        var_list = []
        variables_spaces = []
        # get var information
        for var_name in subStateSpace_vars:
            var = vars_dict[var_name]
            var_list.append([var.vertex_id, var.name, var.domain, var.controlType])

        # sort the variable based on the vertex_id
        # make sure the order of variable is consistent comply with vertex_id, in each execution when call 'sort_and_group_var_list' function
        var_list.sort(key=lambda x: (x[0]))

        # sort the variable, make sure the order of variable is consistent in each execution when constructing the states space for rows of Q-table
        # sort the variable based on the vertex_id, and then based on the name of variable
        # Use 'var.vertex_id' as the sort key to ensure that variables in the same vertex are continuous in the sorted var_list so that the filter_tensor function can correctly filter illegal actions
        var_list = sort_and_group_var_list(var_list)
        sub_stateSpace_vars_sorted_list.append(var_list)

        # Initialize an empty list for the parised variable domain    
        varDomain_list = []

        # each item in 'sameVertex_waitNone_TagindexTensor_list' store the index of the beginning and end of column for someone vertex all variables in the tensor
        # actually for the 'wait' state, the controllable variables in the same vertex should be set to f'{vertex_id}=wait' state, and the other uncontrollable variables should be set to 'None' state
        # but, if use the below actually situation to construct the state space, the state space will waste some memory,
        # because, for the Q-table, if store the 'wait' and 'None' state respectively for the variables in the same vertex, the state space for wait and None of this vertex will be 2*2
        # compare with store the 'wait' state for the all variables in the same vertex together, the memory will be saved, and without difference in the learning process, the state space for wait and None of this vertex will be 2
        # the reason is, for the Q-table to find the current state in the Q-table ( if current is the 'wait' state), the uncontrolled variables state is not related to this process
        # so if one vertex stay in the 'wait' state, we also store the 'wait' state for the uncontrolled variables in the same vertex
        # For example, if a vertex has controllable vars  A, B, and uncontrollable vars C.
        # in the initial state, the controllable vars A, B are set to 'None' state to represent the initial state, the process (PMDP) havn't executed here yet
        # when PMDP execute to this vertex, and policy decide to 'wait', then use A,B,C all is 'wait' state to represent the state of the vertex
        # rather than, use A,B is 'wait' state, and C is 'None' state to represent the state of the vertex, because the 'wait' state of the vertex in Q-table is not related to the uncontrollable variables
        sameVertex_waitNone_TagindexTensor_list = []
        temp_wait_None_Tag = []

        current_var_vertex_id = var_list[0][0]
        for i in range(len(var_list)):
            var = var_list[i]
            if len(temp_wait_None_Tag) == 0:
                temp_wait_None_Tag.append(i)
            # 对子空间中的变量进行解析 and map the domain of variables to the int16 type
            varDomain_list.append(parse_variable_domain(var[0], var[1], var[2], var[3]))

            if i < len(var_list) - 1:
                next_var = var_list[i+1]
                if len(temp_wait_None_Tag) == 1 and (next_var[0] != current_var_vertex_id):
                    temp_wait_None_Tag.append(i+1)
                    sameVertex_waitNone_TagindexTensor_list.append(temp_wait_None_Tag)
                    temp_wait_None_Tag = []
                
                current_var_vertex_id = next_var[0] 

            if i == len(var_list) - 1:
                if len(temp_wait_None_Tag) == 1:
                    temp_wait_None_Tag.append(i+1)
                    sameVertex_waitNone_TagindexTensor_list.append(temp_wait_None_Tag)

        wait_or_none_varDomain_list = []
        wait_or_none_spaces =[]
        for i in range(len(var_list)):
            var = var_list[i]
            # Parse the variables in the subspace and map the domain of variables to the int16 type
            wait_or_none_varDomain_list.append(parse_variable_domain(var[0], var[1], var[2], var[3], domainType = 'wait_none'))

        # append the parsed variable domain to the torch list
        for i in range(len(varDomain_list)):
            variables_spaces.append(torch.tensor(varDomain_list[i], dtype=torch_intType, device = learning_device))
            wait_or_none_spaces.append(torch.tensor(wait_or_none_varDomain_list[i], dtype=torch_intType, device = learning_device))


        if len(wait_or_none_spaces) == 1 and len(var_list) == 1 and wait_or_none_spaces[0].size()[0] > 1:
            subStateSpace2 = wait_or_none_spaces[0].clone().detach().unsqueeze(1)
        else:
            # all variables in each row of the tensor are set to 'wait' or 'none' state
            subStateSpace2 = torch.cartesian_prod(*wait_or_none_spaces)
        
        temp_dict = {}
        temp_dict.update(wait_int16_to_string_map)
        temp_dict.update(None_int16_to_string_map)
        filtered_subStateSpace2 = filter_subStateSpace_tensor(env, subStateSpace2, sameVertex_waitNone_TagindexTensor_list, temp_dict)


        vertex_tensor_list = []

        for index_range in sameVertex_waitNone_TagindexTensor_list:
            a, b = index_range
            variables_tensor_list_in_same_vertex = []
            variables_spaces_tensorList_in_same_vertex = variables_spaces[a:b]
            variables_tensor_list_in_same_vertex.append(variables_spaces_tensorList_in_same_vertex)

            vertex_subStateSpace_without_waitNone = torch.cartesian_prod(*variables_tensor_list_in_same_vertex[0])

            # convert the 1D tensor to 2D tensor, when the number of variables in the vertex is only one
            if vertex_subStateSpace_without_waitNone.dim() == 1:
                vertex_subStateSpace_without_waitNone = vertex_subStateSpace_without_waitNone.unsqueeze(1)
            
            wait_None_hashIds = wait_or_none_varDomain_list[a]

            wait_None_tensor_space = filter_tensor_by_elements(filtered_subStateSpace2[:, a:b], wait_None_hashIds)
            wait_None_tensor_space = stable_2Dtensor_unique(wait_None_tensor_space)

            vertex_tensor_list.append(torch.cat((vertex_subStateSpace_without_waitNone, wait_None_tensor_space), dim=0))

        for i in range(len(vertex_tensor_list)):
            if i + 1 == len(vertex_tensor_list):
                break
            else:   
                vertex_tensor_list[i+1] = compute_2Dtensor_cartesian_product(vertex_tensor_list[i], vertex_tensor_list[i+1])

        sub_stateTag_tensor_list.append(vertex_tensor_list[-1])

    return sub_stateSpace_vars_sorted_list, sub_stateTag_tensor_list


# The 'parse_variable_domain' function is used to parse the variable domain of variable of the vertex
# In action space, if the vertex has controllable variables, the f'{vertex_id}=wait' action is added to the action space of each variable domain.
# If the vertex does not have any controllable variables, the f'{vertex_id}=finish' action is added to the action space of the vertex, 
# for the action will tell PMDP to execute this vertex, but it is not related to any state of Q-table, because it have no any controllable variables even maybe have uncontrollable variables.

# Except the 'empty...' vertex (no any variables), the 'wait' and 'finish' actions are only one can be selected in the action space of the vertex
# Because if the vertex has no controllable variables, we don't need to wait. We can directly transite to next vertex
# For each 'wait' action, and use vertex_id as the wait action name (i.e., 'vertex_id=wait' in wait_or_none_int16_to_string_map
def get_actionSpace_for_Q_columns(env : PMDPEnvironment, sub_stateSpace_vars_sorted_list: List, ) -> torch.Tensor:

    subActionSpace_tensor_list = []

    # Get ids of all vertieces in the graph
    vertices_ids = env.graph.vertices.keys()

    # vars_dict is a dictionary, which stores the objects of variables about all vertices in the graph
    vars_dict = retrieve_variables_from_graph(env.graph)


    result_verticesOrder_rowsColumnsTags_inTensor_list = []

    
    for subStateSpace_vars in sub_stateSpace_vars_sorted_list:
        
        result_subActionSpace_vertices_rowsColumnsTags_data = []

        # Initialize an empty tensor list
        current_action_spaces = []
        # make sure the order of variable is consistent comply with vertex_id and varName, in each execution when call 'sort_and_group_var_list' function、



        ########### 2024-10-30 Comment the following 2 lines of code if subStateSpace_vars_sorted_list has been sorted in get_states_for_Q_rows function for new version of PMDP ##########
        subStateSpace_vars.sort(key=lambda x: (x[0]))
        subStateSpace_vars = sort_and_group_var_list(subStateSpace_vars)

        sub_actionSpace_var_names = [var[1] for var in subStateSpace_vars]
        sub_actionSpace_var_names_dict = {key: vars_dict[key] for key in sub_actionSpace_var_names if key in vars_dict}

        
        # get the vertices that are related to the controllable vars in sub action space
        subAction_vertices_ids = []

        for var in sub_actionSpace_var_names_dict.values():
            subAction_vertices_ids.append(var.vertex_id)
        # in case have duplicate vertex ids, remove the duplicate vertex ids
        subAction_vertices_ids = list(set(subAction_vertices_ids)) 
        subAction_vertices_ids.sort()

        waitNoneFinish_actions_TagindexTensor_list = []
        Tag_index_inTensor = 0

        for vertex_id in subAction_vertices_ids:
            # the item in the 'controllable_vars_dict' is {var.name: [var.id, var.name, var.domain, var.controlType] }
            # get all controllable variables for the vertex
            controllable_vars_dict = env.get_controllable_vars(vertex_id)
            # vars_intersection contains the names of variables that are both in the controllable_vars_dict and vars_in_subStateSpace
            # vars_intersection = list(set(list(controllable_vars_dict.keys())) & set(sub_actionSpace_var_names))

            # The new verison, has make sure each vertex in the graph at least have one controllable variable, so the 'vars_intersection' is based on all state space variables, not only sub state space variables
            vars_intersection = [var_name for var_name in controllable_vars_dict.keys() if var_name in retrieve_variables_from_graph(env.graph).keys()]
            vars_intersection.sort()

            controllable_vars_intersection_dict = {}
            for var_name in vars_intersection:
                # only keep the controllable variables that are in the subStateSpace
                controllable_vars_intersection_dict[var_name] = controllable_vars_dict[var_name]

            # Initialize an empty list for the parised variable domain    
            var_domain_list = []
            temp_tag = []

            # 'temp_tag' use to tag the column extent of the variables of same vertex in the tensor action space, which will be used to filter the illegal actions 
            # (i.e., when having any variable in the vertex is set 'wait', then all variables must be 'wait' in the vertex, otherwise, the action is illegal)
            # the 'wait' action is represented by the 'vertex_id=wait' in 'wait_or_none_int16_to_string_map' for each variable in the same vertex
            temp_tag.append(Tag_index_inTensor)

            # Has more than one controllable variables for the vertex
            # The 'wait' action is added to the action space of each variable domain if the vertex has controllable variables, 
            # only when all controllable variables in the vertes are set to 'wait' is legal, we will filter the illegal actions by 'filter_tensor'
            # actually, the here version 'varDirection (i.e.,  varDomain_varDirection[1])' is not used
            for var in controllable_vars_intersection_dict.values():
                # Parse the variable domain, it will contain a 'wait' action, and use vertex_id as the wait action name (i.e., 'vertex_id=wait' in wait_or_none_int16_to_string_map)
                var_domain_list.append(parse_variable_domain(var[0], var[1], var[2], var[3], domainType = 'action'))
                Tag_index_inTensor += 1   

            if controllable_vars_intersection_dict:
                temp_tag.append(Tag_index_inTensor)
                waitNoneFinish_actions_TagindexTensor_list.append(temp_tag)

            
            # Not any controllable variables for the vertex
            # If the vertex does not have any controllable variables, but the vertex has uncontrollable variables, add 'finish' as an action for the vertex means the vertex is executed
            # because we assume those variables are independent or have a joint distribution in same vertex for some controllable and uncontrollbale variables, 
            # so if the vertex does not have any controllable variables, even the vertex has uncontrollable variables, we can directly add 'finish' as an action for the vertex, 
            # add 'wait' is not necessary and without any meaning, because by  the 'finish' we can get more information in new state. Rather, if have 'wait', nothing will happen for the vertex
            # The 'not controllable_vars_intersection_dict' means the vertex have uncontrollable variables involving in the sub state space, but no controllable variables
            if not controllable_vars_intersection_dict:
                var_domain = []
                full_item = f"{vertex_id}=None"
                if full_item in None_int16_to_string_map.values():
                    # If the same string already exists, directly take the hash value
                    hashed_value = [key for key, value in None_int16_to_string_map.items() if value == full_item][0]
                else:
                    # Use hash to map the string to a unique integer
                    hashed_value = custom_hash_32bit(full_item, existing_hashes)

                None_int16_to_string_map[hashed_value] = full_item     
                var_domain.append(hashed_value)
             
                full_item = f"{vertex_id}=finish"
                if full_item in finish_int16_to_string_map.values():
                    # If the same string already exists, directly take the hash value
                    hashed_value = [key for key, value in finish_int16_to_string_map.items() if value == full_item][0]
                else:
                    # Use hash to map the string to a unique integer
                    hashed_value = custom_hash_32bit(full_item, existing_hashes)

                finish_int16_to_string_map[hashed_value] = full_item     
                var_domain.append(hashed_value)
                var_domain_list.append(var_domain)
                Tag_index_inTensor += 1
                temp_tag.append(Tag_index_inTensor)
                waitNoneFinish_actions_TagindexTensor_list.append(temp_tag)  
            

            # append the parsed variable domain to the torch list
            for item in var_domain_list:
                current_action_spaces.append(torch.tensor(item, dtype = torch_intType, device = learning_device))

        

        if len(current_action_spaces) == 1 and len(vertices_ids) == 1 and current_action_spaces[0].size()[0] > 1:
            # If there is only one vertex and one controllable variable, return the action space directly
            action_space = current_action_spaces[0].clone().detach().unsqueeze(1)
        else:
            # Calculate the Cartesian product of all action spaces of variables, that is, the action space of the current time-step (i.e., current state)
            # the 'torch.cartesian_prod' function can only take 1-D tensors as input, that is, the len(current_action_spaces) should larger than 1
            action_space = torch.cartesian_prod(*current_action_spaces)


        # because the 'wait/None' should be at the vertex level to construct cartesian product, due to the 'torch.cartesian_prod' function can only take 1-D tensors as input
        # so we add the 'wait/None' to each variable domain (i.e., variable level), so will result some redundant (illegal) actions in the action space
        # we need to filter these redundant actions 
        temp_dict = {}
        temp_dict.update(wait_int16_to_string_map)
        temp_dict.update(None_int16_to_string_map)
        temp_dict.update(finish_int16_to_string_map)
        if action_space.dim() > 1:
            # 'filter_subStateSpace_tensor' will return all variables in the same vertex are set to 'wait' or 'None' state together
            temp_tensor = filter_subStateSpace_tensor(env, action_space, waitNoneFinish_actions_TagindexTensor_list, temp_dict)
            filtered_tensor, rows_tgas = filter_ActionTensor_by_columns(temp_tensor, waitNoneFinish_actions_TagindexTensor_list, None_int16_to_string_map, subAction_vertices_ids)
        else:
            # the action space only have one variable, so no need to filter the action space
            filtered_tensor = action_space
            rows_tgas = [[0, action_space.size()[0]]]

        subActionSpace_tensor_list.append(filtered_tensor)



        # 'subAction_vertices_ids' show the vertex order in the tensor action space
        result_subActionSpace_vertices_rowsColumnsTags_data.append(subAction_vertices_ids)
        # 'rows_tgas' show the row of each vertex in the tensor action space
        result_subActionSpace_vertices_rowsColumnsTags_data.append(rows_tgas)

        # 'waitNoneFinish_actions_TagindexTensor_list' show the column of each vertex in the tensor action space 
        # for the subActionSpace only has one variable, the 'waitNoneFinish_actions_TagindexTensor_list' is not used, so we set it to  [[0, 1]], any value is ok
        result_subActionSpace_vertices_rowsColumnsTags_data.append(waitNoneFinish_actions_TagindexTensor_list)

        result_verticesOrder_rowsColumnsTags_inTensor_list.append(result_subActionSpace_vertices_rowsColumnsTags_data)


    return subActionSpace_tensor_list, result_verticesOrder_rowsColumnsTags_inTensor_list


def progressive_filter_action_space_gpu(pmdp_env, tensor_list, column_indices, value_dict, subAction_vertices_ids, batch_size=10000, horizon=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tensor_list = [tensor.to(device) for tensor in tensor_list]
    sizes = [tensor.size(0) for tensor in tensor_list]
    n_dims = len(sizes)
    total_size = torch.prod(torch.tensor(sizes)).item()

    filtered_results = []
    row_tags_results = []

    num_batches = (total_size + batch_size - 1) // batch_size
    count = 0

    # Precompute the cumulative size of each dimension (used to compute indices)
    cumulative_sizes = [1] * n_dims
    for i in range(n_dims - 2, -1, -1):
        cumulative_sizes[i] = cumulative_sizes[i + 1] * sizes[i + 1]
    cumulative_sizes = torch.tensor(cumulative_sizes, device=device)

    for batch_start in range(0, total_size, batch_size):
        batch_end = min(batch_start + batch_size, total_size)
        batch_range = batch_end - batch_start

        # Generate global indices within the batch
        batch_indices = torch.arange(batch_start, batch_end, device=device)

        # Calculate the index for each dimension
        indices = (batch_indices.unsqueeze(1) // cumulative_sizes) % torch.tensor(sizes, device=device)

        # Select the corresponding tensors using the indices
        batch_tensors = [tensor.index_select(0, indices[:, dim]) for dim, tensor in enumerate(tensor_list)]

        # Merge to generate action space
        action_space_batch = torch.stack(batch_tensors, dim=1)

        # Execute filtering operation
        filtered_tensor_batch, _ = filter_ActionTensor_by_columns(
            action_space_batch, column_indices, value_dict, subAction_vertices_ids, horizon
        )

        if filtered_tensor_batch.size(0) > 0:
            filtered_results.append(filtered_tensor_batch)

        # Optional: Print progress
        count += 1
        print(f"Processed batch {count}/{num_batches}")

        # Clear cache and free GPU memory
        del batch_tensors, action_space_batch, filtered_tensor_batch, indices, batch_indices
        torch.cuda.empty_cache()
    
    if filtered_results:
        final_filtered_tensor = torch.cat(filtered_results, dim=0)

        # before 'batch' processing, the 'row_tags_results' is not correct, so need to re-calculate the 'row_tags_results'
        _1, row_tags_results = filter_ActionTensor_by_columns(
            final_filtered_tensor, column_indices, value_dict, subAction_vertices_ids, horizon
        )

        return final_filtered_tensor, row_tags_results
    else:
        empty_shape = (0, len(tensor_list))
        empty_tensor = torch.empty(empty_shape, dtype=tensor_list[0].dtype, device=device)
        return empty_tensor, []


def progressive_filter_action_space_gpu_v2(pmdp_env, tensor_list, column_indices, value_dict, subAction_vertices_ids, batch_size=10000, horizon=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tensor_list = [tensor.to(device) for tensor in tensor_list]
    sizes = [tensor.size(0) for tensor in tensor_list]
    n_dims = len(sizes)
    total_size = torch.prod(torch.tensor(sizes)).item()

    filtered_results = []
    row_tags_results = []

    num_batches = (total_size + batch_size - 1) // (max(batch_size // 100, 10000))
    count = 0

    # Precompute the cumulative size of each dimension (used to compute indices)
    cumulative_sizes = [1] * n_dims
    for i in range(n_dims - 2, -1, -1):
        cumulative_sizes[i] = cumulative_sizes[i + 1] * sizes[i + 1]
    cumulative_sizes = torch.tensor(cumulative_sizes, device=device)

    # Split index computation into smaller chunks
    chunk_size = max(batch_size // 100, 10000)  # Smaller computation chunks
    for batch_start in tqdm(range(0, total_size, batch_size), desc="Processing Batches"):
        batch_end = min(batch_start + batch_size, total_size)
        batch_range = batch_end - batch_start

        for chunk_start in range(0, batch_range, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_range)
            chunk_indices = torch.arange(batch_start + chunk_start, batch_start + chunk_end, device=device)

            # Calculate the index for each dimension
            indices = (chunk_indices.unsqueeze(1) // cumulative_sizes) % torch.tensor(sizes, device=device)

            # Select the corresponding tensors using the indices
            batch_tensors = [tensor.index_select(0, indices[:, dim]) for dim, tensor in enumerate(tensor_list)]

            # Merge to generate action space
            action_space_batch = torch.stack(batch_tensors, dim=1)

            # Execute filtering operation
            filtered_tensor_batch, _ = filter_ActionTensor_by_columns(
                action_space_batch, column_indices, value_dict, subAction_vertices_ids, horizon
            )

            if filtered_tensor_batch.size(0) > 0:
                filtered_results.append(filtered_tensor_batch)


            # Optional: Print progress
            count += 1
            print(f"Horizon Q Initializing batches: {count}/{num_batches}")

            # Clear cache and free GPU memory
            del batch_tensors, action_space_batch, filtered_tensor_batch, indices, chunk_indices
            torch.cuda.empty_cache()

    # Merge all batch results
    if filtered_results:
        final_filtered_tensor = torch.cat(filtered_results, dim=0)

        # before 'batch' processing, the 'row_tags_results' is not correct, so need to re-calculate the 'row_tags_results'
        _1, row_tags_results = filter_ActionTensor_by_columns(
            final_filtered_tensor, column_indices, value_dict, subAction_vertices_ids, horizon
        )

        return final_filtered_tensor, row_tags_results
    else:
        empty_shape = (0, len(tensor_list))
        empty_tensor = torch.empty(empty_shape, dtype=tensor_list[0].dtype, device=device)
        return empty_tensor, []





def get_QHorizon_actionSpace_columns(env : PMDPEnvironment, horizon_space, filtered_QHorizon_tensor: List) -> torch.Tensor:

        result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list = []
        QHorizon_columns_TagindexTensor_list = []
        QHorizon_columns_vertices_list = []
        Tag_index_inTensor = 0


        # the '{vertex_id}=wait' will be used to update 'the variable state' of PMDP if the vertex have variables
        # but if the vertex does not have any variables, such as 'empty...', the '{vertex_id}=wait' will be used to represent the action of the vertex (make the PMDP transit to next state), it is not related to 'any variable state' of Q-table
        # for the '{vertex_id}=finish', it is not used to update 'the variable state' of PMDP, it is used to represent the action of the vertex (make the PMDP transit to next state), it is not related to 'any variable state' of Q-table
        # In QHorizon all are used to represent the action of the vertex (make the PMDP transit to next state), it is not related to 'any variable state' of Q-table, in other words, the action only related to the vertex having no any variables.
        Qhorizon_actionSpace_tensor_list = []
        temp_None_QHorizon_int16_to_string_map = {}
       
        # Iterate through each active vertex to collect action spaces
        for itemHash in filtered_QHorizon_tensor:
            timeStep = horizon_int16_to_string_map[itemHash.item()]
            vertices_ids_list = timeStep.split(',')
            timeStep_domain_list = []
            temp_tag = []

            for vertex_id in vertices_ids_list:
                vertex_domain = []
                c_vars = env.graph.vertices[vertex_id].C_v
                u_vars = env.graph.vertices[vertex_id].U_v
                
                if vertex_id.startswith('empty') or (not c_vars and not u_vars):
                    if len(temp_tag) == 0:
                        temp_tag.append(Tag_index_inTensor)

                    # for the 'empty...' vertex, add the 'wait' and 'finish' to the action space of the vertex in the timeStep 
                    # 'empty...' vertex has no controllable and uncontrollable variables
                    if vertex_id.startswith('empty'):
                        full_item = f"{vertex_id}=wait"
                        if full_item in wait_int16_to_string_map.values():
                            # If the same string already exists, directly take the hash value
                            hashed_value = [key for key, value in wait_int16_to_string_map.items() if value == full_item][0]
                        else:
                            # Use hash to map the string to a unique integer
                            hashed_value = custom_hash_32bit(full_item, existing_hashes)
                            wait_int16_to_string_map[hashed_value] = full_item
                        vertex_domain.append(hashed_value)
                        
                        full_item = f"{vertex_id}=finish"
                        if full_item in finish_int16_to_string_map.values():
                            # If the same string already exists, directly take the hash value
                            hashed_value = [key for key, value in finish_int16_to_string_map.items() if value == full_item][0]
                        else:
                            # Use hash to map the string to a unique integer
                            hashed_value = custom_hash_32bit(full_item, existing_hashes)
                            finish_int16_to_string_map[hashed_value] = full_item
                        vertex_domain.append(hashed_value)
                       
                        
                        # to represent the action without any meaning, when the vertex is not active vertex in that time step
                        full_item = f"{vertex_id}=None"
                        if full_item in None_int16_to_string_map.values():
                            # If the same string already exists, directly take the hash value
                            hashed_value = [key for key, value in None_int16_to_string_map.items() if value == full_item][0]

                            temp_None_QHorizon_int16_to_string_map[hashed_value] = full_item
                        else:
                            # Use hash to map the string to a unique integer
                            hashed_value = custom_hash_32bit(full_item, existing_hashes)
                            None_int16_to_string_map[hashed_value] = full_item
                            temp_None_QHorizon_int16_to_string_map[hashed_value] = full_item
                        vertex_domain.append(hashed_value)
                        
                    if (not c_vars and not u_vars) and not vertex_id.startswith('empty'):
                        # If the vertex does not have any variables, add 'finish' as an action for the vertex means the vertex is executed
                
                        full_item = f"{vertex_id}=finish"
                        if full_item in finish_int16_to_string_map.values():
                            # If the same string already exists, directly take the hash value
                            hashed_value = [key for key, value in finish_int16_to_string_map.items() if value == full_item][0]
                        else:
                            # Use hash to map the string to a unique integer
                            hashed_value = custom_hash_32bit(full_item, existing_hashes)
                            finish_int16_to_string_map[hashed_value] = full_item
                        vertex_domain.append(hashed_value)
                             

                        # to represent the action without any meaning, when the vertex is not active vertex in that time step
                        full_item = f"{vertex_id}=None"
                        if full_item in None_int16_to_string_map.values():
                            # If the same string already exists, directly take the hash value
                            hashed_value = [key for key, value in None_int16_to_string_map.items() if value == full_item][0]

                            temp_None_QHorizon_int16_to_string_map[hashed_value] = full_item
                        else:
                            # Use hash to map the string to a unique integer
                            hashed_value = custom_hash_32bit(full_item, existing_hashes)
                            None_int16_to_string_map[hashed_value] = full_item
                            temp_None_QHorizon_int16_to_string_map[hashed_value] = full_item
                        vertex_domain.append(hashed_value)
                        
                    Tag_index_inTensor += 1
                    timeStep_domain_list.append(vertex_domain)

            temp_tag.append(Tag_index_inTensor)
            QHorizon_columns_TagindexTensor_list.append(temp_tag)
            QHorizon_columns_vertices_list.append(timeStep)

            # append the timeStep domain to the torch list
            for item in timeStep_domain_list:
                Qhorizon_actionSpace_tensor_list.append(torch.tensor(item, dtype=torch_intType, device=filtered_QHorizon_tensor.device))


        if len(Qhorizon_actionSpace_tensor_list) == 1 and len(QHorizon_columns_vertices_list) == 1 and Qhorizon_actionSpace_tensor_list[0].size()[0] > 1:
            # If there is only one vertex and one controllable variable, return the action space directly
            action_space = Qhorizon_actionSpace_tensor_list[0].clone().detach().unsqueeze(1)
        elif len(Qhorizon_actionSpace_tensor_list) == 0:
            filtered_action_space = torch.empty((0, 1), dtype=torch_intType, device=filtered_QHorizon_tensor.device)
        
        else:

            action_space = torch.cartesian_prod(*Qhorizon_actionSpace_tensor_list)
            filtered_action_space, QHorizon_rows_TagindexTensor_list = filter_ActionTensor_by_columns(action_space, QHorizon_columns_TagindexTensor_list, temp_None_QHorizon_int16_to_string_map, QHorizon_columns_vertices_list, horizion=True)

            ########## below line code will be use less memory, but the result is same as the above code, but time cost is more than the above code ##########
            #filtered_action_space, QHorizon_rows_TagindexTensor_list =  progressive_filter_action_space_gpu_v2(env, Qhorizon_actionSpace_tensor_list, QHorizon_columns_TagindexTensor_list, temp_None_QHorizon_int16_to_string_map, QHorizon_columns_vertices_list, batch_size=100000000, horizon=True)
          
            #print(f"Same tensors: {torch.equal(t_filtered_action_space, filtered_action_space)}")
            #print(f"Same lists: { t_QHorizon_rows_TagindexTensor_list == QHorizon_rows_TagindexTensor_list}")


        # unify results in a list, the three data all are same length and same order relationship
        for i in range(len(QHorizon_columns_vertices_list)):
            temp = []
            temp.append(QHorizon_columns_vertices_list[i])
            temp.append(QHorizon_rows_TagindexTensor_list[i])
            temp.append(QHorizon_columns_TagindexTensor_list[i])
            result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list.append(temp)
        

        return filtered_action_space, result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list



def find_row_index(tensor_2d, tensor_1d):
    """
    Get the row index of the given 1D tensor from a 2D tensor.

    Parameters:
    - tensor_2d: 2D PyTorch tensor.
    - tensor_1d: 1D PyTorch tensor.

    Returns:
    - int: If uniquely exists, return the row index.
    """
    # Check if the input is valid
    if tensor_1d.size(0) != tensor_2d.size(1):
        raise ValueError("The size of the 1D tensor must be the same as the number of columns in the 2D tensor.")

    # Find matching rows
    matches = (tensor_2d == tensor_1d).all(dim=1)
    
    # Get the index of the matching row
    index = torch.nonzero(matches, as_tuple=True)[0]

    # Return unique index
    return index.item()

def find_element_index(tensor_1d, element):
    """
    Get the index of a given element in a 1D tensor.

    Parameters:
    - tensor_1d: 1D PyTorch tensor.
    - element: the element to find.

    Returns:
    - int: the index of the element if it exists uniquely.
    """
    # Find matching elements
    matches = (tensor_1d == element)

    # Get the index of the matching element
    index = torch.nonzero(matches, as_tuple=True)[0]

    # Check if unique
    if index.numel() != 1:
        raise ValueError("The element is not uniquely found in the 1D tensor.")

    # Return unique index
    return index.item()

def initialize_Q_tables(
    subStateSpace_number, sub_stateTag_tensor_list, subActionSpace_tensor_list, sub_stateSpace_vars_sorted_list,
    filtered_QHorizonTag_StateSpace_tensor, filtered_QHorizon_ActionSpace_tensor
):
    Q = {}
    Q_ = {}
    Qkeys = {}
    subStateSpace_vars_name_list = []
    subStateSpace_vertices_name_list = []
    subStateSpace_vars_number_list = []

    # Initialize Q tables for each sub-state space
    for i in range(subStateSpace_number):
        nS = sub_stateTag_tensor_list[i].size(0)
        nA = subActionSpace_tensor_list[i].size(0)

        # Use a dictionary to store the Q table for each sub-state space
        Q[f'{i}'] = torch.zeros((nS, nA), dtype=torch.float32, device=learning_device)
        Q_[f'{i}'] = torch.zeros((nS, nA), dtype=torch.float32, device=learning_device)

        vars_name_list = []
        vertices_name_list = []
        for var in sub_stateSpace_vars_sorted_list[i]:
            vars_name_list.append(var[1])
            vertices_name_list.append(var[0])

        Qkeys[f'{i}'] = vars_name_list
        subStateSpace_vars_name_list.append(vars_name_list)
        subStateSpace_vertices_name_list.append(vertices_name_list)
        subStateSpace_vars_number_list.append(len(vars_name_list))

    # Initialize QH and QH_
    nS, nA = filtered_QHorizonTag_StateSpace_tensor.size(0), filtered_QHorizon_ActionSpace_tensor.size(0)
    QH = torch.zeros((nS, nA), dtype=torch.float32, device=learning_device)
    QH_ = torch.zeros((nS, nA), dtype=torch.float32, device=learning_device)

    return Q, Q_, Qkeys, subStateSpace_vars_name_list, subStateSpace_vertices_name_list, subStateSpace_vars_number_list, QH, QH_

def get_state_and_action_indices_in_Qspaces(
    pmdp_env, active_vertices_of_State_verticesList, state, subStateSpace_number, subStateSpace_vars_number_list,
    subStateSpace_vars_name_list, subStateSpace_vertices_name_list, all_string_to_int16,
    result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list, sub_stateTag_tensor_list):


    Q_state_rowIndexs = []
    Q_actionSpace_columnIndices_list = []
    Q_actionSpace_rowsIndices_list = []

    for i in range(subStateSpace_number):
        vars_state_int16_list = []
        subStateSpace_actionSpace_rowsIndices_list = []
        subStateSpace_actionSpace_columnIndexs_list = []

        for j in range(subStateSpace_vars_number_list[i]):
            var_value = state[subStateSpace_vars_name_list[i][j]]
            if var_value is None:
                # If the variable is 'None', set the string in the tensor and Q table to f'{vertex_id}=None'
                full_item = f"{subStateSpace_vertices_name_list[i][j]}=None"
            else:
                full_item = f"{subStateSpace_vars_name_list[i][j]}={var_value}"
                if var_value == 'NotSelected':
                    # When the variable is 'NotSelected' means in this trajectoiry the the path of the variable is not selected, we need to set the variable to f'{vertex_id}=None' state make compatible with the Q-table
                    # For not wasting the memory, we do not use 'NotSelected' state in the Q-table
                    full_item = f"{subStateSpace_vertices_name_list[i][j]}=None"
            vars_state_int16_list.append(all_string_to_int16[full_item])

        current_subStateSpace_active_verticesIndex_in_result = []
        for j in range(len(result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0])):
            if result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0][j] in active_vertices_of_State_verticesList:
                #current_subStateSpace_active_verticesIndex_in_result.append([result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0][j], j])

                var_name_list = []
                for var in pmdp_env.graph.vertices[result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0][j]].C_v:
                    var_name_list.append(var.name)
                set_var_name_list = set(var_name_list)
                set_subStateSpace_vars_name_list = set(subStateSpace_vars_name_list[i])
                if not set_var_name_list.isdisjoint(set_subStateSpace_vars_name_list):
                    current_subStateSpace_active_verticesIndex_in_result.append([result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0][j], j])
            

        for item in current_subStateSpace_active_verticesIndex_in_result:
            subStateSpace_actionSpace_rowsIndices_list.append(result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][1][item[1]])
            subStateSpace_actionSpace_columnIndexs_list.append(result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][2][item[1]])

        state_tensor = torch.tensor(vars_state_int16_list, dtype=torch_intType, device=learning_device)
        Q_state_rowIndexs.append(find_row_index(sub_stateTag_tensor_list[i], state_tensor))

        #if len(subStateSpace_actionSpace_rowsIndices_list) > 0 and len(subStateSpace_actionSpace_columnIndexs_list) > 0:
        Q_actionSpace_rowsIndices_list.append(subStateSpace_actionSpace_rowsIndices_list)
        Q_actionSpace_columnIndices_list.append(subStateSpace_actionSpace_columnIndexs_list)

    return Q_state_rowIndexs, Q_actionSpace_columnIndices_list, Q_actionSpace_rowsIndices_list

def get_horizon_state_indices(
    result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list, active_vertices_of_State_verticesList,
    all_string_to_int16):
    """
    Find the index and tensor of the active vertices state in the current time step.

    Parameters:
    - result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list: List containing the order, rows and columns of horizon vertices.
    - active_vertices_of_State_verticesList: List of active vertices of the current state.
    - all_string_to_int16: Dictionary of string to int16 mappings.
    - learning_device: Learning device (such as 'cpu' or 'cuda').

    Returns:
    - QH_state_rowIndex: Row index of QH.
    - current_horizon_actionSpace_rowIndices: QH column index, which is also the row index of action space.
    - current_horizon_actionSpace_columnIndices: Column index of action space.
    """
    active_vertices = None
    for i in range(len(result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list)):
        item_list = result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list[i][0].split(',')
        if all(item in active_vertices_of_State_verticesList for item in item_list):
            active_vertices = result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list[i][0]
            horizon_state_int16 = all_string_to_int16[active_vertices]
            current_horizon_stateSpace_rowIndex = i
            current_horizon_actionSpace_rowIndices = result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list[i][1]
            current_horizon_actionSpace_columnIndices = result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list[i][2]
            break
    
    # currrent timeStep will be not for QH, so the QH_state_rowIndex is None
    if active_vertices == None:
        current_horizon_stateSpace_rowIndex = None
        current_horizon_actionSpace_columnIndices = []
        current_horizon_actionSpace_rowIndices = []

    return current_horizon_stateSpace_rowIndex, current_horizon_actionSpace_columnIndices, current_horizon_actionSpace_rowIndices

def generate_action_tensor(
    subStateSpace_number, Q_actionSpace_columnIndices_list, Q_actionSpace_rowsIndices_list, active_vertices_of_State_verticesList, result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list,
    Q_state_rowIndexs, Q, epsilons, e, subActionSpace_tensor_list, QH_state_rowIndex, QH,
    QH_actionSpace_rowsIndices, QH_actionSpace_columnIndices, filtered_QHorizon_ActionSpace_tensor, 
    all_int16_to_string_map , all_string_to_int16, learning_device
):
    """
    Generates the concatenated action_tensor.

    Parameters:
    - subStateSpace_number: number of subStateSpaces.
    - Q_actionSpace_columnIndices_list: list of Q-table action space column indices.
    - Q_actionSpace_rowsIndices_list: list of Q-table action space row indices.
    - active_vertices_of_State_verticesList: list of active vertices of the current state.
    - result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list: list of action space vertex order, rows and columns labels for each subStateSpace.
    - Q_state_rowIndexs: list of Q-table state row indices.
    - Q: Q-table dictionary.
    - epsilons: list of epsilon values.
    - e: index of the current epsilon.
    - subActionSpace_tensor_list: list of subActionSpace tensors.
    - QH_state_rowIndex: QH table state row index.
    - QH_actionSpace_rowsIndices: QH table action space row index.
    - QH_actionSpace_columnIndices: QH table action space column index.
    - filtered_QHorizon_ActionSpace_tensor: Filtered QHorizon action space tensor.
    - learning_device: Learning device (e.g. 'cpu' or 'cuda').

    Returns:
    - action_tensor: The concatenated action tensor.
    """
    action_tensor = torch.empty(0, dtype=torch_intType, device=learning_device)
    values_Q_list = [0] * subStateSpace_number
    values_QH = 0

    subStateSpace_ActionIndice_list = []
    QH_ActionIndice_list = []
    for i in range(subStateSpace_number):
        
        # the vars number in the subStateSpace decide the length of the subActionSpace_tensor_list
        subStateSpace_Var_ActionIndice_list = []
        # # PMDP A complete action involves each subspace and Horizon State, where each subspace (except Horizon StateSpace) involves one or more activation variables. Given a state (i.e., a row in the Q table), each activation variable will have a corresponding column range in the Q table. An action must be selected within this range to form a complete action.
        for j in range(len(Q_actionSpace_columnIndices_list[i])):
            # len(Q_actionSpace_columnIndices_list[i]) > 1 means the subActionSpace more than one variable
            # we need those value of each variable in the subActionSpace, then construct the action tensor
           
            row_start, row_end = Q_actionSpace_rowsIndices_list[i][j][0], Q_actionSpace_rowsIndices_list[i][j][1]
            col_start, col_end = Q_actionSpace_columnIndices_list[i][j][0], Q_actionSpace_columnIndices_list[i][j][1]

            subStateSpace_ActionIndice = select_action(Q_state_rowIndexs[i], row_start, row_end, Q[f'{i}'], epsilons[e])
            subStateSpace_Var_ActionIndice_list.append(subStateSpace_ActionIndice)
            
            if subActionSpace_tensor_list[i].dim() != 1:
                # the subActionSpace more than one variable
                legal_subActionTensor = subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end]
                action_tensor = torch.cat((action_tensor, legal_subActionTensor))
            else:
                # the subActionSpace only have one variable
                legal_subActionTensor = subActionSpace_tensor_list[i][subStateSpace_ActionIndice]
                action_tensor = torch.cat((action_tensor, legal_subActionTensor.unsqueeze(0)), dim=0)
            # get the subStateSpace Q value and store the Q value
            #values_Q += Q[f'{i}'][Q_state_rowIndexs[i], subStateSpace_ActionIndice].item()
            values_Q_list[i] = Q[f'{i}'][Q_state_rowIndexs[i], subStateSpace_ActionIndice].item()
            
        subStateSpace_ActionIndice_list.append(subStateSpace_Var_ActionIndice_list)


    legal_subHorizonActionTensor = torch.empty(0, dtype=torch_intType, device=learning_device)
    if QH_state_rowIndex is not None:
        row_start, row_end = QH_actionSpace_rowsIndices[0], QH_actionSpace_rowsIndices[1]
        col_start, col_end = QH_actionSpace_columnIndices[0], QH_actionSpace_columnIndices[1]

        QH_ActionIndice = select_action(QH_state_rowIndex, row_start, row_end, QH, epsilons[e])
        QH_ActionIndice_list.append(QH_ActionIndice)
        legal_subHorizonActionTensor = filtered_QHorizon_ActionSpace_tensor[QH_ActionIndice, col_start:col_end]
        if legal_subHorizonActionTensor.numel() > 1:
                all_start_with_empty = True
                empty_action_strings = []
                for element in legal_subHorizonActionTensor:
                    element_string = all_int16_to_string_map[element.item()]
                    empty_action_strings.append(element_string)
                    if not element_string.startswith('empty'):
                        all_start_with_empty = False
                        break
                if all_start_with_empty:
                    # Processing logic for all element strings starting with 'empty'
                    for empty_action_string in empty_action_strings:
                        empty_id = empty_action_string.split('=')[0]
                        finish_action_string = f"{empty_id}=finish"
                        if finish_action_string in all_string_to_int16.keys():
                            action_tensor = torch.cat((action_tensor, torch.tensor([all_string_to_int16[finish_action_string]], dtype=torch_intType, device=learning_device)))
                else:
                    # Processing logic for element strings that do not start with 'empty'
                    action_tensor = torch.cat((action_tensor, legal_subHorizonActionTensor))
        else:
            action_tensor = torch.cat((action_tensor, legal_subHorizonActionTensor))
        # get the QH value and store the QH value
        values_QH = QH[QH_state_rowIndex, QH_ActionIndice].item()
        

    return action_tensor, values_Q_list, values_QH, subStateSpace_ActionIndice_list, QH_ActionIndice_list, legal_subHorizonActionTensor


def update_state_and_parse_next(pmdp_env, action_tensor, all_int16_to_string_map, activeVertices_of_State, parallel_asynchronous_prob = 1, frequency_prob_dict = {}, MDP_example_mode = 'PMDP', uncontrol_var_unknow_distributions = None, mode = 'any'):
    """
   Update the state and resolve the next active vertex.

    Parameters:
    - pmdp_env: PMDP environment instance.
    - action_tensor: action tensor.
    - all_int16_to_string_map: int16 to string map.
    - activeVertices_of_State: active vertices of the current state.
    - parallel_asynchronous_prob: parallel asynchronous probability. 0 - 0.9999, if set 1, means the parallel asynchronous is not used.

    Returns:
    - activeVertices_of_State: updated active vertices.
    """
    # parse_and_update_state returns a list of three lists
    current_state_info = pmdp_env.parse_and_update_state(action_tensor, all_int16_to_string_map, {})

    # current_state_info[0] is the updated variable list
    # current_state_info[1] is the waiting vertex list
    # current_state_info[2] is the completed vertex list

    # parse_nexeStates returns the next active vertex and updates the 'current_state' instance
    activeVertices_of_State = pmdp_env.parse_nexeStates(
        activeVertices_of_State, 
        current_state_info[0], 
        current_state_info[1], 
        current_state_info[2],
        uncontrol_var_unknow_distributions, 
        mode,
        parallel_asynchronous_prob,
        frequency_prob_dict,
        MDP_example_mode
    )
    
    return activeVertices_of_State


def calculate_td_error(pmdp_env, active_vertices_of_State_verticesList, subStateSpace_number, subStateSpace_vars_number_list,
                       subStateSpace_vars_name_list, subStateSpace_vertices_name_list, all_string_to_int16,
                       result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list, sub_stateTag_tensor_list,
                       result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list, Q, Q_, QH, QH_, epsilons, e,
                       immediate_rewards_list, gamma, values_Q_list, values_QH):
    """
    Compute TD error.

    Parameters:
    - pmdp_env: PMDP environment instance.
    - active_vertices_of_State_verticesList: list of active vertices of the state at time step t+1.
    - subStateSpace_number: number of substate spaces.
    - subStateSpace_vars_number_list: list of substate space variable numbers.
    - subStateSpace_vars_name_list: list of substate space variable names.
    - subStateSpace_vertices_name_list: list of substate space vertex names.
    - all_string_to_int16: mapping of string to int16.
    - result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list: list of action space vertex order labels.
    - sub_stateTag_tensor_list: list of substate tag tensors.
    - result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list: list of time horizon vertex order labels.
    - Q: fast Q function.
    - Q_: slow Q function.
    - QH: time horizon Q function.
    - QH_: time horizon slow Q function.
    - epsilons: list of epsilon values.
    - e: current epsilon index.
    - immediate_rewards: immediate reward.
    - gamma: discount factor.
    - values_Q: current Q value.
    - values_QH: current time horizon Q value.

    Returns:
    - TD_error: TD error.
    - replay_buffer_data: Data stored in the replay buffer.
    """
    # TD error calculation uses the slow updated Q_ function and QH_ function
    # All in One TD error calculation
    #next_values_Q_ = 0
    #next_values_QH_ = 0

    next_values_Q_list= [0] * subStateSpace_number
    # the column in Q (subStateSpace), but it is the row in subActionSpace
    #next_values_Q_columnIndices_list = []
    next_values_QH_list= [] 
    next_state = pmdp_env.current_state.copy()

    # Get the indices of next_state
    Q_state_rowIndexs, Q_actionSpace_columnIndices_list, Q_actionSpace_rowsIndices_list = get_state_and_action_indices_in_Qspaces(
        pmdp_env, active_vertices_of_State_verticesList, next_state, subStateSpace_number, subStateSpace_vars_number_list,
        subStateSpace_vars_name_list, subStateSpace_vertices_name_list, all_string_to_int16,
        result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list, sub_stateTag_tensor_list)

    # Get the index of the horizon state
    QH_state_rowIndex, QH_actionSpace_columnIndices, QH_actionSpace_rowsIndices = get_horizon_state_indices(
        result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list, active_vertices_of_State_verticesList, all_string_to_int16)

    # Use slow Q_ function to get Q value of next_state, but action is selected using fast Q function
    for i in range(subStateSpace_number):
        #next_values_Q_columnIndices_list.append([])
        for j in range(len(Q_actionSpace_rowsIndices_list[i])):
            row_start, row_end = Q_actionSpace_rowsIndices_list[i][j][0], Q_actionSpace_rowsIndices_list[i][j][1]
            subStateSpace_ActionIndice = select_action(Q_state_rowIndexs[i], row_start, row_end, Q[f'{i}'], epsilons[e], only_greedy=True)
            #next_values_Q_ += Q_[f'{i}'][Q_state_rowIndexs[i], subStateSpace_ActionIndice].item()
            next_values_Q_list[i] = Q_[f'{i}'][Q_state_rowIndexs[i], subStateSpace_ActionIndice].item()
            #next_values_Q_columnIndices_list[i] = [row_start, row_end]

    if QH_state_rowIndex is not None:
        row_start, row_end = QH_actionSpace_rowsIndices[0], QH_actionSpace_rowsIndices[1]
        QH_ActionIndice = select_action(QH_state_rowIndex, row_start, row_end, QH, epsilons[e], only_greedy=True)
        #next_values_QH_ += QH_[QH_state_rowIndex, QH_ActionIndice].item()
        next_values_QH_list.append(QH_[QH_state_rowIndex, QH_ActionIndice].item())

    # all in one TD error calculation
    #TD_target = immediate_rewards + gamma * (next_values_Q_ + next_values_QH_)
    #TD_error = TD_target - (values_Q + values_QH)

    TD_error_list = [0] * (subStateSpace_number + 1)
    for i in range(subStateSpace_number):
        # Some time, the transition of PMDP is not related to the subStateSpace (rather, about QH), so 'values_Q_list' will not have the value of the subStateSpace, we need to add 0 to the list for compatibility of the PMDP learning
        while len(values_Q_list) <= i:
            values_Q_list.append(0)
        # similiar with the 'values_Q_list', we need to add 0 to the list for compatibility of the PMDP learning
        while len(next_values_Q_list) <= i:
            next_values_Q_list.append(0)
        #for j in range(len(Q_actionSpace_rowsIndices_list[i])):
        # 'immediate_rewards_list[0]' is the sum of HC and SC rewards in the list
        TD_target = immediate_rewards_list[0][i] + gamma * next_values_Q_list[i]
        TD_error_value = TD_target - values_Q_list[i]
        TD_error_list[i] = TD_error_value
            
    if QH_state_rowIndex is not None:
        # use -0.1 as the part of reward, to represent the negative reward strategy of hoping to reach the end of the trajectory with fast speed
        # '(-0.1 + immediate_rewards_list[3]) to represent the reward strategy of hoping to reach the end of the trajectory with fast speed, while hoping select suitable wait or finish action with the highest constraints reward
        # 'immediate_rewards_list[3]' is the All_Rewards_All_subStates for all HC and SC rewards
        TD_target_horizon = (-0.1 + immediate_rewards_list[3]) + gamma * next_values_QH_list[0]
        TD_error_horizon = TD_target_horizon - values_QH
        TD_error_list[-1] = (TD_error_horizon)
    else:
        # current state transition is not related to the QH, so the TD error of QH is 0
        TD_error_list[-1] = 0

    
    # Prepare data to be stored in the replay buffer
    # here data is not include the horizon state
    next_states_indices_list =  [Q_state_rowIndexs[i]  for i in range(subStateSpace_number)]
   

    return TD_error_list, next_states_indices_list


def update_fast_Q(Q, Q_state_rowIndexs, Q_state_ActionIndice_list, weight, alphas, e, TD_error):
    """
    Updates the columns of a specific row in a 2D tensor, taking advantage of PyTorch's parallel computing.

    Parameters:
    - Q: 2D tensor.
    - Q_state_rowIndexs: list of row indices.
    - Q_state_ActionIndice_list: list of column indices.
    - alphas: list of learning rates.
    - weight: weights in the replay buffer.
    - e: current epsilon index.
    - TD_error: TD error.

    Returns:
    - The updated 2D tensor.
    """
    # Convert row and column indices to tensors
    #row_indices = torch.tensor(Q_state_rowIndexs, dtype=torch.long)
    #col_indices = [torch.tensor(cols, dtype=torch.long) for cols in Q_state_ActionIndice_list]

    # Convert Q_state_rowIndexs to numpy.ndarray if it is not already
    if not isinstance(Q_state_rowIndexs, np.ndarray):
        Q_state_rowIndexs = np.array(Q_state_rowIndexs, dtype=np.int64)

    # Directly convert numpy.ndarray to PyTorch tensor
    row_indices = torch.tensor(Q_state_rowIndexs, dtype=torch.long)

    # Convert Q_state_ActionIndice_list to numpy.ndarray and then to tensor
    Q_state_ActionIndice_np = np.array(Q_state_ActionIndice_list, dtype=np.int64)
    col_indices = torch.tensor(Q_state_ActionIndice_np, dtype=torch.long)

    # Directly update the Q tensor
    for i, row_index in enumerate(row_indices):
        # col_indices[i] is a tensor or list containing the column indices to access.
        # For example, if col_indices[i] is [0, 2], it means accessing columns 0 and 2 of Q.
        # The formula.15 in the paper
        Q[row_index, col_indices[i]] += alphas[e] * TD_error * weight

    return Q

def update_eligibility_trace(eligibility_trace, Q_state_rowIndexs, Q_state_ActionIndice_list):
    """
    Returns:
    - The updated 2D tensor.
    """
    # Convert row and column indices to tensors
    #row_indices = torch.tensor(Q_state_rowIndexs, dtype=torch.long)
    #col_indices = [torch.tensor(cols, dtype=torch.long) for cols in Q_state_ActionIndice_list]

    # Convert Q_state_rowIndexs to numpy.ndarray if it is not already
    if not isinstance(Q_state_rowIndexs, np.ndarray):
        Q_state_rowIndexs = np.array(Q_state_rowIndexs, dtype=np.int64)

    # Directly convert numpy.ndarray to PyTorch tensor
    row_indices = torch.tensor(Q_state_rowIndexs, dtype=torch.long)

    # Convert Q_state_ActionIndice_list to numpy.ndarray and then to tensor
    Q_state_ActionIndice_np = np.array(Q_state_ActionIndice_list, dtype=np.int64)
    col_indices = torch.tensor(Q_state_ActionIndice_np, dtype=torch.long)

    # Directly update the Q tensor
    for i, row_index in enumerate(row_indices):
        # col_indices[i] is a tensor or list containing the column indices to access.
        # For example, if col_indices[i] is [0, 2], it means accessing columns 0 and 2 of Q.
        # The formula.15 in the paper
        eligibility_trace[row_index, col_indices[i]] += 1

    return eligibility_trace

def update_slow_Q(slow_Q, fast_Q, Q_state_rowIndexs, Q_state_ActionIndice_list, eta):
    """
    Updates the columns of a specific row in a 2D tensor, taking advantage of PyTorch's parallel computing.

    Parameters:
    - slow_Q: The slow 2D tensor to update.
    - fast_Q: The fast 2D tensor to update.
    - Q_state_rowIndexs: A list of row indices.
    - Q_state_ActionIndice_list: A list of column indices.
    - eta: The update coefficient.

    Returns:
    - The updated slow 2D tensor.
    """
    # Convert row indices and column indices to tensors
    #row_indices = torch.tensor(Q_state_rowIndexs, dtype=torch.long)
    #col_indices = [torch.tensor(cols, dtype=torch.long) for cols in Q_state_ActionIndice_list]

    # Convert Q_state_rowIndexs to numpy.ndarray if it is not already
    if not isinstance(Q_state_rowIndexs, np.ndarray):
        Q_state_rowIndexs = np.array(Q_state_rowIndexs, dtype=np.int64)

    # Directly convert numpy.ndarray to PyTorch tensor
    row_indices = torch.tensor(Q_state_rowIndexs, dtype=torch.long)

    # Convert Q_state_ActionIndice_list to numpy.ndarray and then to tensor
    Q_state_ActionIndice_np = np.array(Q_state_ActionIndice_list, dtype=np.int64)
    col_indices = torch.tensor(Q_state_ActionIndice_np, dtype=torch.long)

    # Directly update the slow_Q tensor
    for i, row_index in enumerate(row_indices):
        # col_indices[i] is a tensor or list containing the column indices to access.
        # For example, if col_indices[i] is [0, 2], it means accessing columns 0 and 2 of slow_Q.
        # The formula.16 in the paper
        slow_Q[row_index, col_indices[i]] = eta * fast_Q[row_index, col_indices[i]] + (1 - eta) * slow_Q[row_index, col_indices[i]]

    return slow_Q


def perform_priority_experience_replay(subStateSpace_number, replay_buffer_list, n_warmup_batches, step_count, replay_interval, epsilons, e, Q, Q_, replay_buffer_actionDict_list, gamma, alphas, eta):
    """
    Perform priority experience replay if enough samples are stored.

    Parameters:
    - subStateSpace_number: Number of sub-state spaces.
    - replay_buffer_list: List of replay buffers for each sub-state space.
    - n_warmup_batches: List of warmup batch numbers for each sub-state space.
    - step_count: List of step counts for each sub-state space.
    - replay_interval: Interval for replay.
    - epsilons: Epsilon values for sampling.
    - e: Current episode number.
    - Q: Dictionary of Q-tables for each sub-state space.
    - Q_: Dictionary of slow-updating Q-tables for each sub-state space.
    - replay_buffer_actionDict_list: List of action dictionaries for each replay buffer.
    - gamma: Discount factor.
    - alphas: Learning rates.
    - eta: Update rate for slow Q-table.
    """
    td_errors_for_subStateSpace_list = []
    for i in range(subStateSpace_number):
        td_errors_for_subStateSpace_list.append([])

        if len(replay_buffer_list[i]) >= (replay_buffer_list[i].batch_size * n_warmup_batches[i]) and step_count[i] % replay_interval == 0:
            idxs, weights, samples = replay_buffer_list[i].sample(epsilons, e)
            # Process the sampled experience of the subStateSpace
            states, actions, rewards, next_states, dones = samples
            
            # the length of the 'idxs' is same as the length of the 'weights', 'samples'
            # the 'actions' is store the index of dict, it same with the value of idxs
            for j, idx in enumerate(idxs):
                state, action, reward, next_state, done = states[j], replay_buffer_actionDict_list[i][idx.item()], rewards[j], next_states[j], dones[j]
                # The next state Q value should be calculated by the Q_ (i.e., the slow updating Q value)
                next_state_value_Q_subStateSpace = Q_[f'{i}'][next_state, :].max().item()

                # Since each subspace (except Horizon StateSpace) involves one or more activation variables, after a given state (i.e., a row of the Q table), each activation variable will have a corresponding column range in the Q table, and an action must be selected within the range to form a complete action for the subspace
                # However, for Q, although we need to select multiple columns in the same row to form an action_tensor to PMDP
                # However, when we need to calculate the TD target, that is, we need the state-action_tensor pair in Q, we only need to select one of the multiple columns, because we store the same Q value on these columns, which represents the Q value of the action_tensor in this state
                if len(action) > 1:
                    # There are multiple columns, i.e., multiple actions, only one action needs to be selected, any one is fine (here, use action[0]), because the Q values stored in the Q table are the same
                    value_Q_subStateSpace = Q[f'{i}'][state, action[0]].item()
                else:
                    value_Q_subStateSpace = Q[f'{i}'][state, action].item()

                # Calculate the TD errors for each sample
                td_target = reward + gamma * next_state_value_Q_subStateSpace * (not done)
                td_error = td_target.item() - value_Q_subStateSpace
                Q[f'{i}'] = update_fast_Q(Q[f'{i}'], [state], [action], weights[j].item(), alphas, e, td_error)
                # The 'weights[j]' has been applied to Q[f'{i}'] in the 'update_fast_Q' function
                # So the 'weights[j]' is not used in the 'update_slow_Q' function directly, but it is actually used by the 'update_fast_Q' function
                # Because the 'update_slow_Q' function is used to update the Q_[f'{i}'] with the Q[f'{i}'] in the 'update_fast_Q' function
                Q_[f'{i}'] = update_slow_Q(Q_[f'{i}'], Q[f'{i}'], [state], [action], eta)
                td_errors_for_subStateSpace_list[i].append(td_error)
                
            # Update the priorities of the samples
            replay_buffer_list[i].update(idxs, td_errors_for_subStateSpace_list[i])
            # Add a fake counter so that experience replay is not performed again before the next experience replay cycle
            # That is, if step_count[i] % replay_interval == n, if PMDP does not have any state transitions related to any subStateSpace in the subsequent 1 step or more, it is ensured that experience replay will not be performed again due to step_count[i] % replay_interval == n
            step_count[i] += 1


##### This function is the table-based Cdouble Q-learning algorithm for PMDPs #####
##### It not use for the current version, but it is useful for the future version #####
##### 20250721 ####
def Cdouble_Qlearning(file_path: str, log_file_path = '', PER_mode= False, CET_mode = False, punish_mode = False, gamma=0.8, init_alpha=0.1, min_alpha=1e-4, alpha_decay_ratio=0.9,
                                init_epsilon=1.0, min_epsilon=0.01, epsilon_decay_ratio=0.9,
                                n_episodes=2000, max_samples_buffer = 20000, n_warmup_batches = [100, 50], trace_decay=0.9, constraint_precision=0.01, eta=0.5, whc=0.5, wsc=0.5):
    
    Cdouble_Qlearning_tensor_int = 16
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes, log_start = -0.1) 
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes,  log_start = -0.1)


    graph = parse_bpmn(file_path)
    pmdp_env = PMDPEnvironment(graph)
    pmdp_env.debug_mode = False
    horizon_space = pmdp_env.get_horizon_space()


    if log_file_path != '':

        if 'CSSC' in file_path: 
            max_bound_dict, min_bound_dict = calculate_max_min_from_txt(log_file_path)
            for key in max_bound_dict.keys():
                min_val, max_val = min_bound_dict[key], max_bound_dict[key]
                mdp_update_dynamic_bounds(pmdp_env, key, min_val, [min_val, max_val])
                mdp_update_dynamic_bounds(pmdp_env, key, max_val, [min_val, max_val])


    filtered_QHorizonTag_StateSpace_tensor = process_horizon(pmdp_env, horizon_space, horizon_int16_to_string_map, existing_hashes, learning_device)
    filtered_QHorizon_ActionSpace_tensor, result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list = get_QHorizon_actionSpace_columns(pmdp_env, horizon_space, filtered_QHorizonTag_StateSpace_tensor)

    sub_stateSpace_vars_sorted_list, sub_stateTag_tensor_list = get_states_for_Q_rows(pmdp_env, horizon_space, graph)
    subActionSpace_tensor_list, result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list = get_actionSpace_for_Q_columns(pmdp_env, sub_stateSpace_vars_sorted_list)  

    # get the number of (Q-tables), that is, the number of subStateSpaces
    subStateSpace_number = len(sub_stateTag_tensor_list)

    # Q and Q_ are dictionaries to store Q-tables, enable automatic naming of Q-tables
    # Qkeys is a dictionary (for persistent) to store the keys of the Q and Q_ dictionaries about each subStateSpace consisting of what variables

    # For the Q and Q_, the toatl rows of subStateSpace is the number of state in the subStateSpace, that is, the rows of corresponding sub_stateTag_tensor in 'sub_stateTag_tensor_list'
    # For the Q and Q_, the total columns of subStateSpace is the number of state in the subActionSpace (i.e., rows of subActionSpace), that is, the rows of corresponding subActionSpace_tensor in 'subActionSpace_tensor_list'
    # Additional, each action in the subStateSpace has a corresponding tensor data with legal columns (1 or more) in the subActionSpace

    # For the QH and QH_, having the similar structure with Q and Q_, the rows of QH and QH_ is the number of state in the 'filtered_QHorizonTag_StateSpace_tensor'
    # and the columns of QH and QH_ is the number of state in the filtered_QHorizon_ActionSpace_tensor
    # each action in the 'filtered_QHorizonTag_StateSpace_tensor' has a corresponding tensor data with legal columns (1 or more) in the filtered_QHorizon_ActionSpace_tensor
    Q, Q_, Qkeys, subStateSpace_vars_name_list, subStateSpace_vertices_name_list, subStateSpace_vars_number_list, QH, QH_ = initialize_Q_tables(subStateSpace_number, sub_stateTag_tensor_list, 
                                                                                                                                                subActionSpace_tensor_list, sub_stateSpace_vars_sorted_list,
                                                                                                                                                filtered_QHorizonTag_StateSpace_tensor, filtered_QHorizon_ActionSpace_tensor)


    # The 'subState_HC_weights_list' and 'subState_SC_weights_list' have been computed with the 'whc' and 'wsc' weights
    # The 'subState_HC_weights_list' and 'subState_SC_weights_list' will store the HC and SC weights of each subStateSpace with the same order of the subStateSpace
    HC_list, SC_list, subState_HC_weights_list, subState_SC_weights_list, whc, wsc = classify_constraints_by_substateSpace(pmdp_env, subStateSpace_vars_name_list, whc, wsc)
    
    HC_name_list = []
    SC_name_list = []

    # Initialize replay buffer for each subStateSpace
    replay_buffer_list = []
    replay_buffer_actionDict_list = []
    next_state_Q_columnIndicesDict_list = []
    replay_interval = 2
    step_count = [0] * subStateSpace_number

    CET_list = []
    eligibility_list = []
    ConstraintState_list = []

    subCET_list = []

    CET_HC_count_list = []
    CET_SC_count_list = []

    HC_weight_list = [0] * subStateSpace_number
    SC_weight_list = [0] * subStateSpace_number

    for i in range(subStateSpace_number):
        replay_buffer_list.append(PrioritizedReplayBuffer(max_samples = max_samples_buffer, batch_size=16))
        replay_buffer_actionDict_list.append({})
        next_state_Q_columnIndicesDict_list.append({})
                   
        # Use the sorted() function to perform stable sorting to ensure that the order of each constraint in the constraint state list is fixed
        HC_list[i] = sorted(HC_list[i], key=lambda x: x.name)
        SC_list[i] = sorted(SC_list[i], key=lambda x: x.name)
  
        HC_name_list.append([hc.name for hc in HC_list[i] if hc.weight > 0])
        SC_name_list.append([sc.name for sc in SC_list[i] if sc.weight > 0])

        for hc in HC_list[i]:
            HC_weight_list[i] += hc.weight

        for sc in SC_list[i]:
            SC_weight_list[i] += sc.weight

        
        total_weight = HC_weight_list[i] * whc + SC_weight_list[i] * wsc

        # Normalize HC_weight_list[i]
        if total_weight != 0:
            HC_weight_list[i] = (HC_weight_list[i] * whc) / total_weight

        # Normalize SC_weight_list[i]
        if total_weight != 0:
            SC_weight_list[i] = (SC_weight_list[i] * wsc) / total_weight



        if CET_mode:
            # get the size of the subStateSpace Q-table
            nS = sub_stateTag_tensor_list[i].shape[0]
            nA = subActionSpace_tensor_list[i].shape[0]

            # Initialize the Constraint-Aware Expected Eligibility Traces (CET) and regular eligibility traces
            CET_list.append(ConstraintAwareExpectedEligibilityTraces(nS, nA))
            eligibility_list.append(torch.zeros((nS, nA), dtype=torch.float32, device=learning_device))

            subCET_list.append(torch.zeros((nS, nA), dtype=torch.float32, device=learning_device))

            # Initialize constraint state management
            ConstraintState_list.append(ConstraintState( HC_list[i] + SC_list[i], beta = constraint_precision))

            CET_SC_count_list.append(0)
            CET_HC_count_list.append(0)
        else:
            # For compatibilit, we need to add the empty list to the 'ConstraintState_list'
            ConstraintState_list.append([])

    all_int16_to_string_map = {}
    all_int16_to_string_map.update(int16_to_string_map)
    all_int16_to_string_map.update(wait_int16_to_string_map)
    all_int16_to_string_map.update(None_int16_to_string_map)
    all_int16_to_string_map.update(finish_int16_to_string_map)
    all_int16_to_string_map.update(horizon_int16_to_string_map)
    # Swap keys and values
    all_string_to_int16 = {value: key for key, value in all_int16_to_string_map.items()}


    accumulate_HC_SC_rewards_in_each_trajectory = []
    accumulate_HC_rewards_in_each_trajectory = []
    accumulate_SC_rewards_in_each_trajectory = []

    start_vertex = next((v for v in horizon_space if isinstance(v, str) and v.startswith('StartEvent')), None)
    for e in tqdm(range(n_episodes), leave=False):
        done = False
        # each episode begining, set the eligibility traces to zero, and reset the ConstraintState
        if CET_mode:
            SatisfiedHC_list = []
            SatisfiedSC_list = []
            for i in range(subStateSpace_number):
                eligibility_list[i].zero_()
                ConstraintState_list[i].reset(HC_list[i] + SC_list[i])
                SatisfiedHC_list.append([])
                SatisfiedSC_list.append([])

        # get the start event of the BPMN model, that is the initial active vertex (initial state) of the PMDP model
        #activeVertices_of_State = next((v for v in horizon_space if isinstance(v, str) and v.startswith('StartEvent')), None)
        activeVertices_of_State = start_vertex

        # Get the active vertices of the state and follow the order of the sub-state space Q table
        active_vertices_of_State_verticesList = [start_vertex]

        # the 'rowsIndices' is the column index of the action in the state, i.e., 'Q,Q_'
        time_step = 0
        
        # The item of 'accumulate_rewards_list' represent the accumulate rewards list of the subStateSpace,
        # each item of 'accumulate_rewards_list' consist of [0] is accumulate rewards of the 'HC and SC' rewards of the subStateSpace, [1] is the accumulate rewards of the 'HC' rewards of the subStateSpace, [2] is the accumulate rewards of the 'SC' rewards of the subStateSpace
        accumulate_rewards_list = [[0, 0, 0] for _ in range(subStateSpace_number)]
        while not done:
            # get the current state of the PMDP
            state = pmdp_env.current_state

            if pmdp_env.debug_mode:
                print(f"------------------------------------Current Time-Step: {time_step}-----------------------------------------------")
                print(f"Current active vertices: {active_vertices_of_State_verticesList}")
                Curret_active_vertice_with_names = []
                for vertex_id in active_vertices_of_State_verticesList:
                    vertex_name = pmdp_env.graph.vertices[vertex_id].elem_name
                    if vertex_id.startswith('empty'):
                            vertex_name = vertex_id
                    Curret_active_vertice_with_names.append(f"{vertex_name}:{vertex_id}")
                print("Curret_active_vertices with Name: ", Curret_active_vertice_with_names) 
                print(f"Current state: {state}")
                time_step += 1

            # the 'Q_actionSpace_rowsIndices_list' is the column index of the action in the state, i.e., 'Q,Q_' that have same structure and size 
            Q_state_rowIndexs, Q_actionSpace_columnIndices_list, Q_actionSpace_rowsIndices_list = get_state_and_action_indices_in_Qspaces(pmdp_env, active_vertices_of_State_verticesList, state, 
                                                                                                                                    subStateSpace_number, subStateSpace_vars_number_list,
                                                                                                                                    subStateSpace_vars_name_list, subStateSpace_vertices_name_list, 
                                                                                                                                    all_string_to_int16,result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list, 
                                                                                                                                    sub_stateTag_tensor_list)

            # the 'QH_actionSpace_rowsIndices' is the column index of the action in the state, i.e., 'QH'
            QH_state_rowIndex, QH_actionSpace_columnIndices, QH_actionSpace_rowsIndices = get_horizon_state_indices(result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list, 
                                                                                                                       active_vertices_of_State_verticesList,all_string_to_int16)
            
            # Generate a complete action tensor for PMDP by Q and QH
            # The 'values_Q_list (for each subStatSpace)' and 'values_QH' are the sum of Q-values of the pairs of 'state-action' in 'Q,Q_'
            action_tensor, values_Q_list, values_QH, subStateSpace_ActionIndice_list, QH_ActionIndice_list, _ = generate_action_tensor(subStateSpace_number, Q_actionSpace_columnIndices_list, Q_actionSpace_rowsIndices_list, active_vertices_of_State_verticesList, result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list,
                                        Q_state_rowIndexs, Q, epsilons, e, subActionSpace_tensor_list, QH_state_rowIndex, QH,
                                        QH_actionSpace_rowsIndices, QH_actionSpace_columnIndices, filtered_QHorizon_ActionSpace_tensor, all_int16_to_string_map, all_string_to_int16, learning_device)
            


            # for 'CET' apply in the 'priority experience replay'
            current_activeVertices_of_State = activeVertices_of_State
            cet_pmdp_env = copy.deepcopy(pmdp_env)

            # Make action and get 'the next state of the PMDP'
            # the 'activeVertices_of_State' is a horizon item, it is a string, tupple, or nested tuple of strings
            next_activeVertices_of_State = update_state_and_parse_next(
                pmdp_env, 
                action_tensor, 
                all_int16_to_string_map, 
                activeVertices_of_State
            )
          
            # for next time step, update the active vertices of the state
            activeVertices_of_State = next_activeVertices_of_State

            # the first parameter is the weight for the hard constraints, and the second parameter is the weight for the soft constraints
            # immediate_rewards = immediate_rewards_HC_rewards + immediate_rewards_SC_rewards
            #immediate_rewards, immediate_rewards_HC_rewards, immediate_rewards_SC_rewards = pmdp_env.calculate_reward(0.5, 0.5)
            # here is not related to the horizonSpace reward
            # [0, ] is sum of the 'HC and SC' rewards list of the subStateSpace, [1, ] is the sum of the 'HC' rewards list of the subStateSpace, [2, ] is the sum of the 'SC' rewards list of the subStateSpace, [3] is the 'All_Rewards_All_subStates'
            
            #immediate_rewards_list, columns are subStateSpace indexes
            #accumulate_rewards_list, rows are subStateSpace indexes
            immediate_rewards_list, accumulate_rewards_list, ConstraintState_list, evaluated_constraintsName_list = pmdp_env.calculate_reward(whc, wsc, HC_list, SC_list, ConstraintState_list, punish_mode, CET_mode, accumulate_rewards_list, subStateSpace_number = subStateSpace_number)

            # check if the current state is a terminal state
            if activeVertices_of_State is None:
                # Check if the 'active_vertices' represent any vertex in the edges as a target but not as a source
                # i.e., the end event of bpmn model
                done = True
            else:
                active_vertices_of_State_verticesList = pmdp_env.convert_horizon_item_to_list(next_activeVertices_of_State)
            
            if not done:
                TD_error_list, next_states_indices_list = calculate_td_error(pmdp_env, active_vertices_of_State_verticesList, subStateSpace_number, subStateSpace_vars_number_list,
                                            subStateSpace_vars_name_list, subStateSpace_vertices_name_list, all_string_to_int16,
                                            result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list, sub_stateTag_tensor_list,
                                            result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list, Q, Q_, QH, QH_, epsilons, e,immediate_rewards_list, gamma, values_Q_list, values_QH)

                ######################################### Store the experience in replay buffers #########################################
                # The data consist of [state, action, reward, next_state, done], in which varname is not related to the code
                for i in range(subStateSpace_number):
                    # Update the constraint state (CS) for each subStateSpace in here for efficiency, utilize below 'for cycle' to update the CS for each subStateSpace
                    # This can be executed after the calculation of rewards, because the get the immediate rewards of the subStateSpace which means the constraints is satisfied or not, the ConstraintState_value has been updated
                    subCET_idList = []

                    ######################################### Update the CET and Q-tables #########################################
                    if CET_mode:
                        # Update Constraint State
                        ConstraintState_list[i].update_CS()    
                        if len(subStateSpace_ActionIndice_list[i]) > 0:
                            # If current state has action, then update the eligibility traces of the current episode
                            # Update Eligibility Traces of current episode, that is , +1 for the current state-action pair
                            eligibility_list[i] = update_eligibility_trace(eligibility_list[i], [Q_state_rowIndexs[i]], subStateSpace_ActionIndice_list[i])

                        # remove the duplicate constraints in the 'evaluated_constraintsName_list'
                        evaluated_constraintsName_list[i] = list(set(evaluated_constraintsName_list[i]))

                        
                       
                        for cName in evaluated_constraintsName_list[i]:
                            if 'hc' in cName:
                                SatisfiedHC_list[i].append(cName)
                            if 'sc' in cName:
                                SatisfiedSC_list[i].append(cName)
                        
                        HC_all_satifisfied = set(SatisfiedHC_list[i]) == set(HC_name_list[i])

                        if accumulate_rewards_list[i][1] == 1:
                            pass
                        if HC_all_satifisfied and len(SatisfiedHC_list[i]) != 0:
                            # If all (relation and logic HC) SCs are satisfied for the subStateSpace i,  then update the CET
                            for cName in SatisfiedHC_list[i]:
                                CET_id = cName + ':' + f'{ConstraintState_list[i].constraint_values[cName]}'
                                eligibility_trace_weight = ConstraintState_list[i].constraint_values[cName]   
                                         
                                #if eligibility_trace_weight != 0:
                                CET_list[i].update(CET_id, eligibility_list[i] * eligibility_trace_weight * HC_weight_list[i] )
                                #CET_list[i].update(CET_id, eligibility_list[i] * HC_weight_list[i])
                                subCET_idList.append(CET_id)
                                subCET_list[i] += (CET_list[i].z[CET_id] * 1/len(HC_name_list[i]))
                            CET_HC_count_list[i] += 1

                        
                        SC_all_satifisfied = set(SatisfiedSC_list[i]) == set(SC_name_list[i])
                        if len(SatisfiedSC_list[i]) != 0:
                            if SC_all_satifisfied and (e >= 50):
                                # If all (relation and logic SC) SCs are satisfied for the subStateSpace i, and the value of all (optimization SC) SCs are large than '0.8',  then update the CET
                                #SatisfiedSC_list[0].sort()
                                #tag = False
                                for cName in SatisfiedSC_list[i]:
                                    CET_id = cName + ':' + f'{ConstraintState_list[i].constraint_values[cName]}'


                                    eligibility_trace_weight = ConstraintState_list[i].constraint_values[cName]   
                                    
                                    #if eligibility_trace_weight != 0:

                                    CET_list[i].update(CET_id, eligibility_list[i] * eligibility_trace_weight * SC_weight_list[i])
                                    #CET_list[i].update(CET_id, eligibility_list[i] * SC_weight_list[i])
                                    subCET_idList.append(CET_id)
                                    subCET_list[i] += CET_list[i].z[CET_id] * 1/len(SC_name_list[i])

                                    tag = True
                                if tag:
                                    CET_SC_count_list[i] += 1


                        # Because PMDP use dynamic max-min bounds for the constraints, so set after 50 episodes, the dynamic max-min bounds will be updated to more accurate then may to update the CET
                        if (HC_all_satifisfied and  (e >= 50) and len(SatisfiedHC_list[i]) != 0) or (SC_all_satifisfied and  (e >= 50) and len(SatisfiedSC_list[i]) != 0):
                            # Update the Q-tables with the CET
                            Q[f'{i}'] += alphas[e] * TD_error_list[i] * subCET_list[i]
                            Q_[f'{i}'] = eta * Q[f'{i}'] + (1 - eta) *  Q_[f'{i}']

                            SatisfiedHC_list[i] = []
                            SatisfiedSC_list[i] = []
                            subCET_list[i].zero_()

                            #CET_HC_count_list[i] += 1
                            #CET_SC_count_list[i] += 1
                       


                        # eligibility trace 衰减
                        eligibility_list[i] *= gamma * trace_decay

                    ############## Add experience to replay buffer ##############
                    if PER_mode and len(subStateSpace_ActionIndice_list[i]) > 0:
                        # the sample of 'PriorityReplayBuffer' cannot be like 'list', the action in PMDP could be 'list' so we need to store the action in the 'dict'
                        replay_buffer_actionDict_list[i].update({replay_buffer_list[i].next_index: subStateSpace_ActionIndice_list[i]})
  
                        #next_state_Q_columnIndicesDict_list[i].update({replay_buffer_list[i].next_index: next_state_Q_columnStarEndIndices_list[i]})
                        # only current subStateSpace has the action (i.e., the action_tensor has action about this subState), then store the data
                        # otherwise, here is no data produced when len(subStateSpace_ActionIndice_list[i]) == 0
                        # The each experience is stored in the replay buffer as a tuple of '(state, action, reward, next_state, done)'
                        buffer_index = replay_buffer_list[i].next_index
                        replay_buffer_list[i].store([Q_state_rowIndexs[i], buffer_index, immediate_rewards_list[0][i], next_states_indices_list[i], done])
                        step_count[i] += 1


                #perform_priority_experience_replay(subStateSpace_number, replay_buffer_list, n_warmup_batches, step_count, replay_interval, epsilons, e, Q, Q_, replay_buffer_actionDict_list, gamma, alphas, eta)
                

                if PER_mode:
                    ######################################### Perform experience replay if enough samples are stored #########################################
                    td_errors_for_subStateSpace_list = [] 
                    for i in range(subStateSpace_number): 
                        td_errors_for_subStateSpace_list.append([])
                        if len(replay_buffer_list[i]) >= (replay_buffer_list[i].batch_size * n_warmup_batches[i]) and step_count[i] % replay_interval == 0:
                            idxs, weights, samples = replay_buffer_list[i].sample(epsilons, e)
                            # Process the sampled experience of the subStateSpace
                            states, _, rewards, next_states, dones = samples 
                            
                            # the length of the 'idxs' is same as the length of the 'weights', 'samples'
                            # the 'actions' is store the index of dict, it same with the value of idxs
                            for j, idx in enumerate(idxs):
                                state, action, reward, next_state, done = states[j], replay_buffer_actionDict_list[i][idx.item()], rewards[j], next_states[j], dones[j]
                                # The next state Q value should be calculated by the Q_ (i.e., the slow updating Q value)
                                #if next_state_Q_columnIndicesDict_list[i][idx.item()] == []:
                                    #next_state_value_Q_subStateSpace = Q_[f'{i}'][next_state, :].max().item()
                                #else:
                                    #colStart,  colEnd = next_state_Q_columnIndicesDict_list[i][idx.item()]
                                    #next_state_value_Q_subStateSpace = Q_[f'{i}'][next_state, colStart:colEnd].max().item()
                                next_state_value_Q_subStateSpace = Q_[f'{i}'][next_state, :].max().item()

                                # Since each subspace (except Horizon StateSpace) involves one or more activation variables, after a given state (i.e., a row of the Q table), each activation variable will have a corresponding column range in the Q table, and an action must be selected within the range to form a complete action for the subspace
                                # However, for Q, although we need to select multiple columns in the same row to form an action_tensor to PMDP
                                # However, when we need to calculate the TD target, that is, we need the state-action_tensor pair in Q, we only need to select one of the multiple columns, because we store the same Q value on these columns, which represents the Q value of the action_tensor in this state
                                if len(action) > 1:
                                    # There are multiple columns, that is, multiple actions. You only need to select one action, any one (here, use action[0]), because the Q values stored in the Q table are the same
                                    value_Q_subStateSpace = Q[f'{i}'][state, action[0]].item()
                                else:
                                    value_Q_subStateSpace = Q[f'{i}'][state, action].item()

                                # Calculate the TD errors for each sample
                                td_target = reward + gamma * next_state_value_Q_subStateSpace * (not done)
                                td_error = td_target.item() - value_Q_subStateSpace
                                Q[f'{i}'] = update_fast_Q(Q[f'{i}'], [state], [action], weights[j].item(), alphas, e, td_error)
                                # The 'weights[j]' has been applied to Q[f'{i}'] in the 'update_fast_Q' function
                                # So the 'weights[j]' is not used in the 'update_slow_Q' function directly, but it is actually used by the 'update_fast_Q' function
                                # Because the 'update_slow_Q' function is used to update the Q_[f'{i}'] with the Q[f'{i}'] in the 'update_fast_Q' function
                                Q_[f'{i}'] = update_slow_Q(Q_[f'{i}'], Q[f'{i}'], [state], [action], eta)
                                td_errors_for_subStateSpace_list[i].append(td_error)

                            # Update the priorities of the samples
                            replay_buffer_list[i].update(idxs, td_errors_for_subStateSpace_list[i])
                            # Add a fake counter so that experience replay is not performed again before the next experience replay cycle
                            # If step_count[i] % replay_interval == n, if PMDP has not transitioned to any subStateSpace related state in the subsequent 1 step or more, it ensures that experience replay is not repeated because step_count[i] % replay_interval == n
                            step_count[i] += 1
                    
                

                else:
                    # Normal Q updating code version without the priority experience replay
                    
                    for i in range(subStateSpace_number):
                        if len(subStateSpace_ActionIndice_list[i]) > 0:
                            Q[f'{i}'] = update_fast_Q(Q[f'{i}'], [Q_state_rowIndexs[i]], subStateSpace_ActionIndice_list[i], 1, alphas, e, TD_error_list[i])
                            # The formula.16 in the paper
                            Q_[f'{i}'] = update_slow_Q(Q_[f'{i}'], Q[f'{i}'], [Q_state_rowIndexs[i]], subStateSpace_ActionIndice_list[i], eta)


                ######################################### HQ is not used with the priority experience replay #########################################
                # QH_ is not related to any variables
                # Only use for the vertices withouy any variables in PMDP can be transitioned to the next vertices
                if QH_state_rowIndex is not None:
                    # The formula.15 in the paper
                    QH = update_fast_Q(QH, [QH_state_rowIndex], QH_ActionIndice_list, 1, alphas, e, TD_error_list[-1])
                    # The formula.16 in the paper
                    QH_ = update_slow_Q(QH_, QH, [QH_state_rowIndex], QH_ActionIndice_list, eta)

        # For all subStateSpace
        PMDP_accumulate_HC_SC_rewards = 0
        PMDP_accumulate_HC_rewards = 0
        PMDP_accumulate_SC_rewards = 0
        Q_accumulate_values = 0
        
        # The item of 'accumulate_rewards_list' represent the accumulate rewards list of the subStateSpace, the len(accumulate_rewards_list) is the number of subStateSpace
        # each item of 'accumulate_rewards_list' consist of [0] is accumulate rewards of the 'HC and SC' rewards of the subStateSpace, [1] is the accumulate rewards of the 'HC' rewards of the subStateSpace, [2] is the accumulate rewards of the 'SC' rewards of the subStateSpace
        
        
        for i in range(len(accumulate_rewards_list)):
            PMDP_accumulate_HC_rewards += (accumulate_rewards_list[i][1])
            PMDP_accumulate_SC_rewards += (accumulate_rewards_list[i][2])
            Q_accumulate_values += accumulate_rewards_list[i][0]
        PMDP_accumulate_HC_SC_rewards = PMDP_accumulate_HC_rewards + PMDP_accumulate_SC_rewards

        accumulate_HC_SC_rewards_in_each_trajectory.append(PMDP_accumulate_HC_SC_rewards)
        accumulate_HC_rewards_in_each_trajectory.append(PMDP_accumulate_HC_rewards)
        accumulate_SC_rewards_in_each_trajectory.append(PMDP_accumulate_SC_rewards)

        if (e + 1) % 10 == 0:
            print(f"\n Episode {e + 1}: Accumulate HC+SC Rewards: {PMDP_accumulate_HC_SC_rewards}, HC Rewards: {PMDP_accumulate_HC_rewards}, SC Rewards: {PMDP_accumulate_SC_rewards}, Q value: {Q_accumulate_values}")
            print(pmdp_env.get_current_state_Q())
            print(pmdp_env.get_current_context_Q())
            print(f"\n CET_HC_Counter: {CET_HC_count_list}")
            print(f"\n CET_SC_Counter: {CET_SC_count_list}")




        # each trajectory must initialize the state and context instance
        pmdp_env.current_state = pmdp_env.initialize_state()
        pmdp_env.context = pmdp_env.build_pmdp_context()
        pmdp_env.reset_KPIs_Constraints_to_ini_stages()

        #print(f"------------------------------------End of Episode: {e}-----------------------------------------------")
        #print(f"Accumulate rewards in this trajectory: {accumulate_rewards}")


    print(f"------------------------------------End of Episode: {e}-----------------------------------------------")       
    print(f"Accumulate rewards in each trajectory: {accumulate_HC_SC_rewards_in_each_trajectory}")




    # Ensure the "Experiment_Data" folder exists
    experiment_data_folder = "Experiment_Data"
    os.makedirs(experiment_data_folder, exist_ok=True)


    # Save accumulate_rewards_in_each_trajectory to a file to keep experimental results
    bpmn_file_name = file_path.split('/')[-1].split('.')[0]
    file_path = os.path.join(experiment_data_folder, f"accumulate_rewards_{bpmn_file_name}_with_{n_episodes}_eta_{eta}_1.txt")
    with open(file_path, "w") as f:
        for reward in accumulate_HC_SC_rewards_in_each_trajectory:
            f.write(f"{reward}\n")

    # Save accumulate_HC_rewards_in_each_trajectory to a file to keep experimental results
    hc_reward_file_path = os.path.join(experiment_data_folder, f"accumulate_HC_rewards_{bpmn_file_name}_with_{n_episodes}_eta_{eta}_1.txt")
    with open(hc_reward_file_path, "w") as f:
        for reward in accumulate_HC_rewards_in_each_trajectory:
            f.write(f"{reward}\n")

    sc_reward_file_path = os.path.join(experiment_data_folder, f"accumulate_SC_rewards_{bpmn_file_name}_with_{n_episodes}_eta_{eta}_1.txt")
    with open(sc_reward_file_path, "w") as f:
        for reward in accumulate_SC_rewards_in_each_trajectory:
            f.write(f"{reward}\n")
    


    pi_list = []
    V_list = []
    for i in range(subStateSpace_number):
        #Q[f'{i}'] = (Q[f'{i}'] + Q_[f'{i}']) / 2
        pi = torch.argmax(Q[f'{i}'], axis=1)  # Use torch.argmax
        pi_list.append(pi)
        V = torch.max(Q[f'{i}'], axis=1).values  # Use torch.max
        V_list.append(V)

    if len (QH) > 0:
        #QH = (QH + QH_) / 2
        pi = torch.argmax(QH, axis=1)  # Use torch.argmax
        pi_list.append(pi)
        V = torch.max(QH, axis=1).values  # Use torch.max
        V_list.append(V)
    


    #print(f"Accumulate rewards in each trajectory: {accumulate_HC_SC_rewards_in_each_trajectory}")

    # Learning convergence curve
    iterations = list(range(len(accumulate_HC_SC_rewards_in_each_trajectory)))
    print(len(accumulate_HC_SC_rewards_in_each_trajectory))
    # Plot the convergence curve
    plt.figure(figsize=(12, 8))
    # Optimize the curve style and color, remove data point markers
    plt.plot(iterations, accumulate_HC_SC_rewards_in_each_trajectory, linestyle='-', color='#1f77b4', label='Convergence Data', linewidth=2)
    # Add legend
    plt.legend(loc='best', fontsize=12)
    # Set title and labels
    plt.title('Convergence Curve 1', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    # Set x-axis ticks to display every certain interval
    interval = max(1, len(iterations) // 10)  # Display ticks every 10 data points
    plt.xticks(iterations[::interval], fontsize=12)
    plt.yticks(fontsize=12)
    # Show the plot
    plt.show()


    # Evaluate the policy
    num_trials = 2000  # Number of trials to average the rewards

    Policy_accumulate_HC_SC_rewards_in_each_trajectory = []
    Policy_accumulate_HC_rewards_in_each_trajectory = []
    Policy_accumulate_SC_rewards_in_each_trajectory = []

    pmdp_env.debug_mode = False
    Policy_accumulate_rewards_in_each_trajectory = []
    for e in tqdm(range(num_trials), leave=False):
        done = False
        activeVertices_of_State = start_vertex
        pmdp_env.current_state = pmdp_env.initialize_state()
        pmdp_env.context = pmdp_env.build_pmdp_context()
        pmdp_env.reset_KPIs_Constraints_to_ini_stages()
        accumulate_rewards_list = [[0, 0, 0] for _ in range(subStateSpace_number)]

        while not done:
            state = pmdp_env.current_state
            active_vertices_of_State_verticesList = pmdp_env.convert_horizon_item_to_list(activeVertices_of_State)
            
            Q_state_rowIndexs, Q_actionSpace_columnIndices_list, Q_actionSpace_rowsIndices_list = get_state_and_action_indices_in_Qspaces(
                pmdp_env, active_vertices_of_State_verticesList, state, subStateSpace_number, subStateSpace_vars_number_list,
                subStateSpace_vars_name_list, subStateSpace_vertices_name_list, all_string_to_int16,
                result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list, sub_stateTag_tensor_list)

            QH_state_rowIndex, QH_actionSpace_columnIndices, QH_actionSpace_rowsIndices = get_horizon_state_indices(
                result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list, active_vertices_of_State_verticesList, all_string_to_int16)

            
            
            action_tensor = torch.empty(0, dtype=torch_intType, device=learning_device)
            subStateSpace_ActionIndice_list = []
            QH_ActionIndice_list = []
            for i in range(subStateSpace_number):
                for j in range(len(Q_actionSpace_columnIndices_list[i])):
                    row_start, row_end = Q_actionSpace_rowsIndices_list[i][j][0], Q_actionSpace_rowsIndices_list[i][j][1]
                    col_start, col_end = Q_actionSpace_columnIndices_list[i][j][0], Q_actionSpace_columnIndices_list[i][j][1]

                    subStateSpace_ActionIndice = pi_list[i][Q_state_rowIndexs[i]].item()
                    
                    if subActionSpace_tensor_list[i].dim() == 1:
                            # Handle 1D tensor case
                            if torch.all(subActionSpace_tensor_list[i][subStateSpace_ActionIndice] == subActionSpace_tensor_list[i][subStateSpace_ActionIndice].item()) and all_int16_to_string_map[subActionSpace_tensor_list[i][subStateSpace_ActionIndice].item()].split('=')[1] == 'None':
                                subStateSpace_ActionIndice = select_action(Q_state_rowIndexs[i], row_start, row_end, Q[f'{i}'], 0)
                    else:
                        if torch.all(subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end] == subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end][0]) and all_int16_to_string_map[subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end][0].item()].split('=')[1] == 'None':
                            subStateSpace_ActionIndice = select_action(Q_state_rowIndexs[i], row_start, row_end, Q[f'{i}'], 0)
                    subStateSpace_ActionIndice_list.append(subStateSpace_ActionIndice)

                    if subActionSpace_tensor_list[i].dim() == 1:
                        legal_subActionTensor = subActionSpace_tensor_list[i][subStateSpace_ActionIndice].unsqueeze(0)
                    else:
                        legal_subActionTensor = subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end]

                    action_tensor = torch.cat((action_tensor, legal_subActionTensor))

            if QH_state_rowIndex is not None:
                row_start, row_end = QH_actionSpace_rowsIndices[0], QH_actionSpace_rowsIndices[1]
                col_start, col_end = QH_actionSpace_columnIndices[0], QH_actionSpace_columnIndices[1]
            
                QH_ActionIndice = pi_list[-1][QH_state_rowIndex].item()
                QH_ActionIndice_list.append(QH_ActionIndice)
                legal_subHorizonActionTensor = filtered_QHorizon_ActionSpace_tensor[QH_ActionIndice, col_start:col_end]

                # handle the horizonAction exists 'None' illegal horizon action in current Policy for current state, due to not enough learning iterations
                for element in legal_subHorizonActionTensor:
                    element_string = all_int16_to_string_map[element.item()]
                    if element_string.endswith('None'):
                        QH_ActionIndice = select_action(QH_state_rowIndex, row_start, row_end, QH, 0)
                        QH_ActionIndice_list[-1] = QH_ActionIndice
                        legal_subHorizonActionTensor = filtered_QHorizon_ActionSpace_tensor[QH_ActionIndice, col_start:col_end]
                        break
                # handle the special case of sychronization of the horizon action for the current state, to fasten the learning process
                if legal_subHorizonActionTensor.numel() > 1:
                    all_start_with_empty = True
                    empty_action_strings = []
                    for element in legal_subHorizonActionTensor:
                        element_string = all_int16_to_string_map[element.item()]
                        empty_action_strings.append(element_string)
                        if not element_string.startswith('empty'):
                            all_start_with_empty = False
                            break
                    if all_start_with_empty:
                        # All elements' string start with 'empty' handling logic
                        for empty_action_string in empty_action_strings:
                            empty_id = empty_action_string.split('=')[0]
                            finish_action_string = f"{empty_id}=finish"
                            if finish_action_string in all_string_to_int16.keys():
                                action_tensor = torch.cat((action_tensor, torch.tensor([all_string_to_int16[finish_action_string]], dtype=torch_intType, device=learning_device)))
                    else:
                        # Elements' strings do not start with 'empty' handling logic
                        action_tensor = torch.cat((action_tensor, legal_subHorizonActionTensor))
                else:
                    action_tensor = torch.cat((action_tensor, legal_subHorizonActionTensor))

            next_activeVertices_of_State = update_state_and_parse_next(
                pmdp_env, 
                action_tensor, 
                all_int16_to_string_map, 
                activeVertices_of_State
            )
            activeVertices_of_State = next_activeVertices_of_State

            # here the 'accumulate_rewards_list' is not involved the discount factor 'gamma'
            # 'gamma' is used in the 'calculate_td_error' function and other 'td_error' related calculations
            #immediate_rewards_list, accumulate_rewards_list, ConstraintState_list, evaluated_constraintsName_list = pmdp_env.calculate_reward(whc, wsc, HC_list, SC_list, ConstraintState_list, CET_mode, accumulate_rewards_list, subStateSpace_number = subStateSpace_number)
            # Set the 'CET_mode' to 'False' to avoid the 'CET' calculation in the evaluation process
            immediate_rewards_list, accumulate_rewards_list, ConstraintState_list, evaluated_constraintsName_list = pmdp_env.calculate_reward(whc, wsc, HC_list, SC_list, ConstraintState_list, False, False, accumulate_rewards_list, subStateSpace_number = subStateSpace_number)


            if isinstance(next_activeVertices_of_State, str) and next_activeVertices_of_State.startswith('Event'):
                if any(edge.target == next_activeVertices_of_State for edge in pmdp_env.graph.edges) and not any(edge.source == next_activeVertices_of_State for edge in pmdp_env.graph.edges):
                    done = True
        #print(pmdp_env.current_state)
        #print(f"accumulate_rewards_list: {accumulate_rewards_list}")

        # For all subStateSpace
        PMDP_accumulate_HC_SC_rewards = 0
        PMDP_accumulate_HC_rewards = 0
        PMDP_accumulate_SC_rewards = 0
        
        # The item of 'accumulate_rewards_list' represent the accumulate rewards list of the subStateSpace, the len(accumulate_rewards_list) is the number of subStateSpace
        # each item of 'accumulate_rewards_list' consist of [0] is accumulate rewards of the 'HC and SC' rewards of the subStateSpace, [1] is the accumulate rewards of the 'HC' rewards of the subStateSpace, [2] is the accumulate rewards of the 'SC' rewards of the subStateSpace
        for i in range(len(accumulate_rewards_list)):
            PMDP_accumulate_HC_rewards += (accumulate_rewards_list[i][1])
            PMDP_accumulate_SC_rewards += (accumulate_rewards_list[i][2])
        PMDP_accumulate_HC_SC_rewards = PMDP_accumulate_HC_rewards + PMDP_accumulate_SC_rewards


        Policy_accumulate_HC_SC_rewards_in_each_trajectory.append(PMDP_accumulate_HC_SC_rewards)
        Policy_accumulate_HC_rewards_in_each_trajectory.append(PMDP_accumulate_HC_rewards)
        Policy_accumulate_SC_rewards_in_each_trajectory.append(PMDP_accumulate_SC_rewards)



    iterations = list(range(len(Policy_accumulate_HC_SC_rewards_in_each_trajectory)))
    # Plot the convergence curve
    plt.figure(figsize=(12, 8))
    # Optimize the curve style and color, remove data point markers
    plt.plot(iterations, Policy_accumulate_HC_SC_rewards_in_each_trajectory, linestyle='-', color='#1f77b4', label='Convergence Data', linewidth=2)
    # Add legend
    plt.legend(loc='best', fontsize=12)
    # Set title and labels
    plt.title('Optimal Policy Experiment 1', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    # Set x-axis ticks to display every certain interval
    interval = max(1, len(iterations) // 10)  # Display ticks every 10 data points
    plt.xticks(iterations[::interval], fontsize=12)
    plt.yticks(fontsize=12)
    # Show the plot
    plt.show()

    # Save accumulate_rewards_in_each_trajectory to a file to keep experimental results
    
    file_path = os.path.join(experiment_data_folder, f"Policy_accumulate_HC_SC_rewards_in_each_trajectory{bpmn_file_name}_with_{n_episodes}_eta_{eta}_1.txt")
    with open(file_path, "w") as f:
        for reward in Policy_accumulate_HC_SC_rewards_in_each_trajectory:
            f.write(f"{reward}\n")

    file_path = os.path.join(experiment_data_folder, f"Policy_accumulate_HC_rewards_in_each_trajectory{bpmn_file_name}_with_{n_episodes}_eta_{eta}_1.txt")
    with open(file_path, "w") as f:
        for reward in Policy_accumulate_HC_rewards_in_each_trajectory:
            f.write(f"{reward}\n")

    file_path = os.path.join(experiment_data_folder, f"Policy_accumulate_SC_rewards_in_each_trajectory{bpmn_file_name}_with_{n_episodes}_eta_{eta}_1.txt")
    with open(file_path, "w") as f:
        for reward in Policy_accumulate_SC_rewards_in_each_trajectory:
            f.write(f"{reward}\n")
    
    print(pmdp_env.get_current_context_Q)
    # 
    return Q, V, pi


############################ The part of the CSSC_MDP and N_MDP shared functions for compatibale with PMDP ########################################
def get_sub_state_space(pmdp_env, graph):
    """
    Process the sub-state space and return a sorted list of variables.

    Parameters:
    pmdp_env: PMDP environment object, containing graph information.
    graph: graph object, containing variable information.

    Returns:
    list: Sorted list of sub-state space variables.
    """
    constrainst_list = pmdp_env.graph.HC + pmdp_env.graph.SC
    # subStateSpace_vars_list containts the names of variables, not the objects of variables
    subStateSpace_vars_list = classify_vars(pmdp_env, constrainst_list)
    subStateSpace_vars_list = merge_intersections(subStateSpace_vars_list)
    
    # vars_dict is a dictionary, which stores the objects of variables
    vars_dict = retrieve_variables_from_graph(graph)

    sub_stateSpace_vars_sorted_list = []

    for subStateSpace_vars in subStateSpace_vars_list:
        var_list = []
        variables_spaces = []
        # get var information
        for var_name in subStateSpace_vars:
            var = vars_dict[var_name]
            var_list.append([var.vertex_id, var.name, var.domain, var.controlType])

        # sort the variable based on the vertex_id
        # make sure the order of variable is consistent comply with vertex_id, in each execution when call 'sort_and_group_var_list' function
        var_list.sort(key=lambda x: (x[0]))

        # sort the variable, make sure the order of variable is consistent in each execution when constructing the states space for rows of Q-table
        # sort the variable based on the vertex_id, and then based on the name of variable
        # Use ‘var.vertex_id’ as the sort key to ensure that variables in the same vertex are continuous in the sorted var_list, so that the filter_tensor function can correctly filter illegal actions
        var_list = sort_and_group_var_list(var_list)
        sub_stateSpace_vars_sorted_list.append(var_list)
    
    return sub_stateSpace_vars_sorted_list

############################ End of the CSSC_MDP and N_MDP shared functions ########################################

############################ The parts of the CSSC_MDP_with_Qlearning beginning ########################################

def calculate_medians_from_txt(file_path, n):
    """
    Calculate the median of each variable in the given .txt file, split the data into n parts, and calculate the median of each part.

    Parameters:
    file_path (str): The path of the .txt file.
    n (int): The number of parts to split.

    Return:
    list: A list of dictionaries containing the median of each part of the data.
    """
    data = []

    # Read the file and parse the data
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(ast.literal_eval(line.strip()))

    # Split the data
    chunk_size = len(data) // n
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
    if len(data) % n != 0:
        chunks[-1].extend(data[n * chunk_size:])

    # Calculate the median of each data
    all_medians = []
    for chunk in chunks:
        # Extract the values for each variable
        variables = chunk[0].keys()
        values = {var: [] for var in variables}

        for entry in chunk:
            for var in variables:
                value = entry[var]
                if value is not None:
                    values[var].append(int(value))

        # Calculating the median
        medians = {}
        for var in variables:
            if values[var]:
                medians[var] = statistics.median(values[var])
            else:
                medians[var] = None  

        all_medians.append(medians)

    return all_medians

def calculate_means(medians_list):
    """
    Compute the mean for each key in each dictionary.

    Parameters:
    medians_list (list): List containing dictionaries of medians.

    Returns:
    dict: Dictionary of means for each key.
    """
    if not medians_list:
        return {}

    keys = medians_list[0].keys()
    sums = {key: 0 for key in keys}
    counts = {key: 0 for key in keys}

    for medians in medians_list:
        for key, value in medians.items():
            if value is not None:
                sums[key] += value
                counts[key] += 1

    means = {key: (sums[key] / counts[key] if counts[key] > 0 else None) for key in keys}
    return means


def process_matrices(n=30, k = 1160, decimal=1, base_path=None):
    """
    Process RT and TP matrices and calculate the mean for each abstract_service.

    Parameters:
    - n (int): number of blocks, i.e. number of tasks/activities in a BPMN model.
    - decimal (int): number of decimal places to round to.
    - var_name (str): variable name, used to generate dictionary keys.
    - base_path (str, optional): base path, defaults to the current file directory.

    Returns:
    - abstract_service_mean_dict (dict): dictionary of mean values for each abstract_service.
    """
    if base_path is None:
        base_path = os.path.dirname(__file__)

    # Jump up one level to get the parent directory, due to now stay in 'src' folder
    if 'src' in base_path:
        base_path = os.path.dirname(base_path)
        base_path = os.path.dirname(base_path)

    rt_matrix_path = os.path.join(base_path, 'Datasets', 'wsdream', 'rtMatrix.txt')
    tp_matrix_path = os.path.join(base_path, 'Datasets', 'wsdream', 'tpMatrix.txt')

    # Read the matrices
    rt_matrix = read_matrix(rt_matrix_path)[:, :k]
    tp_matrix = read_matrix(tp_matrix_path)[:, :k]

    # Split the matrices
    rt_matrix_round_blocks = split_matrix(rt_matrix, n, decimal=decimal)
    tp_matrix_round_blocks = split_matrix(tp_matrix, n, decimal=decimal)

    # Calculate median
    abstract_service_values_median = []
    for i, abstract_services in enumerate(rt_matrix_round_blocks):
        # Convert from ‘339x194’ to ‘194x339’, that is, each row is a concrete service, the columns are the records
        abstract_services = abstract_services.T
        median_values = []
        for concrete_services in abstract_services:
            # Remove -1 values
            filtered_services = concrete_services[concrete_services != -1]
            if filtered_services.size > 0:
                median = np.median(filtered_services, axis=0)
            else:
                continue
            median_values.append(median)
        abstract_service_values_median.append(median_values)

    # Calculate mean
    abstract_service_mean_dict = {}
    for i, median_values in enumerate(abstract_service_values_median):
        mean_value = np.mean(median_values, axis=0)
        abstract_service_mean_dict[f'rt{i+1}'] = mean_value

    # Calculate median
    abstract_service_values_median = []
    for i, abstract_services in enumerate(tp_matrix_round_blocks):
        # Convert from ‘339x194’ to ‘194x339’, that is, each row is a concrete service, the columns are the records
        abstract_services = abstract_services.T
        median_values = []
        for concrete_services in abstract_services:
            # Remove -1 values
            filtered_services = concrete_services[concrete_services != -1]
            if filtered_services.size > 0:
                median = np.median(filtered_services, axis=0)
            else:
               continue
            median_values.append(median)
        abstract_service_values_median.append(median_values)

    # Calculate mean

    for i, median_values in enumerate(abstract_service_values_median):
        mean_value = np.mean(median_values, axis=0)
        abstract_service_mean_dict[f'tp{i+1}'] = mean_value
    

    return abstract_service_mean_dict



def calculate_var_portion_values(pmdp_env, n, subStateSpace_vars_name_list, subStateSpace_number, SC_list, total_candidates_num = 1160):
    """
    Calculate the median of each variable in a given .txt file, split the data into n parts, and calculate the median of each part.
    Each part is equivalent to splitting the real-world data set into n parts in CSSC, and each part of the data is equivalent to a candidate state QoS record

    Parameters:
    file_path (str): The path of the .txt file.
    n (int): The number of parts to be split.

    Return:
    dict1: The dictionary of variance percentage values for each variable.
    sc_Sum_dict: the upper limit (for all variables) of the constraints to which each variable belongs 
    """ 

    ''' 
    Old version 
    # each line of median_list is a dict, which stores the median value of each variable in the abstract service according to the concrete service log file 
    medians_list = calculate_medians_from_txt(file_path, n) 
    # 'means' is a dict, which stores the mean value of each variable in the abstract service according to the concrete service log file 
    means = calculate_means(medians_list)
    '''
    
    means = {}
    # new version, to calculate the means for 'CSSC-MDP Paper' about '3.2 State Level Division' standard, but the 'tp' is not the sum of all candidates, it should be the minimum value of all candidates about the throughput, so will beupdated in the following code
    means = process_matrices(n, k = total_candidates_num, decimal=1)

    sc_Sum_dict = {}

    # CSSC all constraints stored in the 'SC_list', it only support it


    # 'subStateSpace_number' usual is '1'
    subStateSpace_ConstraintMean_sum_list = []
    subStateSpace_ConstraintBound_list = []
    for i in range(subStateSpace_number):
        subStateSpace_ConstraintMean_sum_list.append([])

        subStateSpace_ConstraintBound_list.append([])

        # CSSC all constraints stored in the 'SC_list', it only support it, so only one item in the 'subStateSpace_ConstraintBound_list'
        #subStateSpace_ConstraintBound_list.append([])

        constraints = SC_list[i]
        vars_list_in_constraint = []
        for constraint in constraints:
            vars_list_in_constraint.append(retrieve_Vars_from_expression(pmdp_env, constraint.expression))
            if r'\le' in constraint.expression:
                left_expression = constraint.expression.split(r'\le')[0].strip()
                right_expression = constraint.expression.split(r'\le')[1].strip()
            elif r'<' in constraint.expression:
                left_expression = constraint.expression.split(r'<')[0].strip()
                right_expression = constraint.expression.split(r'<')[1].strip()
            elif r'>' in constraint.expression:
                raise ValueError(f"Constraint '{constraint.expression}' is not a valid constraint about CSSC-MDP, which only support \le or < (trying convert {constraint.expression} via multipy -1), and bound must be set at right (e.g., ... \le bound, or ... < bound).")
            elif r'\ge' in constraint.expression:
                raise ValueError(f"Constraint '{constraint.expression}' is not a valid constraint about CSSC-MDP, which only support \le or < (trying convert {constraint.expression} via multipy -1), and bound must be set at right (e.g., ... \le bound, or ... < bound).")
            if left_expression.isdigit():
                raise ValueError(f"Constraint '{constraint.expression}' is not a valid constraint about CSSC-MDP, the bound must be set at right (e.g., ... \le bound, or ... < bound).")
            #elif right_expression.isdigit():
            # Handle the case of the negative number
            elif right_expression.lstrip('-').replace('.', '', 1).isdigit():
                subStateSpace_ConstraintBound_list[i].append(float(right_expression))
            else:
                raise ValueError(f"Constraint '{constraint.expression}' is not a valid constraint about CSSC-MDP, or a illegal bound for this constraint.")
        
        # Only for ws-dream1 dataset about rtmatrix and tpmatrix
        #vars_list_in_constraint[1] = vars_list_in_constraint[0]
        #vars_list_in_constraint = vars_list_in_constraint[0:2]

        for j in range(len(vars_list_in_constraint)):
            subStateSpace_ConstraintMean_sum_list[i].append(0)
            for var in vars_list_in_constraint[j]:
                if 'rt' in var or 'tp' in var:
                    var_ = var
                    #tmp_var = var.replace('U', '')
                    #if j == 0:
                        #var_ = f'rt{tmp_var}'
                    #elif j == 1:
                        #var_ = f'tp{tmp_var}'
                else:
                    raise ValueError(f"Constraint '{constraint.expression}' is not compatible about wsdream_dataset and CSSC-MDP.")
                if var_ not in means or var_ not in means:
                    raise ValueError(f"Variable '{var}' not found in the dataset.")
                elif var in subStateSpace_vars_name_list[i]:
                    subStateSpace_ConstraintMean_sum_list[i][j] += means[var_]
                else:
                    raise ValueError(f"Variable '{var}' is not in the subStateSpace variables list: {subStateSpace_vars_name_list}.")
                
            #hc_meanSum_dict[constraints[j].name] = subStateSpace_ConstraintMean_sum_list[i][j]
            sc_Sum_dict[constraints[j].name] = subStateSpace_ConstraintBound_list[i][j]

    subStateSpace_portion_value_dict_list = []
    for i in range(subStateSpace_number):
        # This make every dict is independent, not share the same dict
        subStateSpace_portion_value_dict_list.append([{} for _ in vars_list_in_constraint])
        for j in range(len(vars_list_in_constraint)):
            for var in vars_list_in_constraint[j]:
                if 'rt' in var or 'tp' in var:
                    var_ = var
                    #tmp_var = var.replace('U', '')
                    #if j == 0:
                        #var_ = f'rt{tmp_var}'
                    #elif j == 1:
                        #var_ = f'tp{tmp_var}'
                else:
                    raise ValueError(f"Constraint '{constraint.expression}' is not compatible about wsdream_dataset and CSSC-MDP.")
                if 'rt' in var_:
                    subStateSpace_portion_value_dict_list[i][j][var] = (means[var_] / subStateSpace_ConstraintMean_sum_list[i][j]) * subStateSpace_ConstraintBound_list[i][j]
                else:
                    # 'tp' in var_
                    # CSSC-MDP, example the throughput, and it is not sum of all candidates, it should be the minimum value of all candidates about the throughput
                    # The 'StateDivision' in the 'CSSC-MDP' paper, cannot be used this calculation, because the 'StateDivision' is the sum of all candidates
                    # And the 'StateDivision' only support Aggregation QoS < Bound, so multiply -1 to convert the 'kpi > number' to '-kpi < -number', so 'subStateSpace_ConstraintBound_list[i][j]' for 'tp' is the negative number
                    subStateSpace_portion_value_dict_list[i][j][var] = (subStateSpace_ConstraintBound_list[i][j])

    
    # The first row of 'subStateSpace_portion_value_dict_list' is the RT, and the second row is the TP 
    return subStateSpace_portion_value_dict_list, sc_Sum_dict

def CSSC_calculate_reward(pmdp, subStateSpace_portion_value_dict_list, sc_Sum_dict, active_vertex, subStateSpace_vars_name_list, w_hc: float, w_sc: float, HC_list, SC_list, accumulate_rewards_list, old_state_level = 1, subStateSpace_number = 1) -> list[float]:
        new_state_level = old_state_level
        state_level_list = []

        for i in range(subStateSpace_number):
            #relateds_variables = pmdp.graph.vertices[active_vertex].C_v + pmdp.graph.vertices[active_vertex].U_v
            relateds_variables = pmdp.graph.vertices[active_vertex].U_v
            for variable in relateds_variables:
                if 'Horizon_Action' not in variable.name:
                    QoS_value = pmdp.context[variable.name]
                    if QoS_value != None:
                        for j, sc in enumerate(SC_list[i]):
                             # This special, only for wsdreamDataset 'j=0' is the RT, and 'j=1' is the TP
                            #QoS_value_ = QoS_value[j]
                            # below is old code, it model the RT and TP as the '[rt, tp]' in one variable, the new split the RT and TP to two variables
                            QoS_value_ = QoS_value
                            if j == 1:
                                # CSSC-MDP, only support kpi < number, so the TP must be converted to negative number, for the original if 'kpi > number' that is converted to '-kpi < -number'
                                # the bound of TP in 'portion_value_dict' has been converted to the negative number
                                #QoS_value_ = QoS_value[j] * -1

                                QoS_value_ = QoS_value * -1

                            if variable.name in retrieve_Vars_from_expression(pmdp, sc.expression):
                                constraint_name = sc.name
                                # The first row of 'subStateSpace_portion_value_dict_list' is the RT, and the second row is the TP 
                                portion_value_dict = subStateSpace_portion_value_dict_list[i][j]
                                if j != 1:
                                    if QoS_value_ <= portion_value_dict[f'{variable.name}'] * 0.85:
                                        state_level_list.append(1)
                                        #new_state_level = 1
                                    elif QoS_value_ <= portion_value_dict[f'{variable.name}']:
                                        state_level_list.append(2)
                                        #new_state_level = 2
                                    elif QoS_value_ <= sc_Sum_dict[constraint_name]:
                                        state_level_list.append(3)
                                        #new_state_level = 3
                                    else:
                                        state_level_list.append(4)
                                        #new_state_level = 4
                                else:
                                    # for 'throughput' in the 'CSSC-MDP', it means every candidate must be less than the bound, so the 'portion_value_dict[f'{variable.name}']' is the negative number
                                    if QoS_value_ <= sc_Sum_dict[constraint_name]:
                                        state_level_list.append(1)
                                        #new_state_level = 1
                                    else:
                                        state_level_list.append(4)
                                        #new_state_level = 2

        if len(state_level_list) == 0:
            new_state_level = old_state_level
        else:
            new_state_level = max(state_level_list)

        reward = 10 * (old_state_level - new_state_level) + 10

        # the len(subStateVarsName)+1 must equal to the 'rewards_number'
        # the 'HC_list/SC_list,' is the list of constraints fo each sub-state, and is has the fixed order about the sub-state for each call of 'calculate_reward' function
        Rewards_HC_list = [0] * subStateSpace_number
        Rewards_SC_list = [0] * subStateSpace_number
        Rewards_HC_plus_SC_list = [0] * subStateSpace_number

        All_Rewards_All_subStates = 0

        updated_stage_HC = []
        updated_stage_SC = []
        HC_evaluated_constraintsName_list = []
        SC_evaluated_constraintsName_list = []
        evaluated_constraintsName_list = []
       
        for i in range(subStateSpace_number):
            HC_evaluated_constraintsName_list.append([])
            SC_evaluated_constraintsName_list.append([])
            evaluated_constraintsName_list.append([])

            sub_HC_Rewards = 0
            sub_SC_Rewards = reward

            Rewards_HC_list[i] = sub_HC_Rewards * w_hc
            Rewards_SC_list[i] = sub_SC_Rewards * w_sc
            Rewards_HC_plus_SC_list[i] = Rewards_HC_list[i] + Rewards_SC_list[i]
            All_Rewards_All_subStates += Rewards_HC_plus_SC_list[i]
   

            accumulate_rewards_list[i][0] += Rewards_HC_plus_SC_list[i]
            accumulate_rewards_list[i][1] += Rewards_HC_list[i]
            accumulate_rewards_list[i][2] += Rewards_SC_list[i]

        
        
        # The 'Rewards_HC_plus_SC_list' consist of two parts, the first part is the HC rewards, and the second part is the SC rewards, each part is weight i
        return [Rewards_HC_plus_SC_list, Rewards_HC_list, Rewards_SC_list, All_Rewards_All_subStates], accumulate_rewards_list, new_state_level

def create_alpha_list(initial_alpha, n_episodes, decay_rate=0.000007):
    """
    Create a list of alpha values, with alpha decaying at each iteration.

    Parameters:
    initial_alpha (float): initial alpha value.
    n_episodes (int): number of iterations.
    decay_rate (float): alpha decay rate at each iteration.

    Returns:
    list: list of alpha values at each iteration.
    """
    alpha_list = []
    alpha = initial_alpha

    for _ in range(n_episodes):
        alpha_list.append(alpha)
        alpha -= decay_rate

    return alpha_list


def evaluate_cssc_policy(
    Q_CSSC,
    QH,
    eval_number,
    subStateSpace_number,
    result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list,
    result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list,
    all_string_to_int16,
    all_int16_to_string_map,
    subActionSpace_tensor_list,
    filtered_QHorizon_ActionSpace_tensor,
    torch_intType,
    learning_device,
    pmdp_env,
    start_vertex,
    frequency_prob_dict,
    MDP_example_mode,
    new_state_level,
    select_action,
    get_horizon_state_indices,
    update_state_and_parse_next,
    whc,
    wsc,
    HC_list,
    SC_list,
    ConstraintState_list,
    CSSC_calculate_reward,
    subStateSpace_portion_value_dict_list,
    sc_Sum_dict,
    subStateSpace_vars_name_list,
    CSSC_MDP_SC_list
):
    import torch
    from tqdm import tqdm

    pi_list = []
    V_list = []
    for i in range(subStateSpace_number):
        pi = torch.argmax(Q_CSSC[f'{i}'], axis=1)
        pi_list.append(pi)
        V = torch.max(Q_CSSC[f'{i}'], axis=1).values
        V_list.append(V)

    if len(QH) > 0:
        pi = torch.argmax(QH, axis=1)
        pi_list.append(pi)
        V = torch.max(QH, axis=1).values
        V_list.append(V)



    Policy_accumulate_HC_SC_rewards_in_each_trajectory = []
    Policy_accumulate_HC_rewards_in_each_trajectory = []
    Policy_accumulate_SC_rewards_in_each_trajectory = []

    pmdp_env.debug_mode = False
    Policy_accumulate_rewards_in_each_trajectory = []

    for e in tqdm(range(eval_number), leave=False):
        done = False
        activeVertices_of_State = start_vertex
        pmdp_env.current_state = pmdp_env.initialize_state()
        pmdp_env.context = pmdp_env.build_pmdp_context()
        pmdp_env.reset_KPIs_Constraints_to_ini_stages()
        accumulate_rewards_list = [[0, 0, 0] for _ in range(subStateSpace_number)]
        CSSC_accumulate_rewards_list = [[0, 0, 0] for _ in range(subStateSpace_number)]

        while not done:
            state = pmdp_env.current_state
            active_vertices_of_State_verticesList = pmdp_env.convert_horizon_item_to_list(activeVertices_of_State)

            old_activeVertices_of_State = activeVertices_of_State
            if 'StartEvent' in old_activeVertices_of_State:
                old_state_level = 1
            else:
                old_state_level = new_state_level

            # Q_CSSC is the Q-table for CSSC
            for i in range(subStateSpace_number):
                if old_activeVertices_of_State in result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0]:
                    active_index = result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0].index(old_activeVertices_of_State)
                    Q_CSSC_state_rowIndexs = [active_index * 4 + old_state_level - 1]
                    Q_CSSC_actionSpace_columnIndices_list = [[result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][2][active_index]]]
                    Q_CSSC_actionSpace_rowsIndices_list = [[result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][1][active_index]]]

            # Get horizon state indices
            QH_state_rowIndex, QH_actionSpace_columnIndices, QH_actionSpace_rowsIndices = get_horizon_state_indices(
                result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list, active_vertices_of_State_verticesList, all_string_to_int16)

            action_tensor = torch.empty(0, dtype=torch_intType, device=learning_device)
            subStateSpace_ActionIndice_list = []
            QH_ActionIndice_list = []

            if old_activeVertices_of_State in result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0]:
                for i in range(subStateSpace_number):
                    for j in range(len(Q_CSSC_actionSpace_columnIndices_list[i])):
                        row_start, row_end = Q_CSSC_actionSpace_rowsIndices_list[i][j][0], Q_CSSC_actionSpace_rowsIndices_list[i][j][1]
                        col_start, col_end = Q_CSSC_actionSpace_columnIndices_list[i][j][0], Q_CSSC_actionSpace_columnIndices_list[i][j][1]

                        # Learned Policy
                        subStateSpace_ActionIndice = pi_list[i][Q_CSSC_state_rowIndexs[0]].item()

                        if subActionSpace_tensor_list[i].dim() == 1:
                            # Handle 1D tensor
                            if torch.all(subActionSpace_tensor_list[i][subStateSpace_ActionIndice] == subActionSpace_tensor_list[i][subStateSpace_ActionIndice].item()) and \
                               all_int16_to_string_map[subActionSpace_tensor_list[i][subStateSpace_ActionIndice].item()].split('=')[1] == 'None':
                                subStateSpace_ActionIndice = select_action(Q_CSSC_state_rowIndexs[0], row_start, row_end, Q_CSSC[f'{i}'], 0)
                        else:
                            if torch.all(subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end] == subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end][0]) and \
                               all_int16_to_string_map[subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end][0].item()].split('=')[1] == 'None':
                                subStateSpace_ActionIndice = select_action(Q_CSSC_state_rowIndexs[0], row_start, row_end, Q_CSSC[f'{i}'], 0)
                        subStateSpace_ActionIndice_list.append(subStateSpace_ActionIndice)

                        if subActionSpace_tensor_list[i].dim() == 1:
                            legal_subActionTensor = subActionSpace_tensor_list[i][subStateSpace_ActionIndice].unsqueeze(0)
                        else:
                            legal_subActionTensor = subActionSpace_tensor_list[i][subStateSpace_ActionIndice, col_start:col_end]

                        action_tensor = torch.cat((action_tensor, legal_subActionTensor))

            if QH_state_rowIndex is not None:
                row_start, row_end = QH_actionSpace_rowsIndices[0], QH_actionSpace_rowsIndices[1]
                col_start, col_end = QH_actionSpace_columnIndices[0], QH_actionSpace_columnIndices[1]

                QH_ActionIndice = pi_list[-1][QH_state_rowIndex].item()
                QH_ActionIndice_list.append(QH_ActionIndice)
                legal_subHorizonActionTensor = filtered_QHorizon_ActionSpace_tensor[QH_ActionIndice, col_start:col_end]

                # Handle actions ending with 'None'
                for element in legal_subHorizonActionTensor:
                    element_string = all_int16_to_string_map[element.item()]
                    if element_string.endswith('None'):
                        QH_ActionIndice = select_action(QH_state_rowIndex, row_start, row_end, QH, 0)
                        QH_ActionIndice_list[-1] = QH_ActionIndice
                        legal_subHorizonActionTensor = filtered_QHorizon_ActionSpace_tensor[QH_ActionIndice, col_start:col_end]
                        break

                if legal_subHorizonActionTensor.numel() > 1:
                    all_start_with_empty = True
                    empty_action_strings = []
                    for element in legal_subHorizonActionTensor:
                        element_string = all_int16_to_string_map[element.item()]
                        empty_action_strings.append(element_string)
                        if not element_string.startswith('empty'):
                            all_start_with_empty = False
                            break
                    if all_start_with_empty:
                        # All elements start with 'empty'
                        for empty_action_string in empty_action_strings:
                            empty_id = empty_action_string.split('=')[0]
                            finish_action_string = f"{empty_id}=finish"
                            if finish_action_string in all_string_to_int16.keys():
                                action_tensor = torch.cat((
                                    action_tensor,
                                    torch.tensor([all_string_to_int16[finish_action_string]], dtype=torch_intType, device=learning_device)
                                ))
                    else:
                        # Some elements do not start with 'empty'
                        action_tensor = torch.cat((action_tensor, legal_subHorizonActionTensor))
                else:
                    action_tensor = torch.cat((action_tensor, legal_subHorizonActionTensor))

            next_activeVertices_of_State = update_state_and_parse_next(
                pmdp_env,
                action_tensor,
                all_int16_to_string_map,
                activeVertices_of_State,
                frequency_prob_dict=frequency_prob_dict,
                MDP_example_mode=MDP_example_mode
            )
            activeVertices_of_State = next_activeVertices_of_State

            # Calculate rewards
            immediate_rewards_list, accumulate_rewards_list, ConstraintState_list, evaluated_constraintsName_list = pmdp_env.calculate_reward(
                whc, wsc, HC_list, SC_list, ConstraintState_list, False, False, accumulate_rewards_list, subStateSpace_number=subStateSpace_number
            )

            # CSSC reward calculation
            CSSC_immediate_rewards_list, CSSC_accumulate_rewards_list, new_state_level = CSSC_calculate_reward(
                pmdp_env,
                subStateSpace_portion_value_dict_list,
                sc_Sum_dict,
                old_activeVertices_of_State,
                subStateSpace_vars_name_list,
                whc,
                wsc,
                HC_list,
                CSSC_MDP_SC_list,
                CSSC_accumulate_rewards_list,
                old_state_level=old_state_level,
                subStateSpace_number=subStateSpace_number
            )

            if isinstance(next_activeVertices_of_State, str) and next_activeVertices_of_State.startswith('Event'):
                if any(edge.target == next_activeVertices_of_State for edge in pmdp_env.graph.edges) and \
                   not any(edge.source == next_activeVertices_of_State for edge in pmdp_env.graph.edges):
                    done = True

        # Accumulate rewards
        PMDP_accumulate_HC_SC_rewards = 0
        PMDP_accumulate_HC_rewards = 0
        PMDP_accumulate_SC_rewards = 0

        for i in range(len(accumulate_rewards_list)):
            PMDP_accumulate_HC_rewards += accumulate_rewards_list[i][1]
            PMDP_accumulate_SC_rewards += accumulate_rewards_list[i][2]
        PMDP_accumulate_HC_SC_rewards = PMDP_accumulate_HC_rewards + PMDP_accumulate_SC_rewards

        Policy_accumulate_HC_SC_rewards_in_each_trajectory.append(PMDP_accumulate_HC_SC_rewards)
        Policy_accumulate_HC_rewards_in_each_trajectory.append(PMDP_accumulate_HC_rewards)
        Policy_accumulate_SC_rewards_in_each_trajectory.append(PMDP_accumulate_SC_rewards)

    HC_score_sum = 0
    SC_score_sum = 0
    for i in range(len(Policy_accumulate_HC_rewards_in_each_trajectory)):
        HC_score_sum += Policy_accumulate_HC_rewards_in_each_trajectory[i]
        SC_score_sum += Policy_accumulate_SC_rewards_in_each_trajectory[i]


    # Calculate the proportion of instances where total_rewards is equal to 1
    count_ones = 0
    for reward in Policy_accumulate_SC_rewards_in_each_trajectory:
        if reward == 1:
            count_ones += 1
    success_service_compostions_proportion = count_ones / len(Policy_accumulate_SC_rewards_in_each_trajectory) if Policy_accumulate_SC_rewards_in_each_trajectory else 0
   
    HC_score_avg = 0
    SC_score_avg = SC_score_sum / len(Policy_accumulate_SC_rewards_in_each_trajectory)


    '''
    HC_score_avg = HC_score_sum / len(Policy_accumulate_HC_rewards_in_each_trajectory)
    SC_score_avg = SC_score_sum / len(Policy_accumulate_SC_rewards_in_each_trajectory)
    total_score_avg = (HC_score_avg + SC_score_avg) 
    '''

    return success_service_compostions_proportion, HC_score_avg, SC_score_avg



def CSSC_MDP_with_Qlearing(bpmn_file_path='', n_abstract_number=30, total_candidates_num = 1160, gamma=0.8, init_alpha=0.5, min_alpha=1e-4, alpha_decay_ratio=0.9,
                            init_epsilon=1.0, min_epsilon=0.01, epsilon_decay_ratio=0.9, n_episodes=2000, eta=1, whc=0, wsc=1, CSSC_evaluation_rewards_list = [],  CSSC_evaluation_HC_rewards_list = [], 
                            CSSC_evaluation_SC_rewards_list = [], eval_interval=100, eval_number=1000, Evaluate_mode = True, Training_time_record = []):
    # The superparameter 'eta' is set to '1', which mean only Q-learning, not the Double Q-learning
    MDP_example_mode = 'CSSC_MDP'
    graph = parse_bpmn(bpmn_file_path, n_abstract_number, total_candidates_num = total_candidates_num, CSSC_MDP_example_mode = True, NMDP_example_mode = False)

    ######## Initialize the graph based on the 'wsdream_dataset1' and BPMN file ########
    #- frequency_prob_dict (dict): A dictionary of frequency probabilities for each variable.
    #- activity_variable_doamin_string_list (list): The domain string list of activity variables C and U
    frequency_prob_dict, _ = compute_frequency_probabilities_domain(n_abstract_number, k = total_candidates_num, mode_str='Initializing The Probability Dirbution of Dataset: ', decimal=1)
    if len(frequency_prob_dict) != len(graph.vertices) - 2:
        raise ValueError(f"The number of activities in the BPMN file is not equal to the number of activities in the wsdream_dataset1.\n Please make sure the number of activities in the BPMN file is equal to the parameter of 'n_abstract_number'.")


    pmdp_env = PMDPEnvironment(graph)
    pmdp_env.debug_mode = False
    horizon_space = pmdp_env.get_horizon_space()

    #alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes, log_start = -0.5) 
    #epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes,  log_start = -1)
    alphas = create_alpha_list(init_alpha, n_episodes)
    # The CSSC-MDP paper says that the best epsilon is set to '0.6 (in that paper, if random number (0,1) is less than 0.6, then choose the greedy action, otherwise choose the random action), , which means 60% to exploit, and 40% to explore'
    # In standard Q-learning, realize the same epsilon, we need to set the 'epsilons' to [0.4] * n_episodes, which means 60% to exploit, and 40% to explore, if a random number (0,1) is less than 0.4, then choose the greedy action, otherwise choose the random action
    epsilons = [0.4] * n_episodes 
    #epsilons = [1.0] * n_episodes 


    Initialize_statesActions_PMDPMappingDict(pmdp_env, graph)
    all_int16_to_string_map = {}
    all_int16_to_string_map.update(int16_to_string_map)
    all_int16_to_string_map.update(wait_int16_to_string_map)
    all_int16_to_string_map.update(None_int16_to_string_map)
    all_int16_to_string_map.update(finish_int16_to_string_map)
    all_int16_to_string_map.update(horizon_int16_to_string_map)
    # Swap keys and values
    all_string_to_int16 = {value: key for key, value in all_int16_to_string_map.items()}



    filtered_QHorizonTag_StateSpace_tensor = process_horizon(pmdp_env, horizon_space, horizon_int16_to_string_map, existing_hashes, learning_device)
    filtered_QHorizon_ActionSpace_tensor, result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list = get_QHorizon_actionSpace_columns(pmdp_env, horizon_space, filtered_QHorizonTag_StateSpace_tensor)

    sub_stateSpace_vars_sorted_list = get_sub_state_space(pmdp_env, graph)
    
    #sub_stateSpace_vars_sorted_list, sub_stateTag_tensor_list = get_states_for_Q_rows(pmdp_env, horizon_space, graph)
    #subActionSpace_tensor_list, result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list = get_actionSpace_for_Q_columns(pmdp_env, sub_stateSpace_vars_sorted_list) 


    ################# This part to replace 'get_actionSpace_for_Q_columns(.....)' #################
    # 'sub_stateSpace_vars_sorted_list' only need in which variables are controllable 
    C_sub_stateSpace_vars_sorted_list = []
    for var in sub_stateSpace_vars_sorted_list[0]:
        if var[1].startswith('C'):
            C_sub_stateSpace_vars_sorted_list.append(var)

    # For compatibility of old code
    C_sub_stateSpace_vars_sorted_list = [C_sub_stateSpace_vars_sorted_list]

    ActionSpace_var_list = []
    # NMDP_result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list is for compatibility in applyint the PMDP environment
    
    NMDP_result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list = []
    for _ in range(3):
        NMDP_result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list.append([])

    # One by one to handle the variable in the C_sub_stateSpace_vars_sorted_list[0]
    #end_index = len(C_sub_stateSpace_vars_sorted_list[0][0][2])
    end_index = -1
    for i, item in enumerate(C_sub_stateSpace_vars_sorted_list[0]):
        NMDP_result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[0].append(item[0])
        NMDP_result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[2].append([i, i+1])
        var_name = item[1]
        # 'end_index + 1' for filter the 'None' value of previous one variable
        # The number of 'None' will be increse accoding to previous number of variables, so + i
        #start_index = end_index + i + 1
        #end_index = start_index + len(item[2]) - i - 1
        #NMDP_result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[1].append([start_index, end_index])

        start_index = end_index + 1
        end_index = start_index + len(item[2]) - 1
        NMDP_result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[1].append([start_index, end_index])

        #print(len(item[2]))
        for var_value in item[2]:
            if var_name.startswith('C'):
                var_value = var_value.replace("[", "").replace("]", "")
            else:
                raise(ValueError(f"In CSSC-MDP executing in P-MDP environment, please use 'C...' as Controllable variable name, and 'U...' as Uncontrollable variable name, but got '{var_name}'"))
            dict_id = f"{var_name}={var_value}"
            
            
            var_tensor = all_string_to_int16[dict_id]
            tmp_var_list = [0] * len(C_sub_stateSpace_vars_sorted_list[0])
            tmp_var_list[i] = var_tensor
            
            for j in range(i):
                tmp_var_list[j] = all_string_to_int16[f"{C_sub_stateSpace_vars_sorted_list[0][j][0]}=None"]
            for j in range(i + 1, len(C_sub_stateSpace_vars_sorted_list[0])):
                tmp_var_list[j] = all_string_to_int16[f"{C_sub_stateSpace_vars_sorted_list[0][j][0]}=None"]
            
            ActionSpace_var_list.append(tmp_var_list)

    two_dim_tensor = torch.tensor(ActionSpace_var_list, dtype=torch_intType, device=learning_device)
    result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list = [NMDP_result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list]
    subActionSpace_tensor_list = [two_dim_tensor]

    ################# This part to replace 'get_actionSpace_for_Q_columns(.....)' #################


    subStateSpace_number = len(sub_stateSpace_vars_sorted_list)
    if subStateSpace_number != 1:
        raise ValueError("The number of subStateSpace must be '1' in CSSC-MDP")

    ######## Construct the Q-tables for CSSC, each activity has 4 states: 1, 2, 3, 4########
    #### Actually the 'subStateSpace_number' is only '1' in CSSC, but we still use the 'subStateSpace_number' to keep the same structure of the code
    Q_CSSC = {}
    sub_stateTag_tensor_list = []
    for i in range(subStateSpace_number):
        nS = 4 * len(result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0])
        nA = subActionSpace_tensor_list[i].size(0)
        
        # Use a dictionary to store the Q table for each sub-state space
        Q_CSSC[f'{i}'] = torch.zeros((nS, nA), dtype=torch.float32, device=learning_device)

        sub_stateTag_tensor_list.append(torch.empty(0, dtype=torch.float32, device=learning_device))

    ######## End of Construct the Q-tables for CSSC ########


    Q, Q_, Qkeys, subStateSpace_vars_name_list, subStateSpace_vertices_name_list, subStateSpace_vars_number_list, QH, QH_ = initialize_Q_tables(subStateSpace_number, sub_stateTag_tensor_list, 
                                                                                                                                                subActionSpace_tensor_list, sub_stateSpace_vars_sorted_list,
                                                                                                                                                filtered_QHorizonTag_StateSpace_tensor, filtered_QHorizon_ActionSpace_tensor)


    # The 'subState_HC_weights_list' and 'subState_SC_weights_list' have been computed with the 'whc' and 'wsc' weights
    # The 'subState_HC_weights_list' and 'subState_SC_weights_list' will store the HC and SC weights of each subStateSpace with the same order of the subStateSpace
    HC_list, SC_list, subState_HC_weights_list, subState_SC_weights_list, whc, wsc = classify_constraints_by_substateSpace(pmdp_env, subStateSpace_vars_name_list, whc, wsc)

    ### Special for the 'CSSC-MDP' and wsdreamDataset ###
    CSSC_MDP_SC_list = copy.deepcopy([[graph.SC[0], graph.SC[1]]])

    subStateSpace_portion_value_dict_list, sc_Sum_dict = calculate_var_portion_values(pmdp_env, n_abstract_number, subStateSpace_vars_name_list, subStateSpace_number, CSSC_MDP_SC_list, total_candidates_num = total_candidates_num)

    HC_weight_list = [0] * subStateSpace_number
    SC_weight_list = [0] * subStateSpace_number

    ConstraintState_list = []
    for i in range(subStateSpace_number):
    
        # For compatibilit, we need to add the empty list to the 'ConstraintState_list'
        ConstraintState_list.append([])


    accumulate_HC_SC_rewards_in_each_trajectory = []
    accumulate_HC_rewards_in_each_trajectory = []
    accumulate_SC_rewards_in_each_trajectory = []

    evaluation_rewards = []
    evaluation_HC_rewards = []
    evaluation_SC_rewards = []


    start_vertex = next((v for v in horizon_space if isinstance(v, str) and v.startswith('StartEvent')), None)

    start_time = time.time()

    for e in tqdm(range(n_episodes), leave=False):

    

        if e % 100 == 0 and e > 0:
            elapsed_time = time.time() - start_time
            Training_time_record.append(elapsed_time)
            #print(f"Episode {e}: Elapsed Time = {elapsed_time:.2f} seconds")

        done = False

        # get the start event of the BPMN model, that is the initial active vertex (initial state) of the PMDP model
        #activeVertices_of_State = next((v for v in horizon_space if isinstance(v, str) and v.startswith('StartEvent')), None)
        activeVertices_of_State = start_vertex

        # Get the active vertex of the state and follow the order of the sub-state space Q table
        active_vertices_of_State_verticesList = [start_vertex]

        # the 'rowsIndices' is the column index of the action in the state, i.e., 'Q,Q_'
        time_step = 0
        
        # The item of 'accumulate_rewards_list' represent the accumulate rewards list of the subStateSpace,
        # each item of 'accumulate_rewards_list' consist of [0] is accumulate rewards of the 'HC and SC' rewards of the subStateSpace, [1] is the accumulate rewards of the 'HC' rewards of the subStateSpace, [2] is the accumulate rewards of the 'SC' rewards of the subStateSpace
        accumulate_rewards_list = [[0, 0, 0] for _ in range(subStateSpace_number)]
        CSSC_accumulate_rewards_list = [[0, 0, 0] for _ in range(subStateSpace_number)]

        # Find vertices with no out-degree, i.e., the last vertex of the BPMN
        no_out_degree_vertices = copy.deepcopy(list(pmdp_env.graph.vertices.keys()))
        for edge in pmdp_env.graph.edges:
                from_vertex, to_vertex = edge.source, edge.target
                if from_vertex in no_out_degree_vertices:
                    no_out_degree_vertices.remove(from_vertex)

        while not done:
            # get the current state of the PMDP
            state = pmdp_env.current_state

            Q_CSSC_subStateSpace_ActionIndice_list = [[]] * subStateSpace_number
            Q_CSSC_action_tensor = torch.empty(0, dtype=torch_intType, device=learning_device)
            Q_CSSC_values_Q_list = [0] * subStateSpace_number
            Q_CSSC_values_QH = 0



            old_activeVertices_of_State = activeVertices_of_State
            if 'StartEvent' in old_activeVertices_of_State:
                old_state_level = 1
            else:
                old_state_level = new_state_level

            if pmdp_env.debug_mode:
                print(f"------------------------------------Current Time-Step: {time_step}-----------------------------------------------")
                print(f"Current active vertices: {active_vertices_of_State_verticesList}")
                Curret_active_vertice_with_names = []
                for vertex_id in active_vertices_of_State_verticesList:
                    vertex_name = pmdp_env.graph.vertices[vertex_id].elem_name
                    if vertex_id.startswith('empty'):
                            vertex_name = vertex_id
                    Curret_active_vertice_with_names.append(f"{vertex_name}:{vertex_id}")
                print("Curret_active_vertices with Name: ", Curret_active_vertice_with_names) 
                print(f"Current state: {state}")
                time_step += 1

            Q_state_rowIndexs, Q_actionSpace_columnIndices_list, Q_actionSpace_rowsIndices_list =  [[], [[]], [[]]]

            ########################## Q_CSSC is the Q-tables for CSSC, each activity has 4 states: 1, 2, 3, 4 ##########################
            ##### subStateSpace_number is only '1' in CSSC
            for i in range(subStateSpace_number):
                if old_activeVertices_of_State in result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0]:
                    active_index = result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0].index(old_activeVertices_of_State)
                    Q_CSSC_state_rowIndexs, Q_CSSC_actionSpace_columnIndices_list, Q_CSSC_actionSpace_rowsIndices_list = [[active_index * 4 + old_state_level - 1], 
                                                                                                                          [[result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][2][active_index]]],
                                                                                                                          [[result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][1][active_index]]],
                                                                                                                          ]
                    Q_CSSC_action_tensor, Q_CSSC_values_Q_list, Q_CSSC_values_QH, Q_CSSC_subStateSpace_ActionIndice_list, Q_CSSC_QH_ActionIndice_list, _ = generate_action_tensor(subStateSpace_number, Q_CSSC_actionSpace_columnIndices_list, Q_CSSC_actionSpace_rowsIndices_list, active_vertices_of_State_verticesList, result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list,
                                    Q_CSSC_state_rowIndexs, Q_CSSC, epsilons, e, subActionSpace_tensor_list, QH_state_rowIndex, torch.empty(0),
                                    QH_actionSpace_rowsIndices, QH_actionSpace_columnIndices, filtered_QHorizon_ActionSpace_tensor, all_int16_to_string_map, all_string_to_int16, learning_device)
            ########################## End of Q_CSSC is the Q-tables for CSSC, each activity has 4 states: 1, 2, 3, 4 ##########################


            # the 'QH_actionSpace_rowsIndices' is the column index of the action in the state, i.e., 'QH'
            QH_state_rowIndex, QH_actionSpace_columnIndices, QH_actionSpace_rowsIndices = get_horizon_state_indices(result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list, 
                                                                                                                       active_vertices_of_State_verticesList,all_string_to_int16)
            
            # Generate a complete action tensor for PMDP by Q and QH
            # The 'values_Q_list (for each subStatSpace)' and 'values_QH' are the sum of Q-values of the pairs of 'state-action' in 'Q,Q_'
            action_tensor, values_Q_list, values_QH, subStateSpace_ActionIndice_list, QH_ActionIndice_list, horizon_action_tensor_cssc = generate_action_tensor(subStateSpace_number, Q_actionSpace_columnIndices_list, Q_actionSpace_rowsIndices_list, active_vertices_of_State_verticesList, result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list,
                                        Q_state_rowIndexs, Q, epsilons, e, subActionSpace_tensor_list, QH_state_rowIndex, QH,
                                        QH_actionSpace_rowsIndices, QH_actionSpace_columnIndices, filtered_QHorizon_ActionSpace_tensor, all_int16_to_string_map, all_string_to_int16, learning_device)
            
            if old_activeVertices_of_State not in result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0]:
                Q_CSSC_action_tensor = torch.cat((Q_CSSC_action_tensor, horizon_action_tensor_cssc))
            # Make action and get 'the next state of the PMDP'
            # the 'activeVertices_of_State' is a horizon item, it is a string, tupple, or nested tuple of strings
            next_activeVertices_of_State = update_state_and_parse_next(
                pmdp_env, 
                #action_tensor, 
                Q_CSSC_action_tensor,
                all_int16_to_string_map, 
                activeVertices_of_State,
                frequency_prob_dict = frequency_prob_dict,
                MDP_example_mode = MDP_example_mode
            )
          
            # for next time step, update the active vertices of the state
            activeVertices_of_State = next_activeVertices_of_State

            # the first parameter is the weight for the hard constraints, and the second parameter is the weight for the soft constraints
            # immediate_rewards = immediate_rewards_HC_rewards + immediate_rewards_SC_rewards
            #immediate_rewards, immediate_rewards_HC_rewards, immediate_rewards_SC_rewards = pmdp_env.calculate_reward(0.5, 0.5)
            # here is not related to the horizonSpace reward
            # [0, ] is sum of the 'HC and SC' rewards list of the subStateSpace, [1, ] is the sum of the 'HC' rewards list of the subStateSpace, [2, ] is the sum of the 'SC' rewards list of the subStateSpace, [3] is the 'All_Rewards_All_subStates'
            
            #immediate_rewards_list, Column is the index of subStateSpace
            #accumulate_rewards_list, Row is the index of subStateSpace
            immediate_rewards_list, accumulate_rewards_list, _, _ = pmdp_env.calculate_reward(whc, wsc, HC_list, SC_list, ConstraintState_list, False, False, accumulate_rewards_list, subStateSpace_number = subStateSpace_number)


            ######################### Use CSSC reward calculation ########################################
            CSSC_immediate_rewards_list, CSSC_accumulate_rewards_list, new_state_level = CSSC_calculate_reward(pmdp_env, subStateSpace_portion_value_dict_list, sc_Sum_dict, old_activeVertices_of_State, subStateSpace_vars_name_list, whc, wsc, HC_list, CSSC_MDP_SC_list, CSSC_accumulate_rewards_list, old_state_level = old_state_level, subStateSpace_number = subStateSpace_number)

            # check if the current state is a terminal state
            if activeVertices_of_State is None:
                # Check if the 'active_vertices' represent any vertex in the edges as a target but not as a source
                # i.e., the end event of bpmn model
                done = True
            else:
                active_vertices_of_State_verticesList = pmdp_env.convert_horizon_item_to_list(next_activeVertices_of_State)
            


            if not done:
                ########################## Q_CSSC is the Q-tables for CSSC, each activity has 4 states: 1, 2, 3, 4 ##########################
                ##### subStateSpace_number is only '1' in CSSC
                if old_activeVertices_of_State in result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0]:
                    next_values_Q_CSSC_list = [0] * subStateSpace_number
                    # Calculate the TD error for each subStateSpace
                    if no_out_degree_vertices[0] != next_activeVertices_of_State:
                        for i in range(subStateSpace_number):
                            active_index = result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list[i][0].index(next_activeVertices_of_State)
                            #next_values_Q_columnIndices_list.append([])
                            for j in range(len(Q_CSSC_actionSpace_rowsIndices_list[i])):
                                row_start, row_end = Q_CSSC_actionSpace_rowsIndices_list[i][j][0], Q_CSSC_actionSpace_rowsIndices_list[i][j][1]
                                # TD error must use greedy action
                                subStateSpace_ActionIndice = select_action(active_index * 4 + new_state_level - 1, row_start, row_end, Q_CSSC[f'{i}'], epsilons[e], only_greedy=True)
                                next_values_Q_CSSC_list[i] =  Q_CSSC[f'{i}'][active_index * 4 + new_state_level - 1, subStateSpace_ActionIndice].item()
                                #next_values_Q_columnIndices_list[i] = [row_start, row_end]

                        CSSC_TD_error_list = [0] * subStateSpace_number
                        for i in range(subStateSpace_number):
                            # Some time, the transition of PMDP is not related to the subStateSpace (rather, about QH), so 'values_Q_list' will not have the value of the subStateSpace, we need to add 0 to the list for compatibility of the PMDP learning
                            while len(Q_CSSC_values_Q_list) <= i:
                                Q_CSSC_values_Q_list.append(0)
                            while len(next_values_Q_CSSC_list) <= i:
                                next_values_Q_CSSC_list.append(0)
                            TD_target = CSSC_immediate_rewards_list[0][i] + gamma * next_values_Q_CSSC_list[i]
                            TD_error_value = TD_target - Q_CSSC_values_Q_list[i]
                            CSSC_TD_error_list[i] = TD_error_value
                    else:
                        ## Final state, the next state is None
                        for i in range(subStateSpace_number):
                            TD_target = CSSC_immediate_rewards_list[0][i] + gamma * 0
                            TD_error_value = TD_target - Q_CSSC_values_Q_list[i]
                            CSSC_TD_error_list[i] = TD_error_value
                    # Update the Q-tables for CSSC
                    for i in range(subStateSpace_number):
                       if len(Q_CSSC_subStateSpace_ActionIndice_list[i]) > 0:
                            Q_CSSC[f'{i}'] = update_fast_Q(Q_CSSC[f'{i}'], [Q_CSSC_state_rowIndexs[i]], Q_CSSC_subStateSpace_ActionIndice_list[i], 1, alphas, e, CSSC_TD_error_list[i])
        
                ########################## End of Q_CSSC is the Q-tables for CSSC, each activity has 4 states: 1, 2, 3, 4 ##########################

        ################# Evaluate the CSSS-MDP Policy #################
        if ((e >  1 and e % eval_interval == 0) or (e == n_episodes - 1)) and Evaluate_mode:
            success_rate, avg_HC_reward, avg_SC_reward = evaluate_cssc_policy(Q_CSSC,
                                                                            QH,
                                                                            eval_number,
                                                                            subStateSpace_number,
                                                                            result_ActionSpaceverticesOrder_rowsColumnsTags_inTensor_list,
                                                                            result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list,
                                                                            all_string_to_int16,
                                                                            all_int16_to_string_map,
                                                                            subActionSpace_tensor_list,
                                                                            filtered_QHorizon_ActionSpace_tensor,
                                                                            torch_intType,
                                                                            learning_device,
                                                                            pmdp_env,
                                                                            start_vertex,
                                                                            frequency_prob_dict,
                                                                            MDP_example_mode,
                                                                            new_state_level,
                                                                            select_action,
                                                                            get_horizon_state_indices,
                                                                            update_state_and_parse_next,
                                                                            whc,
                                                                            wsc,
                                                                            HC_list,
                                                                            SC_list,
                                                                            ConstraintState_list,
                                                                            CSSC_calculate_reward,
                                                                            subStateSpace_portion_value_dict_list,
                                                                            sc_Sum_dict,
                                                                            subStateSpace_vars_name_list,
                                                                            CSSC_MDP_SC_list)    
            if e % eval_interval == 0  or (e == n_episodes - 1):
                print(f"\n################ CSSC-MDP Episode {e}, Average Success Rate: {success_rate} ################")
                #print(f"################ CSSC-MDP Episode {e}, Average Average Total Reward: {avg_SC_reward + avg_HC_reward} ################")
            '''
            if e == n_episodes - 1:
                evaluation_rewards[-1] = avg_reward
                evaluation_HC_rewards[-1] = avg_HC_reward
                evaluation_SC_rewards[-1] = avg_SC_reward
            else:
            '''
            evaluation_rewards.append(success_rate)
            evaluation_HC_rewards.append(avg_HC_reward)
            evaluation_SC_rewards.append(avg_SC_reward)
        
    CSSC_evaluation_rewards_list.append(evaluation_rewards)
    CSSC_evaluation_HC_rewards_list.append(evaluation_HC_rewards)
    CSSC_evaluation_SC_rewards_list.append(evaluation_SC_rewards)

        ################# End of Evaluate the CSSS-MDP Policy #################


    return 

############################ The parts of the CSSC_MDP_with_Qlearning ending ########################################



##################### Optimized learning about Cdoube_Qlearning #####################


# sigma_init=0.017
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Trainable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.sigma_init)
    
    def forward(self, input):
        if self.training:
            #self.weight_epsilon.normal_()
            #self.bias_epsilon.normal_()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()


    def reset_parameters(self, zero_init=False):
        if zero_init:
            nn.init.constant_(self.weight_mu, 0)
            nn.init.constant_(self.weight_sigma, 0)
            nn.init.constant_(self.bias_mu, 0)
            nn.init.constant_(self.bias_sigma, 0)
        else:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.weight_mu, -bound, bound)
            nn.init.constant_(self.weight_sigma, self.sigma_init)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_sigma, self.sigma_init)


class LowRankEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rank=4):
        super(LowRankEmbedding, self).__init__()
        self.A = nn.Embedding(num_embeddings, rank)
        self.B = nn.Parameter(torch.randn(rank, embedding_dim))

    def forward(self, x):
        return self.A(x) @ self.B


class StateActionNetwork(nn.Module):
    def __init__(self, state_var_sizes, action_var_sizes, embedding_dim=16):
        super(StateActionNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        # Calculate the starting index of state and action variables
        self.state_var_offsets = np.cumsum([0] + state_var_sizes[:-1])
        self.action_var_offsets = np.cumsum([0] + action_var_sizes[:-1])

        # Convert to tensor
        self.state_var_offsets = torch.tensor(self.state_var_offsets, dtype=torch.long)
        self.action_var_offsets = torch.tensor(self.action_var_offsets, dtype=torch.long)

        # Use EmbeddingBag
        total_state_categories = sum(state_var_sizes)
        total_action_categories = sum(action_var_sizes)
        self.state_embeddings = nn.EmbeddingBag(total_state_categories, embedding_dim, mode='mean', sparse=True)
        self.action_embeddings = nn.EmbeddingBag(total_action_categories, embedding_dim, mode='mean', sparse=True)

        # Define fully connected layers
        total_embedding_dim = embedding_dim * 2
        self.fc1 = NoisyLinear(total_embedding_dim, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.output = NoisyLinear(128, 1)

    def forward(self, state_indices, action_indices):
        batch_size = state_indices.size(0)

        # Move offsets to input device
        state_offsets = self.state_var_offsets.to(state_indices.device)
        action_offsets = self.action_var_offsets.to(action_indices.device)

        # Adjust indices
        state_indices_global = state_indices + state_offsets  # Shape: [batch_size, num_state_vars]
        action_indices_global = action_indices + action_offsets  # Shape: [batch_size, num_action_vars]

        # Prepare input for EmbeddingBag
        state_indices_flat = state_indices_global.view(-1)
        action_indices_flat = action_indices_global.view(-1)

        # Generate offsets for each sample
        state_offsets_bag = torch.arange(0, batch_size * state_indices.size(1), state_indices.size(1), device=state_indices.device)
        action_offsets_bag = torch.arange(0, batch_size * action_indices.size(1), action_indices.size(1), device=action_indices.device)

        # Get embeds and automatically aggregate
        state_embed = self.state_embeddings(state_indices_flat, state_offsets_bag)
        action_embed = self.action_embeddings(action_indices_flat, action_offsets_bag)

        # Merge embeddings and pass through fully connected layers
        x = torch.cat((state_embed, action_embed), dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.output(x)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.output.reset_noise()


# Optimizer settings
# default epsilon=1e-8
def get_optimizers(model, learning_rate=0.005, epsilon=1e-8):
    # Sparse parameters (Embedding layers)
    sparse_params = list(model.state_embeddings.parameters()) + list(model.action_embeddings.parameters())

    # Dense parameters (All other layers)
    dense_params = []
    for name, param in model.named_parameters():
        if 'state_embeddings' not in name and 'action_embeddings' not in name:
            dense_params.append(param)

    # Define SparseAdam optimizer for embedding layers
    optimizer_sparse = optim.SparseAdam(sparse_params, lr=learning_rate, eps=epsilon)

    # Define Adam optimizer for other layers with specified epsilon
    optimizer_dense = optim.Adam(dense_params, lr=learning_rate, eps=epsilon)

    return optimizer_sparse, optimizer_dense


# Get the index representation of the status
def get_state_indices(state, state_var_maps):
    # state is a list containing multiple state variable values, for example ['A', 'X', 'P']
    state_indices = [state_var_maps[var][value] for var, value in zip(state_var_maps.keys(), state)]
    return torch.tensor(state_indices, dtype=torch.long, device=learning_device)

# Get the index representation of the action
def get_action_indices(action, action_var_maps):
    # action is a list containing multiple action variable values, for example ['B', 'Y']
    action_indices = [action_var_maps[var][value] for var, value in zip(action_var_maps.keys(), action)]
    return torch.tensor(action_indices, dtype=torch.long, device=learning_device)


def get_action_pmdp_action_tensor(action_indices, action_var_reverse_maps, all_string_to_int16):
    action_tensor = []
    for var, index in zip(action_var_reverse_maps.keys(), action_indices):
        value = action_var_reverse_maps[var][index.item()]
        if 'None' in value or 'finish' in value or 'wait' in value:
            action_tensor.append(all_string_to_int16[value])
        else:
            action_tensor.append(all_string_to_int16[f'{var}={value}'])
    
    return torch.tensor(action_tensor, dtype=torch_intType, device=learning_device)


# Action selection function
def SAN_select_action(state_indices, valid_actions, q_network, epsilon=0.1):
    if random.random() < epsilon:
        # Randomly select a valid action
        selected_action = random.choice(valid_actions)
    else:
        # Get the index representation of the status and convert it to a tensor
        #state_indices = get_state_indices(state, state_var_maps).unsqueeze(0)  # [1, num_state_vars]
        state_indices = state_indices.repeat(len(valid_actions), 1)  # [num_valid_actions, num_state_vars]

        # Convert valid actions to action index tensor
        action_indices = torch.stack([action_indice for action_indice in valid_actions])  # [num_valid_actions, num_action_vars]
        
        # Calculate all Q values at once through the network
        with torch.no_grad():
            q_values = q_network(state_indices, action_indices).squeeze()  # [num_valid_actions]

        # Select the action with the maximum Q value
        max_q_index = torch.argmax(q_values)
        selected_action = valid_actions[max_q_index.item()]
    return selected_action

def SAN_select_action_noisy(state_indices, valid_actions, q_network):
    # Reset noise
    #q_network.reset_noise()

    # Repeat state indices to match the number of valid actions
    state_indices = state_indices.repeat(len(valid_actions), 1)  # [num_valid_actions, num_state_vars]

    # Convert valid actions to action index tensor
    action_indices = torch.stack([action for action in valid_actions]).long().to(learning_device)  # [num_valid_actions, num_action_vars]

    # Calculate all Q values at once through the network
    with torch.no_grad():
        q_values = q_network(state_indices, action_indices).squeeze()  # [num_valid_actions]

    # Select the action with the maximum Q value
    max_q_index = torch.argmax(q_values)
    selected_action = valid_actions[max_q_index.item()]
    
    return selected_action



# Choose actions during evaluation, do not add noise, but if random_policy is True, add noise and select the action randomly
def select_action_evaluation_policy(state_indices, valid_actions, q_network, random_policy = False):

    if random_policy:
        ### using introduce noise to select the action randomly
        q_network.reset_noise()

    # Repeat the state index to match the number of valid actions
    state_indices = state_indices.repeat(len(valid_actions), 1)  # [num_valid_actions, num_state_vars]

    # Convert valid actions to action index tensor
    action_indices = torch.stack([action for action in valid_actions]).long().to(learning_device)  # [num_valid_actions, num_action_vars]

    # Calculate all Q values at once through the network
    with torch.no_grad():
        q_values = q_network(state_indices, action_indices).squeeze()  # [num_valid_actions]

    # Select the action with the maximum Q value
    max_q_index = torch.argmax(q_values)
    if random_policy:
        max_q_index = torch.randint(0, len(valid_actions), (1,))
    selected_action = valid_actions[max_q_index.item()]
    
    return selected_action

def get_valid_actions(pmdp_env : PMDPEnvironment, active_vertices_of_State_verticesList, action_var_maps, vars_dict, NMDP_mode = False):
    valid_actions = []
    active_vars_dict = {}
    possible_values_vars_list = []

    # The 'action_var_maps' has ordered the action variables in the same order as the 'pmdp_actions_list'
    for var_name, var_value in action_var_maps.items():
        if not var_name.startswith('empty_') and not vars_dict[var_name].vertex_id in active_vertices_of_State_verticesList:
            # The 'None' action is always used 0 as the index
            possible_values_vars_list.append(torch.tensor([0], dtype=torch.long, device=learning_device))
        elif not var_name.startswith('empty_'):
            # the var is a controllable variable, action only contains the controllable variables domain values
            # The 'domain values' is always used  3 to len(action_var_maps[var_name])-1 as the index

            # Generate a list from 3 to len(action_var_maps[var_name]) - 1
            #index_list = list(range(3, len(action_var_maps[var_name])))
            if not NMDP_mode:
                # new version has no 'wait' action, so the index is from 2 to len(action_var_maps[var_name])-1
                index_list = list(range(2, len(action_var_maps[var_name])))
            else:
                # Only in the NMDP mode, we add another 27 state to model it about 8 QoS is stateless, due to NMDP paper (weakness: Only support single variable in same activity to max, min , or a norm filter)
                # so it has no meaning to classify the each variable values state, if classify the each variable values state, only waste the calculation resource to do nomeaning work
                index_list = list(range(3, len(action_var_maps[var_name])))
                if len(index_list) != 50:
                    raise ValueError(f"Error: The length of the controllable variable domain is not equal to 50 (we only realize the NMDP paper same experiment Example, not all), the length is {len(index_list)}, please revise it")
                    
            possible_values_vars_list.append(torch.tensor(index_list, dtype=torch.long, device=learning_device))
        elif var_name.startswith('empty_') and not var_name in active_vertices_of_State_verticesList:
            # The 'None' action is always used 0 as the index
            possible_values_vars_list.append(torch.tensor([0], dtype=torch.long, device=learning_device))
        else:
            # the var is about the last vertex of parallel thread, for 'finish' and 'wait' actions
            # The 'finish' action is always used 1 as the index, （old version）'wait' action is always used 2 as the index (new version has no 'wait' action)
            #possible_values_vars_list.append(torch.tensor([1,2], dtype=torch.long, device=learning_device))
            possible_values_vars_list.append(torch.tensor([1], dtype=torch.long, device=learning_device))
  
    valid_actions = torch.cartesian_prod(*possible_values_vars_list)
    
    return valid_actions



def evaluate_policy(pmdp_env: PMDPEnvironment, q_network, num_episodes, pmdp_states_list, action_var_maps, state_var_maps, vars_dict, start_vertex, whc, wsc, punish_mode, parallel_asynchronous_prob, all_string_to_int16, all_int16_to_string_map, CSSC_mode, MDP_example_mode, frequency_prob_dict, action_var_reverse_maps, NMDP_mode = False, 
                    uncontrol_var_unknow_distributions=None, mode = 'any', random_policy = False):
    

    # Deactivate 'noisy' mode, so will be out of the training mode
    q_network.eval()  # Set to evaluation mode

    total_rewards = []
    total_HC_rewards = []

    ### store the accumulated rewards for each episodes
    AccumulatedRewards_records = [[], [], []]

    
    if CSSC_mode:

        ### Due to wsDream2 dataset, the responseTime mean is 1.4s, so when the activity more than 20 will result to the direct sample trajectories that is very rarely to sample the trajectories that is less than 23s (no more 3%), resulting P-MDP cannot get any signal about it
        ### Make a deep copy of the 'pmdp_env' to avoid the 'pmdp_env' is changed by the following codes
        ### According this code, when CSSC_mode is True, the evaluation phase will use the same constraints as the 'CSSC-MPDP' model
        pmdp_env = copy.deepcopy(pmdp_env)
        # just for debug
        pmdp_env.rt_count = 1

        
        tmp_tp_weights = []

        # Activities 30 cases
        for constraint in pmdp_env.graph.SC:
            if 'RT<' in constraint.expression:
                constraint.weight = 0.5
            if '\wedge' in constraint.expression:
                tmp_tp_weights.append(constraint.weight)
                constraint.weight = 0.0
            if 'TP<' in constraint.expression:
                constraint.weight = 0.5
        

        '''
        ########################### This codes only for 'Paper Experiment' when compare the 'CSSC-MPDP' model ############################
        # CSSC_mode when evaluation phase, we need to deactive the constraints used in the learning phase that is used to train the Q-network
        # we need to active the constraints that is used in the evaluation phase, that use the same constraints as the 'CSSC-MPDP' model

        
        if episode < train_total_episodes * 0.25: 
            ### Due to wsDream2 dataset, the responseTime mean is 1.4s, so when the activity more than 20 will result to the direct sample trajectories that is very rarely to sample the trajectories that is less than 23s (no more 3%), resulting P-MDP cannot get any signal about it
            ### Make a deep copy of the 'pmdp_env' to avoid the 'pmdp_env' is changed by the following codes
            ### According this code, when CSSC_mode is True, the evaluation phase will use the same constraints as the 'CSSC-MPDP' model
            pmdp_env = copy.deepcopy(pmdp_env)
            # just for debug
            pmdp_env.rt_count = 1

        elif episode < train_total_episodes * 0.5:
            for constraint in pmdp_env.graph.SC:
                if 'RT<23' in constraint.expression:
                    constraint.weight = 0.3
                if 'RTK1' in constraint.expression:
                    constraint.weight = 0.2
            pmdp_env = copy.deepcopy(pmdp_env)
            # just for debug
            pmdp_env.rt_count = 1

        else:
            for constraint in pmdp_env.graph.SC:
                if 'RT<23' in constraint.expression:
                    constraint.weight = 0.4
                if 'RTK1' in constraint.expression:
                    constraint.weight = 0.1
            pmdp_env = copy.deepcopy(pmdp_env)
            # just for debug
            pmdp_env.rt_count = 1
            ###### when sample and guiding the q-network more than 30% of the total episodes about 'ResponseTime', then use the standard constraints via never use deepcopy######

        '''


    for _ in tqdm(range(num_episodes), leave=False):
    #for _ in range(num_episodes):

        done = False
        activeVertices_of_State = start_vertex
        pmdp_env.current_state = pmdp_env.initialize_state()
        pmdp_env.context = pmdp_env.build_pmdp_context()
        pmdp_env.reset_KPIs_Constraints_to_ini_stages()
        accumulate_rewards_list = [[0, 0, 0]]

        activeVertices_of_State = start_vertex
        active_vertices_of_State_verticesList = [start_vertex]

        HC_list = [pmdp_env.graph.HC]
        SC_list = [pmdp_env.graph.SC]

        ConstraintState_list = [[]]

        state = get_current_SAN_state(pmdp_env, pmdp_states_list, state_var_maps, vars_dict)
        done = False
        step = 0

        vertex_wait_times = {}

        while not done:
            # Get the set of valid actions in the current state, assuming that each action is a list containing num_action_vars elements
            valid_actions = get_valid_actions(pmdp_env, active_vertices_of_State_verticesList, action_var_maps, vars_dict, NMDP_mode)

            # Select action
            #epsilon = max(0.01, 0.1 - 0.01 * (episode / 200))
            epsilon = 0

            for active_vertex_id in active_vertices_of_State_verticesList:
                try:
                    vertex_wait_times[active_vertex_id] += 1
                    if vertex_wait_times[active_vertex_id] > 2:
                        epsilon = 1
                except KeyError:
                    vertex_wait_times[active_vertex_id] = 1
                

            #action = SAN_select_action(state, valid_actions, q_network, epsilon)

            
            action = select_action_evaluation_policy(state, valid_actions, q_network, random_policy = random_policy)

            pmdp_action_tensor = get_action_pmdp_action_tensor(action, action_var_reverse_maps, all_string_to_int16)

             # Make action and get 'the next state of the PMDP'
            # the 'activeVertices_of_State' is a horizon item, it is a string, tupple, or nested tuple of strings
            if CSSC_mode:
                        # CSSC_mode, the distribution of uncontrollable variables is set as 'unknown', so need to use the 'frequency_prob_dict' to get the distribution of the uncontrollable variables
                        next_activeVertices_of_State = update_state_and_parse_next(
                        pmdp_env, 
                        pmdp_action_tensor,
                        all_int16_to_string_map, 
                        activeVertices_of_State,
                        frequency_prob_dict = frequency_prob_dict,
                        MDP_example_mode = MDP_example_mode
                        )

            else:
                next_activeVertices_of_State = update_state_and_parse_next(
                    pmdp_env,
                    pmdp_action_tensor,
                    all_int16_to_string_map,
                    activeVertices_of_State,
                    parallel_asynchronous_prob,
                    uncontrol_var_unknow_distributions = uncontrol_var_unknow_distributions,
                    mode = mode
                )
          
            # for next time step, update the active vertices of the state
            activeVertices_of_State = next_activeVertices_of_State

            #immediate_rewards_list, The column is the index of subStateSpace,, here only one subStateSpace, so use Rewards_HC_plus_SC_list[0][0] to index the immediate reward   
            # immediate_rewards_list is consist of '[Rewards_HC_plus_SC_list, Rewards_HC_list, Rewards_SC_list, All_Rewards_All_subStates]'       
            #accumulate_rewards_list, the index of the subStateSpace of the behavior, from 0 to 2, accumulate_HC_plus_SC_rewards, accumulate_HC_rewards, accumulate_SC_rewards
            immediate_rewards_list, accumulate_rewards_list, ConstraintState_list, evaluated_constraintsName_list = pmdp_env.calculate_reward(whc, wsc, HC_list, SC_list, ConstraintState_list, punish_mode, False, accumulate_rewards_list, subStateSpace_number = 1)

            # check if the current state is a terminal state
            if activeVertices_of_State is None:
                # Check if the 'active_vertices' represent any vertex in the edges as a target but not as a source
                # i.e., the end event of bpmn model
                done = True
            else:
                # update the 'active_vertices_of_State_verticesList' for next time step
                active_vertices_of_State_verticesList = pmdp_env.convert_horizon_item_to_list(next_activeVertices_of_State)
             
            
            #next_state, reward, done, _ = env.step(action)
            next_state = get_current_SAN_state(pmdp_env, pmdp_states_list, state_var_maps, vars_dict, NMDP_mode)

            
            state = next_state
        total_rewards.append(accumulate_rewards_list[0][0])
        total_HC_rewards.append(accumulate_rewards_list[0][1])

        ### AccumulatedRewards_records is used to store the accumulated rewards for each episodes for the random policy experiment
        AccumulatedRewards_records[0].append(accumulate_rewards_list[-1][0])
        AccumulatedRewards_records[1].append(accumulate_rewards_list[-1][1])
        AccumulatedRewards_records[2].append(accumulate_rewards_list[-1][2])


    tmp_sum_total_rewards = 0
    tmp_sum_HC_rewards = 0
    # The python library 'sum()' function will be error in 'float' call error, so use the 'for' loop to calculate the sum of the rewards
    for i in range(len(total_rewards)):
        tmp_sum_total_rewards += total_rewards[i]
        tmp_sum_HC_rewards += total_HC_rewards[i]

        #for rewards in total_HC_rewards:
            #tmp_sum_HC_rewards += rewards


    if not CSSC_mode:
        avg_reward = tmp_sum_total_rewards / len(total_rewards)
        avg_HC_reward = tmp_sum_HC_rewards/ len(total_HC_rewards)
        
        # Set the model to training mode
        q_network.train()
        
        return avg_reward, avg_HC_reward, AccumulatedRewards_records
    
    else:
        # Calculate the proportion of total_rewards with a value of 1
        count_ones = 0
        for reward in total_rewards:
            if reward == 1:
                count_ones += 1
        success_service_compostions_proportion = count_ones / len(total_rewards) if total_rewards else 0
        avg_reward = tmp_sum_total_rewards / len(total_rewards)

        # Set the model to training mode
        q_network.train()

        return success_service_compostions_proportion, avg_reward, AccumulatedRewards_records
       


def train_q_network_noisy(q_network, target_network, PER_Buffer, warmup_tag, optimizer_sparse, optimizer_dense, batch_size, loss_values):
    #if len(PER_Buffer) < (PER_Buffer.batch_size * n_warmup_batches):
    if warmup_tag == False or len(PER_Buffer) < (PER_Buffer.batch_size * 1):
        return
    

    #episode_beta_increment = (beta_increment) / 11
    #PER_Buffer.beta = min(1.0, PER_beta)
    
    td_errors_for_PER_Buffer = []
    idxs, weights, samples = PER_Buffer.sample(batch_size=batch_size)


    transitions = samples

    batch_state, batch_action, batch_reward, batch_next_state, batch_done, batch_valid_actions, batch_next_gammas = zip(*transitions)

    # Make sure all tensors are on the correct devices
    batch_state = torch.stack(batch_state).long().to(learning_device)  # [batch_size, num_state_vars]
    batch_action = torch.stack([action for action in batch_action]).long().to(learning_device)  # [batch_size, num_action_vars]
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=learning_device)  # [batch_size]
    batch_next_state = torch.stack(batch_next_state).long().to(learning_device)  # [batch_size, num_state_vars]
    batch_done = torch.tensor(batch_done, dtype=torch.float32, device=learning_device)  # [batch_size]
    weights = weights.clone().detach().to(learning_device) # [batch_size]

    # reset the Q-network noise
    q_network.reset_noise()
    
    #2025-01-17, FOR NOISE NETWORK, each decision in training phase, the noise network should be reset, so the target network should be reset too
    # Only when evalution and deployment, the noise network should be closed via 'q_network.eval()' and don't need to reset the noise when each decision
    target_network.reset_noise()

    # Calculate the current Q value, and with gradient
    q_values = q_network(batch_state, batch_action).squeeze(-1)  # [batch_size]


    # Calculate the maximum Q value for the next state
    max_next_q_values = []
    with torch.no_grad():
        for i in range(len(batch_next_state)):
            if batch_done[i]:
                max_next_q_values.append(0.0)
            else:
                next_state = batch_next_state[i].unsqueeze(0)  # [1, num_state_vars]
                valid_actions = batch_valid_actions[i]
                if len(valid_actions) == 0:
                    max_next_q_values.append(0.0)
                    continue
                # Repeat the next state to match the number of valid actions
                next_state_repeated = next_state.repeat(len(valid_actions), 1)  # [num_valid_actions, num_state_vars]
                action_indices = torch.stack([action for action in valid_actions]).long().to(learning_device)  # [num_valid_actions, num_action_vars]

                # Calculate the Q values for all valid actions
                q_values_next = target_network(next_state_repeated, action_indices).squeeze()  # [num_valid_actions]
                # Get the maximum Q value
                max_q_value = q_values_next.max().item()
                max_next_q_values.append(max_q_value)
    
    max_next_q_values = torch.tensor(max_next_q_values, dtype=torch.float32, device=learning_device)  # [batch_size]

    # Calculate the target Q values
    #if PER_update_mode:
    gamma_tensor = torch.tensor(batch_next_gammas, dtype=torch.float32, device=learning_device)
    target_q_values = batch_reward + gamma_tensor * max_next_q_values * (1 - batch_done)
    #else:
        #target_q_values = batch_reward + gamma * max_next_q_values * (1 - batch_done)
    


    
    td_errors_for_PER_Buffer = target_q_values - q_values
    # Update the priorities of the samples
    PER_Buffer.update(idxs, td_errors_for_PER_Buffer)



    # Compute Huber loss
    # 计算 Huber 损失（Smooth L1 Loss），考虑样本权重
    #criterion = nn.SmoothL1Loss(beta=1.0, reduction='none')  # 设置为 'none' 以获取每个样本的损失
    #loss_elements = criterion(q_values, target_q_values)
    #weighted_loss = weights * loss_elements
    #loss = weighted_loss.mean()

   
    # 计算普通损失，考虑样本权重
    #loss = ((weights * (q_values - target_q_values)) ** 2).mean()

    #2025-01-17
    loss = ((weights * (q_values - target_q_values)) ** 2 * 0.5).mean()

    optimizer_sparse.zero_grad()
    optimizer_dense.zero_grad()
    loss.backward()
    optimizer_sparse.step()
    optimizer_dense.step()
    
    '''
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    '''
    
    # Record loss value
    loss_values.append(loss.item())

    return

def process_gateway_list(input_list):
    """
    Processes a list containing elements starting with "Gateway_"
    
    Parameters:
        input_list: The list to be processed
        
    Returns:
        The processed list, returns empty list if condition is met, otherwise returns original list
    """
    # Calculate total number of elements in the list (n)
    n = len(input_list)
    
    # Calculate number of elements starting with "Gateway_" (g_n)
    g_n = sum(1 for item in input_list if str(item).startswith("Gateway_"))
    
    # Check condition and return processed list
    if n - g_n - 2 == 0:
        return []
    return input_list


def Initialize_statesActions_PMDPMappingDict(env : PMDPEnvironment, graph: BaseGraph):

    global wait_int16_to_string_map
    global none_int16_to_string_map

    # Each item in sub_vars_list stores a series of variables that are related to each other due to constraints, forming a sub state space. 
    # Variables in the same subspace need to be combined using the Cartesian product to construct the sub state space.
    # There is no relationship between each item, and there are no duplicate variables within each item, so each item forms an independent sub state space.
    # Each sub state space is a part of the entire state space, which form a sub Q-table for the Q-function, which model state-action pair about variables that only belong to the sub state space.
    # Finally, all sub Q-tables are combined to form the entire Q-table for the Q-function.
    # The function 'classify_vars' is used to classify the variables in the constraints into different sub state spaces.
    constrainst_list = graph.HC + graph.SC
    # subStateSpace_vars_list containts the names of variables, not the objects of variables
    subStateSpace_vars_list = classify_vars(env, constrainst_list)
    subStateSpace_vars_list = merge_intersections(subStateSpace_vars_list)

    # vars_dict is a dictionary, which stores the objects of variables
    vars_dict = retrieve_variables_from_graph(graph)

    # 'all_vars_in_constraints' only contain names of variables
    all_vars_in_constraints = []
    for sub_vars in subStateSpace_vars_list:
        all_vars_in_constraints.extend(sub_vars)

    # check if there are variables that are not related to any constraints, so they are not included in the any state space
    # 'all_vars_in_StateSpace' only contain names of variables
    all_vars_in_StateSpace = []
    for sub_vars in subStateSpace_vars_list:
        all_vars_in_StateSpace.extend(sub_vars)

    vars_without_relationship_about_Q = get_non_intersecting_elements(vars_dict.keys(), all_vars_in_StateSpace) 
    if len(vars_without_relationship_about_Q) > 0:
        if len(vars_without_relationship_about_Q) == 2 and vars_without_relationship_about_Q[0].startswith('StartEvent_') or vars_without_relationship_about_Q[1].startswith('StartEvent_'):
            # If the 'StartEvent' and 'EndEvent' in BPMN is no have variable, it is ok, so we can ignore it
            vars_without_relationship_about_Q = []
            pass
        if process_gateway_list(vars_without_relationship_about_Q) == []:
            # If the 'Gateway' in BPMN is no have variable, it is ok, so we can ignore it
            vars_without_relationship_about_Q = []
        else:
            # if there are variables that are not related to any constraints, then they are not included in the any state space, so they are not included in the Q-table
            print(f"Warning: the following variables are not included in the Q-Function, because they are not related to any constraints or other variables: {vars_without_relationship_about_Q}")

    sub_stateTag_tensor_list = []
    sub_stateSpace_vars_sorted_list = []
    
    # Non-Neural Network based Q-learning is used 'subStateSpace_vars_list' to classify the variables in the constraints into different sub state spaces
    # here use it for compatibility with the 'earlier' version of the code
    # For SAN-based Q-learning, the below 'loop' is not used to classify the variables in the constraints into different sub state spaces
    # Rather, the for loop is used to construct the all states to int16 tensor at each variable dimension based on the domain of variables
    for subStateSpace_vars in subStateSpace_vars_list:
        var_list = []
        variables_spaces = []
        # get var information
        for var_name in subStateSpace_vars:
            var = vars_dict[var_name]
            var_list.append([var.vertex_id, var.name, var.domain, var.controlType])

        # sort the variable based on the vertex_id
        # make sure the order of variable is consistent comply with vertex_id, in each execution when call 'sort_and_group_var_list' function
        var_list.sort(key=lambda x: (x[0]))

        # sort the variable, make sure the order of variable is consistent in each execution when constructing the states space for rows of Q-table
        # sort the variable based on the vertex_id, and then based on the name of variable
        # Use ‘var.vertex_id’ as the sort key to ensure that variables in the same vertex are continuous in the sorted var_list, so that the filter_tensor function can correctly filter illegal actions
        var_list = sort_and_group_var_list(var_list)
        sub_stateSpace_vars_sorted_list.append(var_list)

        # Initialize an empty list for the parised variable domain    
        varDomain_list = []
        temp_wait_None_Tag = []
        wait_or_none_varDomain_list = []


        for i in tqdm(range(len(var_list)), desc="Converting the string record data of Dataset to computational tensors"):
            var = var_list[i]
            if len(temp_wait_None_Tag) == 0:
                temp_wait_None_Tag.append(i)
            # Parse the variables in the subspace and map the domain of variables to the int16 type
            varDomain_list.append(parse_variable_domain(var[0], var[1], var[2], var[3]))

            wait_or_none_varDomain_list.append(parse_variable_domain(var[0], var[1], var[2], var[3], domainType = 'wait_none'))

    return 



def process_state_variables(pmdp_states_list, vars_dict):
    state_var_values = {}
    state_var_maps = {}
    pattern = r'^\[\s*([^\[\],\s]+(\s*,\s*[^\[\],\s]+)*)?\s*\]$'
    #bool(re.match(pattern, var_value))

    for var in pmdp_states_list:

        if not var.startswith('empty_'):
            #tmp_list = [f'{vars_dict[var].vertex_id}=None', f'{vars_dict[var].vertex_id}=finish', f'{vars_dict[var].vertex_id}=wait']
            tmp_list = [f'{vars_dict[var].vertex_id}=None', f'{vars_dict[var].vertex_id}=finish']
            tmp_list.extend(copy.deepcopy(vars_dict[var].domain))
        else:
            #tmp_list = [f'{var}=None', f'{var}=finish', f'{var}=wait']
            tmp_list = [f'{var}=None', f'{var}=finish']
        
        for i, value in enumerate(tmp_list):
            #if i > 2:
            # if model 'wait' action, then i > 2, otherwise i > 1
            if i > 1:
                if vars_dict[var].controlType == 1:
                    if not bool(re.match(pattern, value)) or ',' not in value:
                        tmp_list[i] = value.strip("[]")
                    else:
                        tmp_list[i] = value
                else:
                    # That is, the variable is an unconstrained variable
                     # The unconstrained variable not only value, also has the probability distribution
                    value = value[0]
                    if not bool(re.match(pattern, value)) or ',' not in value:
                        tmp_list[i] = value.strip("[]")
                    else:
                        tmp_list[i] = value
        
        state_var_values[var] = tmp_list
    
    for var, values in state_var_values.items():
        state_var_maps[var] = {value: idx for idx, value in enumerate(values)}
    
    return state_var_values, state_var_maps

# Don't use any 'wait', model small self-transition probability in PMDP to model the asynchronous behavior in parallel threads in the same parallel block
def process_action_variables(pmdp_actions_list, vars_dict):
    action_var_values = {}
    action_var_maps = {}
    pattern = r'^\[\s*([^\[\],\s]+(\s*,\s*[^\[\],\s]+)*)?\s*\]$'
    
    for var in pmdp_actions_list:
        
        if not var.startswith('empty_'):
            #tmp_list = [f'{vars_dict[var].vertex_id}=None', f'{vars_dict[var].vertex_id}=finish', f'{vars_dict[var].vertex_id}=wait']
            tmp_list = [f'{vars_dict[var].vertex_id}=None', f'{vars_dict[var].vertex_id}=finish']
            tmp_list.extend(copy.deepcopy(vars_dict[var].domain))
        else:
            #tmp_list = [f'{var}=None', f'{var}=finish', f'{var}=wait']
            tmp_list = [f'{var}=None', f'{var}=finish']
        
        for i, value in enumerate(tmp_list):
            #if i > 2:
            # if model 'wait' action, then i > 2, otherwise i > 1
            if i > 1:
                if vars_dict[var].controlType == 1:
                    if not bool(re.match(pattern, value)) or ',' not in value:
                        tmp_list[i] = value.strip("[]")
                    else:
                        tmp_list[i] = value
                else:
                    # That is, the variable is an unconstrained variable
                     # The unconstrained variable not only value, also has the probability distribution
                    value = value[0]
                    if not bool(re.match(pattern, value)) or ',' not in value:
                        tmp_list[i] = value.strip("[]")
                    else:
                        tmp_list[i] = value

        action_var_values[var] = tmp_list
    
    for var, values in action_var_values.items():
        action_var_maps[var] = {value: idx for idx, value in enumerate(values)}
    
    return action_var_values, action_var_maps

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_current_SAN_state(pmdp_env: PMDPEnvironment, pmdp_states_list: List, state_var_maps: Dict, vars_dict, NMDP_mode = False) -> torch.Tensor:

    # get the current state of the PMDP
    state = pmdp_env.current_state
    vars_state_int_list = []
    
    for var_name in pmdp_states_list:
        
        if not NMDP_mode:
            var_value = state[var_name]
        elif NMDP_mode and var_name.startswith('as') and (state[var_name] != None):
            # the 'state[var_name]' is a string, like '[0.963, 0.892, 0.251, 0.967, 0.357, 0.836, 0.356, 0.991, 3, 2, 3]', 
            # the first 8 values are the 'defaultQoS' in Dataset, the last 3 values are the 'NormQoS' model by the 'NMDP paper'
            tmp_var_values = state[var_name].strip('[]').split(',')
           
            '''
            stateLessQoS = [0] * (len(tmp_var_values) - 3)
            NormQoS = [tmp_var_values[-3].strip(), tmp_var_values[-2].strip(), tmp_var_values[-1].strip()]

            var_value = str(stateLessQoS + NormQoS).replace("'", "")
            '''

            var_value = str([0] * len(tmp_var_values)).replace("'", "")

        else:
            var_value = state[var_name]
            #raise ValueError(f"Due to You are using the 'NMDP_mode', and is Constraints stateless about 8 original QoS, \n but the variable name '{var_name}' is not start with 'as', please make all controllable variables start with 'as'.")


        if 'None' == var_value or 'finish' == var_value or 'wait' == var_value:
            vars_state_int_list.append(state_var_maps[var_name][vars_dict[var_name].vertex_id + '=' + var_value])
        if var_value == None or var_value == 'NotSelected':
            vars_state_int_list.append(state_var_maps[var_name][vars_dict[var_name].vertex_id + '=' + 'None'])
        else:
            vars_state_int_list.append(state_var_maps[var_name][var_value])
    state_tensor = torch.tensor(vars_state_int_list, dtype=torch.long, device=learning_device)

    return state_tensor

'''
# Define the soft update function
def soft_update(target_network, online_network, tau):
    for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
'''

def soft_update(target_network, online_network, tau):
    target_state_dict = target_network.state_dict()
    online_state_dict = online_network.state_dict()

    for key in target_state_dict:
        target_state_dict[key].data.copy_(
            tau * online_state_dict[key].data + (1.0 - tau) * target_state_dict[key].data
        )

    target_network.load_state_dict(target_state_dict)



def generate_linear_schedule(beta_start, beta_end, n, beta_min=0.00001, beta_max=1.0):
    """
    Generates a linear schedule with n elements.
    Decays linearly from beta_start to beta_end.

    Returns:
    - beta_schedule (list): A list of n values.
    """
    linear_schedule = []
    step = (beta_start - beta_end) / n  # Calculate the decay value for each step
    beta = beta_start
    for _ in range(n):
        beta = max(beta_end, beta - step)  # Ensure not lower than beta_end
        beta = min(beta_max, max(beta_min, beta))  # Apply min and max value limits
        linear_schedule.append(beta)
    return linear_schedule


# Main training loop
def SAN_main_with_n_steps(bpmn_base_path = '', n_episodes=2000, CET_mode=False, punish_mode=False,
             gamma=0.9, tau = 0.005, PER_alpha = 0.6, batch_size=32, max_samples_buffer=10000, n_warmup_episods=1, initial_learning_rate=0.01,
             whc=0.5, wsc=0.5, eval_interval=100, eval_number=100, n_step=3, PER_beta_start = 0.4, PER_beta_end = 1.0, parallel_asynchronous_prob = 0.05, MDP_example_mode = 'PMDP', instance_evaluation_rewards_list = [],  instance_evaluation_HC_rewards_list = [], 
            instance_evaluation_SC_rewards_list = [], CSSC_mode = False, NMDP_mode = False, n_abstract_number = 10, total_candidates_num = 580 * 10, Evaluate_mode = False, Training_time_record =[], Whatever_Evaluate_mode = False, experiment_timer_mode = False, experiment_timer_episodes = 20000, ex_data = [],
            uncontrol_var_unknow_distributions=None, mode = 'any', random_policy_mode = False, All_seed_Experi_AccumulatedRewards_records = [[],[],[]], last_seed = False):
    
    #print(CSSC_mode)

    if CSSC_mode:
        ### Related work example: CSSC-MDP ###
        MDP_example_mode = 'CSSC_MDP'
        graph = parse_bpmn(bpmn_base_path, n_abstract_number, total_candidates_num = total_candidates_num, CSSC_MDP_example_mode = True, NMDP_example_mode = False,)

        ######## Initialize the graph based on the 'wsdream_dataset1' and BPMN file ########
        #- frequency_prob_dict (dict): A dictionary of frequency probabilities for each variable.
        #- activity_variable_domain_string_list (list): Activity variable C and U domain strings
        frequency_prob_dict, _ = compute_frequency_probabilities_domain(n_abstract_number, k = total_candidates_num, mode_str='Initializing The Probability Dirbution of Dataset: ', decimal=1)
        if len(frequency_prob_dict) != len(graph.vertices) - 2:
            raise ValueError(f"The number of activities in the BPMN file is not equal to the number of activities in the wsdream_dataset1.\n Please make sure the number of activities in the BPMN file is equal to the parameter of 'n_abstract_number'.")
    elif NMDP_mode:
        ### Related work example: NMDP ###
        graph = parse_bpmn(bpmn_base_path, CSSC_MDP_example_mode = False, NMDP_example_mode = True)
        frequency_prob_dict = {}
    else:
        graph = parse_bpmn(bpmn_base_path)
        frequency_prob_dict = {}
        
    pmdp_env = PMDPEnvironment(graph)
    pmdp_env.debug_mode = False
    horizon_space = pmdp_env.get_horizon_space()


    #betas = decay_schedule(1.0, 0.1, 0.999, n_episodes,  log_start = -0.1)
    #betas = sorted(betas, reverse=False)
   

    patience = 50  # 50 consecutive evaluation periods without significant improvement
    patience_counter = 0
    loss_values = []


   
    if 'QWS' in bpmn_base_path: 
        pmdp_env.dynamic_bounds['kpi1'] = (0.0, 1.0)
        pmdp_env.dynamic_bounds['SBI'] = (0.0, 1.0)
        pmdp_env.dynamic_bounds['kpi3'] = (0.0, 8.0)
        pmdp_env.dynamic_bounds['kpi4'] = (0.0, 8.0)
        pmdp_env.dynamic_bounds['kpi5'] = (0.0, 8.0)
        pmdp_env.dynamic_bounds['kpi6'] = (0.0, 8.0)
        pmdp_env.dynamic_bounds['kpi7'] = (0.0, 8.0)
        pmdp_env.dynamic_bounds['kpi8'] = (0.0, 8.0)
        pmdp_env.dynamic_bounds['kpi9'] = (0.0, 8.0)
        pmdp_env.dynamic_bounds['kpi10'] = (0.0, 8.0)
        pmdp_env.dynamic_bounds['kpi14'] = (0.0, 80.0)
        mode = 'QWS'
    elif 'Travel_Agency' in bpmn_base_path:
        pmdp_env.dynamic_bounds['FP'] = (300, 2000)
        pmdp_env.dynamic_bounds['FT'] = (60, 240)
        pmdp_env.dynamic_bounds['TP'] = (100, 1500)
        pmdp_env.dynamic_bounds['RT'] = (120, 600)
        pmdp_env.dynamic_bounds['HP'] = (150, 12000)
        pmdp_env.dynamic_bounds['HC'] = (3, 5)
        mode = 'travel_agency'


    # vars_dict is a dictionary, which stores the objects of variables
    vars_dict = retrieve_variables_from_graph(graph)

    #alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes, log_start = -0.5) 
    #epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes,  log_start = -0.5)
    #epsilons = Normed_MDP_epsilon_decay_schedule(min_epsilon, init_epsilon, epsilon_decay_ratio, n_episodes)

    
    #Variance_weight = 0.5


    filtered_QHorizonTag_StateSpace_tensor = process_horizon(pmdp_env, horizon_space, horizon_int16_to_string_map, existing_hashes, learning_device)
    # The 'get_QHorizon_actionSpace_columns' will create the golobal data that is needed by 'all_int16_to_string_map' and 'all_string_to_int16', don't delete it even it returns is not used in this function
    filtered_QHorizon_ActionSpace_tensor, result_HorizonVerticesOrder_rowsColumnsTags_inTensor_list = get_QHorizon_actionSpace_columns(pmdp_env, horizon_space, filtered_QHorizonTag_StateSpace_tensor)

    Initialize_statesActions_PMDPMappingDict(pmdp_env, graph)

    all_int16_to_string_map = {}
    all_int16_to_string_map.update(int16_to_string_map)
    all_int16_to_string_map.update(wait_int16_to_string_map)
    # The 'None_int16_to_string_map' will has additional items, which are the 'None' action for each vertex in the horizon space
    all_int16_to_string_map.update(None_int16_to_string_map)
    all_int16_to_string_map.update(finish_int16_to_string_map)
    all_int16_to_string_map.update(horizon_int16_to_string_map)
    # swap keys and values
    all_string_to_int16 = {value: key for key, value in all_int16_to_string_map.items()}

    # Assume the dimensions of the state space and action space
    pmdp_states_dict = pmdp_env.current_state
    pmdp_states_list = []
    pmdp_actions_list = []
    for var_name in pmdp_states_dict.keys():  
        if not var_name.endswith('_direction') and not var_name.endswith('_Horizon_Action'):
            pmdp_states_list.append(var_name)

    for key, vertex in graph.vertices.items():
        if not key.startswith('empty_'):
            vertex_contrallable_vars = vertex.C_v 
            for var in vertex_contrallable_vars:
                if not var.name.endswith('_Horizon_Action'):
                    pmdp_actions_list.append(var.name)
        else:
            pmdp_actions_list.append(key)

    # Makesure the order of the state and action variables are consistent in each execution and training process
    pmdp_states_list.sort()
    pmdp_actions_list.sort()

    # handle the state variables
    state_var_values, state_var_maps = process_state_variables(pmdp_states_list, vars_dict)

    # handle the action variables
    # Don't use any 'wait' action, model small self-transition probability in PMDP to model the asynchronous behavior in parallel threads in the same parallel block
    action_var_values, action_var_maps = process_action_variables(pmdp_actions_list, vars_dict)

    # construct the reverse mapping for action variables, which is used to convert the action index back to the original action string
    action_var_reverse_maps = {}
    for var, val_map in action_var_maps.items():
        reverse_map = {val: key for key, val in val_map.items()}
        action_var_reverse_maps[var] = reverse_map




    # get the state and action space size and construct their indices mapping, the original state and action space are the string type
    # define the size of state and action space
    state_sizes = [len(state_var_maps[var]) for var in pmdp_states_list]
    action_var_sizes = [len(action_var_maps[var]) for var in pmdp_actions_list]


    # initialize Q network and target network, using NoisyLinear
    q_network = StateActionNetwork(state_sizes, action_var_sizes).to(learning_device)
    target_network = StateActionNetwork(state_sizes, action_var_sizes).to(learning_device)


    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()




    # define the optimizer
    #optimizer = optim.Adam(q_network.parameters(), lr=initial_learning_rate)

    optimizer_sparse, optimizer_dense = get_optimizers(q_network, learning_rate=initial_learning_rate)

    
    # Episode is same word as trajectory
    accumulate_HC_SC_rewards_in_each_trajectory = []
    accumulate_HC_rewards_in_each_trajectory = []
    accumulate_SC_rewards_in_each_trajectory = []

    episode_steps = []

    start_vertex = next((v for v in horizon_space if isinstance(v, str) and v.startswith('StartEvent')), None)
 

    beta_esti_number = n_episodes * 20
    PER_Buffer = PrioritizedReplayBuffer_Pytorch(max_samples = max_samples_buffer, batch_size = batch_size, 
                                                 device = learning_device, beta_start=PER_beta_start, beta_end = PER_beta_end, beta_number = beta_esti_number, alpha = PER_alpha)
    #PER_Buffer.beta = 0.4
   
    evaluation_rewards = []
    evaluation_HC_rewards = []
    evaluation_SC_rewards = []
    Warmup_training_tag = 0

    #Only use in the 'CSSC-MDP' mode for experiment about 'wsdream_dataset1'
    #cssc_guide_decay_paras = generate_linear_schedule(1, 0.0, int(n_episodes * 0.2 / eval_interval))
    

    if experiment_timer_mode:
        # record start time
        ex_start_time = time.time()


    start_time = time.time()

    AccumulatedRewards_records = [[],[],[]]

    for episode in tqdm(range(n_episodes + n_warmup_episods), leave=False):

        step = 0
        if random_policy_mode == False:
            if episode % 100 == 0 and episode > 0 or episode == n_episodes + n_warmup_episods - 1:
                elapsed_time = time.time() - start_time
                Training_time_record.append(elapsed_time)
                print(f"Episode {episode}: Elapsed Time = {elapsed_time:.2f} seconds")
                #print(f"Episode {episode} started, PER beta: {PER_Buffer.beta}")

            # initialize environment and state
            pmdp_env.current_state = pmdp_env.initialize_state()
            pmdp_env.context = pmdp_env.build_pmdp_context()
            pmdp_env.reset_KPIs_Constraints_to_ini_stages()
            accumulate_rewards_list = [[0, 0, 0]]

            activeVertices_of_State = start_vertex
            active_vertices_of_State_verticesList = [start_vertex]

            HC_list = [pmdp_env.graph.HC]
            SC_list = [pmdp_env.graph.SC]

            # For compatibility with the 'earlier' version of the code, the 'ConstraintState_list' is used to store the constraint state of the PMDP in Cdoube_Qlearning_with_CET
            ConstraintState_list = [[]]

            state = get_current_SAN_state(pmdp_env, pmdp_states_list, state_var_maps, vars_dict)
            done = False
            step = 0



            # create n-step buffer
            n_step_buffer = []

            if episode < n_warmup_episods - 1:
                warmup_tag = False
            elif episode == n_warmup_episods-1:
                warmup_tag = True

                # calculate max value
                max_episode_steps = max(episode_steps)
                PER_Buffer.betas = PER_Buffer.generate_beta_schedule(PER_beta_start, PER_beta_end, max_episode_steps * (n_episodes + n_warmup_episods))


            while not done:
                if step == 43:
                    pass

                valid_actions = get_valid_actions(pmdp_env, active_vertices_of_State_verticesList, action_var_maps, vars_dict, NMDP_mode)
                q_network.reset_noise()
                action = SAN_select_action_noisy(state, valid_actions, q_network)

                # get action tensor representation in PMDP based on embedding index
                pmdp_action_tensor = get_action_pmdp_action_tensor(action, action_var_reverse_maps, all_string_to_int16)

                if CSSC_mode:
                    next_activeVertices_of_State = update_state_and_parse_next(
                    pmdp_env, 
                    pmdp_action_tensor,
                    all_int16_to_string_map, 
                    activeVertices_of_State,
                    frequency_prob_dict = frequency_prob_dict,
                    MDP_example_mode = MDP_example_mode
                    )

                else:
                    next_activeVertices_of_State = update_state_and_parse_next(
                        pmdp_env,
                        pmdp_action_tensor,
                        all_int16_to_string_map,
                        activeVertices_of_State,
                        parallel_asynchronous_prob,
                        uncontrol_var_unknow_distributions = uncontrol_var_unknow_distributions, 
                        mode = mode
                    )

                activeVertices_of_State = next_activeVertices_of_State

                # immediate_rewards_list, the column is the index of subStateSpace,, here only one subStateSpace, so use Rewards_HC_plus_SC_list[0][0] to index the immediate reward
                # immediate_rewards_list is consist of '[Rewards_HC_plus_SC_list, Rewards_HC_list, Rewards_SC_list, All_Rewards_All_subStates]'
                # accumulate_rewards_list, the index of the subStateSpace of the behavior, from 0 to 2, accumulate_HC_plus_SC_rewards, accumulate_HC_rewards, accumulate_SC_rewards
                immediate_rewards_list, accumulate_rewards_list, ConstraintState_list, evaluated_constraintsName_list = pmdp_env.calculate_reward(
                    whc, wsc, HC_list, SC_list, ConstraintState_list, punish_mode, CET_mode, accumulate_rewards_list, subStateSpace_number=1)

                if activeVertices_of_State is None:
                    done = True
                else:
                    active_vertices_of_State_verticesList = pmdp_env.convert_horizon_item_to_list(next_activeVertices_of_State)

                next_state = get_current_SAN_state(pmdp_env, pmdp_states_list, state_var_maps, vars_dict, NMDP_mode)
                reward = immediate_rewards_list[0][0]

                next_valid_actions = get_valid_actions(pmdp_env, active_vertices_of_State_verticesList, action_var_maps, vars_dict, NMDP_mode)

                # store experience in n-step buffer
                n_step_buffer.append((state, action, reward, next_state, done, next_valid_actions))

                # when n-step buffer is full, calculate n-step target and store in replay buffer, that is the n-step buffer is different from the replay buffer
                if len(n_step_buffer) >= n_step:
                    # calculate n-step target
                    cum_reward = 0
                    next_gamma_count = 0
                    for idx in range(n_step):
                        cum_reward += (gamma ** idx) * n_step_buffer[idx][2]  # accumulate reward
                        next_gamma_count += 1
                    next_gamma = gamma ** next_gamma_count
                    next_state_n = n_step_buffer[-1][3]  # the next state at step n
                    next_valid_actions_n = n_step_buffer[-1][5]  # the valid actions at step n
                    done_n = n_step_buffer[-1][4]        # the done flag at step n

                    # store n-step experience in replay buffer
                    PER_Buffer.store([n_step_buffer[0][0], n_step_buffer[0][1], cum_reward, next_state_n, done_n, next_valid_actions_n, next_gamma])

                    # remove the oldest experience from the buffer
                    n_step_buffer.pop(0)

                # when episode ends, process remaining n-step buffer
                if done:
                    while len(n_step_buffer) > 0:
                        # calculate n-step target for remaining steps
                        cum_reward = 0
                        leaving_steps = len(n_step_buffer)
                        next_gamma_count = 0
                        for idx in range(leaving_steps):
                            cum_reward += (gamma ** idx) * n_step_buffer[idx][2]  # accumulate reward
                            next_gamma_count += 1
                        next_gamma = gamma ** next_gamma_count

                        next_state_n = n_step_buffer[-1][3]
                        next_valid_actions_n = n_step_buffer[-1][5]
                        done_n = n_step_buffer[-1][4]

                        PER_Buffer.store([n_step_buffer[0][0], n_step_buffer[0][1], cum_reward, next_state_n, done_n, next_valid_actions_n, next_gamma])
                        n_step_buffer.pop(0)


                    AccumulatedRewards_records[0].append(accumulate_rewards_list[-1][0])
                    AccumulatedRewards_records[1].append(accumulate_rewards_list[-1][1])
                    AccumulatedRewards_records[2].append(accumulate_rewards_list[-1][2])

                    if episode % 50 == 0 and mode == 'QWS':
                        #print(f"Episode {episode}, SBI:{pmdp_env.context['SBI']}\n")
                        #print(f"Episode {episode}, LTS:{pmdp_env.context['LTS']}\n")
                        pass
                        

                    if episode % 2000 == 0:
                        pass
                        #print(f"Episode {episode}, Accumulated Reward Records: {AccumulatedRewards_records}\n")

                        ## generate a random identifier for the filename
                        #random_identifier = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

                       
                        #filename = f"episode_{episode}_{random_identifier}.txt"

                        # store AccumulatedRewards_records as a text file
                        #with open(filename, 'w') as file:
                            # convert the list to string format and write to file
                            #file.write(str(AccumulatedRewards_records))

                        
                        #print(f"Episode {episode}, Accumulated Reward Records: {AccumulatedRewards_records}")
                        #print(f"Accumulated Reward Records saved to {filename}")


                state = next_state
                step += 1


                
                

                if warmup_tag == True and random_policy_mode == False:
                    # When the number of samples in the buffer is greater than the batch size, start training as normal
                    train_q_network_noisy(q_network, target_network, PER_Buffer, warmup_tag, optimizer_sparse, optimizer_dense,
                                batch_size, loss_values)
                    
                    
                    # soft update the target network
                    soft_update(target_network, q_network, tau)
        elif episode == n_warmup_episods-1:
            warmup_tag = True

            
        # Record the number of steps in each episode
        episode_steps.append(step)


        if ((episode > n_warmup_episods - 1 or Whatever_Evaluate_mode == True) and episode % eval_interval == 0 and (warmup_tag == True or Whatever_Evaluate_mode == True) or (episode == n_episodes + n_warmup_episods - 1)) and Evaluate_mode == True and episode > 0:
            # In the 'P-MDP' mode, result1=  avg_reward, result2 = avg_HC_reward
            # In the 'CSSC-MDP' mode, result1= service composition success rate, result2 = avg_reward
            result1, result2, accumulatedRewards_records = evaluate_policy(pmdp_env, q_network, eval_number, pmdp_states_list, action_var_maps, state_var_maps, vars_dict, start_vertex, whc, wsc, False, parallel_asynchronous_prob, all_string_to_int16, all_int16_to_string_map, CSSC_mode, MDP_example_mode, frequency_prob_dict, action_var_reverse_maps, NMDP_mode, uncontrol_var_unknow_distributions, mode, random_policy = random_policy_mode)
        
            if episode % eval_interval == 0  or  (episode == n_episodes + n_warmup_episods - 1):
                if not CSSC_mode:
                    print(f"\n################ Episode {episode}, Average Evaluation Reward: {result1}, Average HC Reward: {result2}, Average SC Reward: {result1 - result2} ################, ")
                    #print(f"################ Episode {episode}, Buffer beta: {PER_Buffer.beta} ################")
                elif CSSC_mode:
                    #print(f"\n################ Episode {episode}, Average Service Composition Success Rate: {result1}, Average total Reward: {result2} ################")
                    print(f"\n################ Episode {episode}, Average Service Composition Success Rate: {result1} ################")
                    #print(f"################ Episode {episode}, Buffer beta: {PER_Buffer.beta} ################")

            evaluation_rewards.append(result1)
            evaluation_HC_rewards.append(result2)
            if not CSSC_mode:
                evaluation_SC_rewards.append(result1 - result2)

            if random_policy_mode:
                pass

        if experiment_timer_mode and episode == experiment_timer_episodes - 1:
            # recpord end time and calculate average time
            ex_end_time = time.time()
            avg_time = (ex_end_time - ex_start_time) / experiment_timer_episodes

            ex_data.append(avg_time)

            # 打印耗时
            print(f"\n #####################\n The total training time of {experiment_timer_episodes}: {ex_end_time - ex_start_time:.3f} seconds \n #####################\n")
            print(f"\n #####################\n The average training time of {experiment_timer_episodes}: {avg_time:.3f} seconds \n #####################\n")
            print(f"\n #####################\n abstract_number: {n_abstract_number}, candidates_num: {total_candidates_num/n_abstract_number} \n #####################\n")

    instance_evaluation_rewards_list.append(evaluation_rewards)
    instance_evaluation_HC_rewards_list.append(evaluation_HC_rewards)
    instance_evaluation_SC_rewards_list.append(evaluation_SC_rewards)
    
    if not last_seed:
        All_seed_Experi_AccumulatedRewards_records[0].append(AccumulatedRewards_records[0])
        All_seed_Experi_AccumulatedRewards_records[1].append(AccumulatedRewards_records[1])
        All_seed_Experi_AccumulatedRewards_records[2].append(AccumulatedRewards_records[2])
        
    else:
        
        All_seed_Experi_AccumulatedRewards_records[0].append(AccumulatedRewards_records[0])
        All_seed_Experi_AccumulatedRewards_records[1].append(AccumulatedRewards_records[1])
        All_seed_Experi_AccumulatedRewards_records[2].append(AccumulatedRewards_records[2])
        
    
        # Create "training_records" directory if it doesn't exist
        experiment_dir = "training_records"
        os.makedirs(experiment_dir, exist_ok=True)
        
        
        # Construct full file path in the experiment directory
        filename = f"{process_id}_accumulatedRewards_records_for_{n_episodes + n_warmup_episods}_episodes_of_training.txt"
        full_path = os.path.join(experiment_dir, filename)

        
        prompts = ['total_rewards:\n', 'hc_rewards:\n', 'sc_rewards:\n']

        
        with open(full_path, 'w') as file:
            for i, sub_list in enumerate(All_seed_Experi_AccumulatedRewards_records):
                # Get the prompt information corresponding to the current line
                prompt = prompts[i]
                # Convert the sub-list elements to strings and join with commas
                line = ', '.join(str(item) for item in sub_list)
                # Add [ and ] around each line, along with the prompt and a newline
                line_with_brackets = f"{prompt} [{line}]\n"
                # Write the processed lines to a file
                file.write(line_with_brackets)
        #print(f"\n {AccumulatedRewards_records[2][-1]}")

        print(f"\nAccumulated Rewards Records ({n_episodes + n_warmup_episods} episodes of training) been saved to  \"training_records\\{filename}\"")
        
        ### Below is used to print the transition count, especially for concurrent flows (self-transition)
        #print(episode_steps)


    return All_seed_Experi_AccumulatedRewards_records

    
def save_experiment_rollout_data(
    pmdp_for_cssc,
    insEval_2Dlist,
    Training_time_record,
    Evaluate_mode,
    process_id
    
):
    """
    Save experiment data to files, including evaluation reward data and training time records (stored in the same directory).
    
    Parameters:
        insEval_2Dlist: 2D list containing three types of evaluation reward data
        Training_time_record: List of training time records
        Evaluate_mode: Flag for evaluation mode; training time is saved only when this is False
        process_id: Process ID used to generate unique filenames
        experiment_instance_number: Number of experiment instances
        initial_learning_rate: Initial learning rate
        n_episodes: Total number of training episodes
    """
    # Ensure the training_records data folder exists
    experiment_data_folder = "training_records"
    os.makedirs(experiment_data_folder, exist_ok=True)

    # Save training time records (only in non-evaluation mode)
    if not Evaluate_mode:
        # Specify path for training time records (saved in Experiment_Data directory)
        time_file_path = os.path.join(experiment_data_folder, 'Training_time_record.txt')
        
        # Write training time records
        with open(time_file_path, 'w') as file:
            json.dump(Training_time_record, file)
        
        print(f"Training time records have been written to {time_file_path}")
    
    # Construct path for experiment data file
    data_file_path = os.path.join(
        experiment_data_folder, 
        f"{process_id}_rollout_rewards.txt"
    )
    
    if not pmdp_for_cssc:
        # Define titles for each reward data (corresponds to the order in the list)
        rewards_titles = [
            "Total_rollout_rewards:\n",
            "HC_rollout_rewards:\n",
            "SC_rollout_rewards:\n"
        ]
    else:
        # Define titles for each reward data (corresponds to the order in the list)
        rewards_titles = [
            "Average Success Rate:\n"
        ]
        
        
    #display_items = [item.replace('\n', '').replace(':', '') for item in rewards_titles]
    #print(f"\nThe following data is composed of these items: {', '.join(display_items)}")
    #print(insEval_2Dlist)

    # Save evaluation reward data with corresponding titles
    with open(data_file_path, 'w') as file:
        for title, rewards in zip(rewards_titles, insEval_2Dlist):
            file.write(title)  # Write title with line break
            line = ','.join(map(str, rewards))
            file.write(line + '\n\n')  # Add empty line after data for separation
    
    print(f"\nRollout Experiment data has been saved to \"{data_file_path}\"")



def run_pmdp_experiments(
    bpmn_base_path,
    seeds,
    experiment_instance_number,
    n_episodes,
    n_warmup_episods,
    n_step,
    batch_size,
    initial_learning_rate,
    punish_mode,
    gamma,
    tau,
    PER_alpha,
    PER_beta_start,
    PER_beta_end,
    max_PER_buffer,
    whc,
    wsc,
    eval_interval,
    eval_number,
    parallel_asynchronous_prob,
    insEval_total_rewards_list,
    insEval_HC_rewards_list,
    insEval_SC_rewards_list,
    CSSC_mode,
    NMDP_mode,
    n_abstract_number,
    total_candidates_num,
    Evaluate_mode,
    Training_time_record,
    Whatever_Evaluate_mode,
    experiment_timer_mode,
    experiment_timer_episodes,
    uncontrol_var_unknow_distributions,
    random_policy_mode,
    Do_time_experiment,
    abstract_number_list=None,
    candidate_number_list=None,
    max_PER_buffer_list=None
):
    """
    Run PMDP-related experiments, including multi-instance benchmark experiments and training time comparison 
    experiments under different parameter configurations.
    
    Parameters:
        bpmn_base_path: Base path for BPMN models
        seeds: List of random seeds
        experiment_instance_number: Number of experiment instances
        n_episodes: Total number of training episodes
        n_warmup_episods: Number of warm-up episodes
        n_step: Number of steps for n-step TD learning
        batch_size: Batch size for training
        initial_learning_rate: Initial learning rate
        punish_mode: Whether to enable punishment mechanism
        gamma: Discount factor for future rewards
        tau: Soft update coefficient for target networks
        PER_alpha: Alpha parameter for Prioritized Experience Replay (priority weighting)
        PER_beta_start: Initial beta parameter for PER (importance sampling weight)
        PER_beta_end: Final beta parameter for PER (linearly annealed)
        max_PER_buffer: Maximum capacity of the PER buffer
        whc: Weight for hard constraints
        wsc: Weight for soft constraints
        eval_interval: Interval (in episodes) for performance evaluation
        eval_number: Number of repetitions for each evaluation
        parallel_asynchronous_prob: Probability for parallel asynchronous execution
        insEval_total_rewards_list: List for recording total rewards (maintained externally)
        insEval_HC_rewards_list: List for recording HC rewards (maintained externally)
        insEval_SC_rewards_list: List for recording SC rewards (maintained externally)
        CSSC_mode: Whether to enable CSSC mode
        NMDP_mode: Whether to enable NMDP mode
        n_abstract_number: Number of abstract services
        total_candidates_num: Total number of candidate services
        Evaluate_mode: Whether to run in evaluation mode
        Training_time_record: List for recording training times (maintained externally)
        Whatever_Evaluate_mode: Whether to force evaluation mode (ignores warm-up phase)
        experiment_timer_mode: Whether to enable experiment timing mode
        experiment_timer_episodes: Number of episodes for timing in timer mode
        uncontrol_var_unknow_distributions: Unknown distributions of uncontrolled variables
        random_policy_mode: Whether to use random policy mode
        Do_time_experiment: Whether to run training time comparison experiments
        abstract_number_list: List of abstract service counts (for time experiments)
        candidate_number_list: List of candidate service counts (for time experiments)
        max_PER_buffer_list: List of PER buffer capacities (for time experiments)
    """
    # Initialize default parameters (if time experiment lists are not provided)
    if Do_time_experiment:
        abstract_number_list = abstract_number_list or [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        candidate_number_list = candidate_number_list or [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        max_PER_buffer_list = max_PER_buffer_list or [10000] * len(abstract_number_list)
        
    All_seed_Experi_AccumulatedRewards_records = [[],[],[]]

    if not Do_time_experiment:
        if Evaluate_mode:
            print(f"\n \n ############################################################################################################################## \n\n Notice: Evaluate_mode is True (i.e., Rollout Experiment is activated), " f"training time in evaluation mode may be longer than expected because the Rollout Experiment. \n \n ##############################################################################################################################\n\n")
        # Part 1: Run multi-instance PMDP experiments
        for _ in range(experiment_instance_number):
            # Print experiment configuration
            print(f"\nPMDP Experiment Instance {_ + 1} with setting: {n_step}-Step TD, Batch Size: {batch_size}, "
                f"Learning Rate: {initial_learning_rate}, Punish Mode: {punish_mode}, Gamma: {gamma}, Tau: {tau}, "
                f"PER_alpha: {PER_alpha}, PER_beta_start: {PER_beta_start}, PER_beta_end: {PER_beta_end}, "
                f"n_warmup_episods: {n_warmup_episods}, Total Experiments: {experiment_instance_number}")

            # Set random seed
            seed = seeds[0 + _]
            set_seed(seed)  # Assume set_seed is defined externally
            print(f"Seed: {seed}")
            if _ == experiment_instance_number-1:
                last_seed = True
            else:
                last_seed = False

            # Run main experiment function
            All_seed_Experi_AccumulatedRewards_records = SAN_main_with_n_steps(  # Assume SAN_main_with_n_steps is defined externally
                bpmn_base_path=bpmn_base_path,
                n_episodes=n_episodes - n_warmup_episods,
                CET_mode=False,
                punish_mode=punish_mode,
                gamma=gamma,
                tau=tau,
                PER_alpha=PER_alpha,
                batch_size=batch_size,
                max_samples_buffer=max_PER_buffer,
                PER_beta_start=PER_beta_start,
                PER_beta_end=PER_beta_end,
                n_warmup_episods=n_warmup_episods,
                initial_learning_rate=initial_learning_rate,
                whc=whc,
                wsc=wsc,
                eval_interval=eval_interval,
                eval_number=eval_number,
                n_step=n_step,
                parallel_asynchronous_prob=parallel_asynchronous_prob,
                instance_evaluation_rewards_list=insEval_total_rewards_list,
                instance_evaluation_HC_rewards_list=insEval_HC_rewards_list,
                instance_evaluation_SC_rewards_list=insEval_SC_rewards_list,
                CSSC_mode=CSSC_mode,
                NMDP_mode=NMDP_mode,
                n_abstract_number=n_abstract_number,
                total_candidates_num=total_candidates_num,
                Evaluate_mode=Evaluate_mode,
                Training_time_record=Training_time_record,
                Whatever_Evaluate_mode=Whatever_Evaluate_mode,
                experiment_timer_mode=experiment_timer_mode,
                experiment_timer_episodes=experiment_timer_episodes,
                uncontrol_var_unknow_distributions=uncontrol_var_unknow_distributions,
                random_policy_mode=random_policy_mode,
                All_seed_Experi_AccumulatedRewards_records = All_seed_Experi_AccumulatedRewards_records,
                last_seed = last_seed
            )
            reset_mappings()  # Assume reset_mappings is defined externally


    ########## Part 2: Training time experiments with different candidate and abstract service counts ##########
    # Store training time experiment data
    ex_data = []
    if Do_time_experiment:
        # Check for conflict between timer mode and evaluation mode
        if experiment_timer_mode and Evaluate_mode:
            print(f"\n #####################\n Warning: experiment_timer_mode is True but Evaluate_mode is also True, "
                  f"training time may be longer than expected in evaluation mode. \n #####################\n")

        # Iterate over all abstract service count configurations
        for _ in range(len(abstract_number_list)):
            # Enable CSSC mode (for WSDREAM dataset)
            
            current_n_abstract = abstract_number_list[_]
            current_max_PER = max_PER_buffer_list[_]

            # Iterate over all candidate service count configurations
            for i in range(len(candidate_number_list)):
                candidates_per_abstract = candidate_number_list[i]
                current_total_candidates = current_n_abstract * candidates_per_abstract
                
                # Construct BPMN model path
                bpmn_path = f'BPMN_Models/Benchmark_experiment_models/Sequence_CSSC_{current_n_abstract}Activity_for_PMDP.bpmn'

                print(f"\n ############## Experiment for {current_n_abstract} Abstract Services and {candidates_per_abstract} Candidate Services ############## \n")
                print(f"PMDP Experiment Instance {_ + 1} with setting: {n_step}-Step TD, Batch Size: {batch_size}, "
                      f"Learning Rate: {initial_learning_rate}, Punish Mode: {punish_mode}, Gamma: {gamma}, Tau: {tau}, "
                      f"PER_alpha: {PER_alpha}, PER_beta_start: {PER_beta_start}, PER_beta_end: {PER_beta_end}, "
                      f"n_warmup_episods: {n_warmup_episods}, Total Experiments: {experiment_instance_number}")

                # Set random seed
                seed = seeds[0]
                set_seed(seed)
                print(f"Seed: {seed}")
                
                
                if _ == experiment_instance_number-1:
                    last_seed = True
                else:
                    last_seed = False

                # Run main experiment function (collect training time data)
                All_seed_Experi_AccumulatedRewards_records = SAN_main_with_n_steps(
                    bpmn_base_path=bpmn_path,
                    n_episodes=n_episodes - n_warmup_episods,
                    CET_mode=False,
                    punish_mode=punish_mode,
                    gamma=gamma,
                    tau=tau,
                    PER_alpha=PER_alpha,
                    batch_size=batch_size,
                    max_samples_buffer=current_max_PER,
                    PER_beta_start=PER_beta_start,
                    PER_beta_end=PER_beta_end,
                    n_warmup_episods=n_warmup_episods,
                    initial_learning_rate=initial_learning_rate,
                    whc=whc,
                    wsc=wsc,
                    eval_interval=eval_interval,
                    eval_number=eval_number,
                    n_step=n_step,
                    parallel_asynchronous_prob=parallel_asynchronous_prob,
                    instance_evaluation_rewards_list=insEval_total_rewards_list,
                    instance_evaluation_HC_rewards_list=insEval_HC_rewards_list,
                    instance_evaluation_SC_rewards_list=insEval_SC_rewards_list,
                    CSSC_mode=CSSC_mode,
                    NMDP_mode=NMDP_mode,
                    n_abstract_number=current_n_abstract,
                    total_candidates_num=current_total_candidates,
                    Evaluate_mode=Evaluate_mode,
                    Training_time_record=Training_time_record,
                    Whatever_Evaluate_mode=Whatever_Evaluate_mode,
                    experiment_timer_mode=experiment_timer_mode,
                    experiment_timer_episodes=experiment_timer_episodes,
                    ex_data=ex_data,
                    All_seed_Experi_AccumulatedRewards_records = All_seed_Experi_AccumulatedRewards_records,
                    last_seed = last_seed
                )
                reset_mappings()

                # Print average training time for current configuration
                print(f"\n ############## Avg training time for {current_n_abstract} Abstract Services and {candidates_per_abstract} Candidate Services: {ex_data[-1]} ############## \n")
                print(f"\n Current experiment data: {ex_data} \n")

        # Print complete training time record
        print(f"Complete training time record: {ex_data}")
    
    return ex_data  # Return training time experiment data (if executed)


    
def load_config(config_filename):
    """
    Load JSON configuration file from the 'training_configs' folder in the same directory
    
    Parameters:
        config_filename: Name of the configuration file in the training_configs folder (e.g., "wsdream_config.json")
    Returns:
        Configuration dictionary
    """
    # Construct the full path to the configuration file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(current_dir, "training_configs")
    config_path = os.path.join(config_dir, config_filename)
    
    # Check if the configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the configuration file
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def training_pmdp(config_filename="default_config.json", process_id=None):
    try:
        # Load configuration from the training_configs folder
        config = load_config(config_filename)
        
        # Initialize externally maintained result recording lists
        insEval_total_rewards_list = []
        insEval_HC_rewards_list = []
        insEval_SC_rewards_list = []
        Training_time_record = []

        # Process the parameter before constructing experiment_params
        # 1. Read the original value from the configuration (JSON null will be parsed as Python None)
        raw_distributions = config["bpmn_config"]["uncontrol_var_unknow_distributions"]

        # 2. Conditional judgment: If it is None (that is, null in JSON), call the function calculation; otherwise use the configuration value
        if 'Travel_Agency' in config["bpmn_config"]["bpmn_base_path"]:
            # Call runtime calculation function (assumed function name is calculate_distributions)
            calculated_distributions = get_travelAgency_distribution()  # Your custom calculation function
            uncontrolled_distributions = calculated_distributions
            print("TravelAgency Case")
        elif 'QWS' in config["bpmn_config"]["bpmn_base_path"]:
            calculated_distributions = get_QWS_distribution('Datasets/QWS/normalized_chunks_10_AbstractServices_Eech_has_250_ConcreteServices.txt', 10)
            uncontrolled_distributions = calculated_distributions
            print("QWS Case")
        else:
            # Use the specific value provided in the configuration
            uncontrolled_distributions = raw_distributions
            
        if 'Sequence_CSSC' in config["bpmn_config"]["bpmn_base_path"]:
            pmdp_for_cssc = True 
            print("Benchmark Case")
        else:
            pmdp_for_cssc = False
                
        

        # Map configuration parameters to experiment function parameters
        experiment_params = {
            # BPMN configuration
            "bpmn_base_path": config["bpmn_config"]["bpmn_base_path"],
            "uncontrol_var_unknow_distributions": uncontrolled_distributions,
            "parallel_asynchronous_prob": config["bpmn_config"]["parallel_async_probability"],
            
            # WSDREAM dataset specific parameters
            #candidate_number is fixed to 116 as default in the WSDREAM dataset
            "n_abstract_number": config["wsdream_experiment_config"]["abstract_service_count"],
            "total_candidates_num": config["wsdream_experiment_config"]["total_candidates"],

            # Random seeds
            "seeds": config["random_seeds"]["values"],
            
            # Training episode configuration
            "experiment_instance_number": config["training_episodes"]["experiment_instances"],
            "n_episodes": config["training_episodes"]["total_episodes"],
            "n_warmup_episods": config["training_episodes"]["warmup_episodes"],
            
            # Evaluation settings
            "eval_interval": config["evaluation_settings"]["eval_interval"],
            "eval_number": config["evaluation_settings"]["eval_repetitions"],
            
            
            # DQN hyperparameters
            "n_step": config["dqn_hyperparameters"]["n_step"],
            "batch_size": config["dqn_hyperparameters"]["batch_size"],
            "initial_learning_rate": config["dqn_hyperparameters"]["initial_learning_rate"],
            "gamma": config["dqn_hyperparameters"]["gamma"],
            "tau": config["dqn_hyperparameters"]["tau"],
            "PER_alpha": config["dqn_hyperparameters"]["per_alpha"],
            "PER_beta_start": config["dqn_hyperparameters"]["per_beta_start"],
            "PER_beta_end": config["dqn_hyperparameters"]["per_beta_end"],
            "max_PER_buffer": config["dqn_hyperparameters"]["per_buffer_capacity"],
            
            # Constraint weights
            "whc": config["constraint_weights"]["whc"],
            "wsc": config["constraint_weights"]["wsc"],
            

            # Mode flags
            "punish_mode": config["mode_flags"]["punish_mode"],
            "NMDP_mode": config["mode_flags"]["nmdp_mode"],
            "Evaluate_mode": config["mode_flags"]["evaluate_mode"],
            "Whatever_Evaluate_mode": config["mode_flags"]["force_evaluate_mode"],
            "experiment_timer_mode": config["mode_flags"]["experiment_timer_mode"],
            "random_policy_mode": config["mode_flags"]["random_policy_mode"],
            "Do_time_experiment": config["mode_flags"]["do_time_experiment"],
            "CSSC_mode": config["mode_flags"]["cssc_mode"],
            
            # Timer settings
            "experiment_timer_episodes": config["timer_settings"]["timer_episodes"],
            
            # Externally maintained result lists
            "insEval_total_rewards_list": insEval_total_rewards_list,
            "insEval_HC_rewards_list": insEval_HC_rewards_list,
            "insEval_SC_rewards_list": insEval_SC_rewards_list,
            "Training_time_record": Training_time_record,
            
            # Time experiment related lists (using default values)
            "abstract_number_list": None,
            "candidate_number_list": None,
            "max_PER_buffer_list": None
        }
        
        # Execute the experiment
        print(f"Starting experiment from configuration file: {os.path.join('training_configs', config_filename)}")
       
        ex_data = run_pmdp_experiments(**experiment_params)
        
        
        if ex_data != []:
            print(f"Training time data: {ex_data}")
        #print(f"Number of recorded total rewards: {len(insEval_total_rewards_list)}")
        
        if not pmdp_for_cssc:
            ###Rollout data preparation for saving
            insEval_2Dlist = []
            insEval_2Dlist.append([insEval_total_rewards_list])
            insEval_2Dlist.append([insEval_HC_rewards_list])
            insEval_2Dlist.append([insEval_SC_rewards_list])
        else:
            # for CSSC compare, this Rollout of Average Success Rate and Average Total Reward are stored in 'insEval_total_rewards_list' and 'insEval_HC_rewards_list'
            insEval_2Dlist = []
            ##Average Success Rate for this mode
            insEval_2Dlist.append([insEval_total_rewards_list])
            ##Average Total Reward for this mode
            #insEval_2Dlist.append([insEval_HC_rewards_list])

        # Call the encapsulated function to save the data
        save_experiment_rollout_data(
            pmdp_for_cssc,
            insEval_2Dlist=insEval_2Dlist,
            Training_time_record=Training_time_record,
            Evaluate_mode=config["mode_flags"]["evaluate_mode"],
            process_id=process_id
        )
        
        # Experiment completion information
        print(f"\nExperiment {process_id} executed successfully for {len(insEval_total_rewards_list)} random seeds\n\n")


    except Exception as e:
        print(f"Error executing experiment: {str(e)}", file=sys.stderr)
        sys.exit(1)

def training_cssc_mdp(config_filename="cssc_mdp_config.json", process_id=None):
    try:
        # Load configuration
        config = load_config(config_filename)
        
        # Initialize result lists
        insEval_total_rewards_list = []
        insEval_HC_rewards_list = []
        insEval_SC_rewards_list = []
        Training_time_record = []


        # Extract parameters from config
        params = {
            # BPMN configuration
            "bpmn_file_path": config["bpmn_config"]["bpmn_base_path"],

            # wsdream_experiment_config
            "n_abstract_number": config["wsdream_experiment_config"]["abstract_service_count"],
            "total_candidates_num": config["wsdream_experiment_config"]["total_candidates"],
            
            
            
            # Training parameters
            "seeds": config["random_seeds"]["values"],
            "experiment_instance_number": config["training_episodes"]["experiment_instances"],
            "n_episodes": config["training_episodes"]["total_episodes"],

            
            # Evaluation parameters
            "eval_interval": config["evaluation_settings"]["eval_interval"],
            "eval_number": config["evaluation_settings"]["eval_repetitions"],
            
            
            # Hyperparameters
            "init_alpha": config["dqn_hyperparameters"]["initial_learning_rate"],

            
            # Mode flags
            "Evaluate_mode": config["mode_flags"]["evaluate_mode"],
            
            
            # Result recording lists
            "insEval_total_rewards_list": insEval_total_rewards_list,
            "insEval_HC_rewards_list": insEval_HC_rewards_list,
            "insEval_SC_rewards_list": insEval_SC_rewards_list,
            "Training_time_record": Training_time_record
        }
        
        # Run experiment instances
        print(f"Starting CSSC-MDP experiments with config: {config_filename}")
        for _ in range(params["experiment_instance_number"]):
            # Set seed from the list
            seed = params["seeds"][0 + _]
            set_seed(seed)  # Assume set_seed is defined
            print(f"Seed: {seed}")
            
            # Run main experiment function
            print(f"CSSC-MDP Experiment Instance {_ + 1}")
            CSSC_MDP_with_Qlearing(
                bpmn_file_path=params["bpmn_file_path"],
                n_abstract_number=params["n_abstract_number"],
                total_candidates_num=params["total_candidates_num"],
                n_episodes=params["n_episodes"],
                init_alpha=params["init_alpha"],
                CSSC_evaluation_rewards_list=params["insEval_total_rewards_list"],
                CSSC_evaluation_HC_rewards_list=params["insEval_HC_rewards_list"],
                CSSC_evaluation_SC_rewards_list=params["insEval_SC_rewards_list"],
                eval_interval=params["eval_interval"],
                eval_number=params["eval_number"],
                Evaluate_mode=params["Evaluate_mode"],
                Training_time_record=params["Training_time_record"]
            )
            
            # Reset mappings if more than one instance
            if params["experiment_instance_number"] > 1:
                reset_mappings()  # Assume reset_mappings is defined
                
        
        print("\nCSSC-MDP Experiment completed successfully")
        #print(f"Total rewards recorded: {len(insEval_total_rewards_list)}")
        
        # Experiment completion information
        print(f"\nExperiment {process_id} executed successfully for {len(insEval_total_rewards_list)} random seeds")

        ###Rollout data preparation for saving
        insEval_2Dlist = []
        insEval_2Dlist.append([insEval_total_rewards_list])
        #insEval_2Dlist.append([insEval_HC_rewards_list])
        #insEval_2Dlist.append([insEval_SC_rewards_list])

        # Call the encapsulated function to save the data
        save_experiment_rollout_data(
            True,
            insEval_2Dlist=insEval_2Dlist,
            Training_time_record=Training_time_record,
            Evaluate_mode=config["mode_flags"]["evaluate_mode"],
            process_id=process_id
        )
        print("\n\n")
        
    except Exception as e:
        print(f"Error executing experiment: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":

    """
    Main entry point for running PMDP or CSSC-MDP experiments.
    
    Key Features:
    - Automatically captures process ID for unique experiment data storage.
    - Supports 3 execution modes via command-line arguments:
      1. No arguments: Runs default core logic (modify the placeholder code).
      2. 1 argument ([pmdp|cssc]): Uses default configs:
         - PMDP: "pmdp_default_config.json"
         - CSSC-MDP: "wsdream_cssc_mdp_10AS_config.json"
      3. 2 arguments ([pmdp|cssc] + config_file): Uses specified config file.
    
    Usage Examples:
    - python script.py
    - python script.py pmdp
    - python script.py cssc custom_config.json
    """


    # get the process ID of the current Python process, it can be used to store experiment data with unique names
    process_id = os.getpid()
    print(f"Python Process ID: {process_id}")

    # No arguments: Skip JSON reading and directly execute the main function body
    if len(sys.argv) == 1:
        print("No-argument mode: Directly executing main function body")
        # Replace this with your "normal main function body" code
        # For example: direct_main_execution()  # Your core logic function
        
    # 1 argument: Experiment type (pmdp or cssc), using corresponding default configuration
    elif len(sys.argv) == 2:
        exp_type = sys.argv[1].lower()
        if exp_type == "pmdp":
            print("Running experiment with PMDP default configuration")
            training_pmdp("pmdp_default_config.json", process_id=process_id)
        elif exp_type == "cssc":
            print("Running experiment with CSSC default configuration")
            training_cssc_mdp("wsdream_cssc_mdp_10AS_config.json", process_id=process_id)
        else:
            print(f"Unknown experiment type: {exp_type}, supported types are 'pmdp' or 'cssc'")
            sys.exit(1)
    
    # 2 arguments: Experiment type + configuration filename, using specified configuration
    elif len(sys.argv) == 3:
        exp_type = sys.argv[1].lower()
        config_file = sys.argv[2]
        if exp_type == "pmdp":
            print(f"Running PMDP experiment with specified configuration: {config_file}")
            training_pmdp(config_file, process_id=process_id)
        elif exp_type == "cssc":
            print(f"Running CSSC experiment with specified configuration: {config_file}")
            training_cssc_mdp(config_file, process_id=process_id)
        else:
            print(f"Unknown experiment type: {exp_type}, supported types are 'pmdp' or 'cssc'")
            sys.exit(1)
    
    # More than 2 arguments: Error
    else:
        print("Too many arguments! Usage:")
        print("  1. No arguments: python script_name.py (directly execute main function body)")
        print("  2. 1 argument: python script_name.py [pmdp|cssc] (use corresponding default configuration)")
        print("  3. 2 arguments: python script_name.py [pmdp|cssc] config_filename (use specified configuration)")
        sys.exit(1)




    