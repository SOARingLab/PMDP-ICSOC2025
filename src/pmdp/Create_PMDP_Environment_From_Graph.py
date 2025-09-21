### This file.py is used to convert a uncertainty grahr of BPMN into a graph for P-MDP

import itertools
import torch
import numpy as np
import hashlib
import random
import re, copy
from typing import Dict, List, Set, Union
# Get the project root directory
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)
from Datasets.TravelAgencyNFPsDataset.ReadData_ReturnProbabilityDistribution import sample_from_distribution
from Datasets.wsdream.Read_Dataset import probability_sampling
from src.pmdp.KPIs_Constraints_Operators_Parse_Evaluation import OPERATOR_INFO, ExpressionParser, ExpressionEvaluator
from src.pmdp.Convert_BPMN_into_Graph import parse_bpmn, Vertex as BaseVertex, Edge as BaseEdge, WeightedDirectedGraph as BaseGraph, Variable as BaseVariable, Constraint as BaseConstraint, KPI as BaseKPI
import src.utilities.Uncontrollable_Variable_Sampling as UVSampler
# pip install cityhash  for farmhash
from farmhash import FarmHash64
import numpy as np


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

class PMDPEnvironment:
    def __init__(self, graph: WeightedDirectedGraph):
        self.graph: WeightedDirectedGraph = graph
        # self.hpst is a dictionary (Dict) with keys of type vertex_id and values of another dictionary (Dict)
        self.hpst = self.build_hpst()
        self.horizon_space = self.build_horizon_space(self.hpst)
        self.horizon_space_dict = self.build_horizon_space_dict(self.horizon_space)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_state = self.initialize_state()
        self.context = self.build_pmdp_context()
        # self.dataset is a list, each itme in the list is a dictionary (Dict) representing a sample

        self.dynamic_bounds = {}
        # use the sub-expression parser to parse the KPIs
        self.parser = ExpressionParser(OPERATOR_INFO)
        self.sub_kpi_evaluator = ExpressionEvaluator(self.context)
        self.debug_mode = False

        self.rt_count = 1
        self.tp_count = 1
        


    def build_hpst(self) -> Dict[str, Dict]:
        """
        Build the Horizon Process Structure Tree (HPST) from the graph.
        """
        hpst = {}  # This should be a tree structure
        # This version code assume the graph (i.e., BPMN) has only one startVertex (i.e., startEvent)
        start_vertex: Vertex = self.graph.vertices[self.graph.startVertex_id]
        
        if start_vertex is None:
            raise ValueError("When constructing the HPST, no start vertex is found in the graph")        
        
        # The following line of code clearly shows the data structure of the nodes in the HPST
        # Initialize the HPST with the root_vertex
        hpst['TreeRoot'] = {'node_id': 'TreeRoot', 'elem_name': 'TreeRoot', 'type': 'SEQUENCE', 'children': [], 'parent': ''}
        
        # Adds the first leaf node of HPST, the start event of the process
        hpst[start_vertex.vertex_id] = {'node_id': start_vertex.vertex_id, 'elem_name': start_vertex.elem_name, 'type': 'Leaf', 'children': [], 'parent': 'TreeRoot'}
        hpst['TreeRoot']['children'].append(start_vertex.vertex_id)
        
        # pidSV is the Parent node id of sourceVertex
        # add_to_hpst(x, y) function will insert the direct descendants (in graph) of vertex x  into the HPST
        def add_to_hpst(sourceVertex_id, pidSV):
            sourceVertex : Vertex = self.graph.vertices[sourceVertex_id]
            targetVertice_ids = [edge.target for edge in self.graph.edges if edge.source == sourceVertex_id and edge.target not in hpst] #  If the vertex id (edge.target) has not been added to the HPST
            # If the out-degree of sourceVertex is 0, it returns directly, indicating that the current path (i.e., someone path) has been traversed.
            if len(targetVertice_ids) == 0:
                return
            # if the vertex has more than one outgoing edge, we will first add the XOR/AND block node to the HPST
            # then traverse the all 'id_firstVertex_eachBranch' (i.e., targetVertex -> next_target_Vertex) via the iterator 'seg_target_ids' 
            targetVertex_id = targetVertice_ids[0]
            targetVertex : Vertex = self.graph.vertices[targetVertex_id]
            
            if 'm' in sourceVertex.T_v:
                # merging parallel/splitting gateway is sourceVertex, then the gatewat block end, 
                # so the parent (pidSV) of targetVertex should be set 'the parent of parent of merging parallel/splitting gateway (i.e., grandfather)', 
                # rather than have same parent with merging parallel/splitting gateway           
                pidSV = hpst[pidSV]['parent']

            # Handle the case where the targetVertex is a splitting exclusive gateway (seg)
            # Add XOR block node and the targetVertex_id (i.e., SMi, the exclusive splitting gateway)
            if targetVertex.T_v == 'seg' and targetVertex_id not in hpst:
                # Construct subtrees for source vertices in the graph that have only one outgoing edge
                # Example: from activity to splitting exclusive gateway (seg), or from any gateway to splitting exclusive gateway (seg)
                # Meaning: from one sequence flow to a splitting exclusive gateway
                # Add XOR subtree node to HPST
                hpst['XOR_Subtree_'+targetVertex_id] = {'node_id': 'XOR_Subtree_'+targetVertex_id,  'elem_name': 'XOR_Subtree_'+ targetVertex.elem_name, 'type': 'XOR', 'children': [], 'parent': pidSV}
                hpst[pidSV]['children'].append('XOR_Subtree_'+targetVertex_id)
                # In case,remove repeated elements in 'hpst[pidSV]['children']'                         
                hpst[pidSV]['children'] = list(set(hpst[pidSV]['children']))

                # If the splitting exclusive gateway's branch contains more than one activity, add a SEQUENCE block and connect it to its descendant. Otherwise, connect the single vertex in the branch to HPST as a leaf node                       
                # Connect multiple subtrees (targetVertices) of an XOR node, representing multiple sequence flows split by the splitting gateway in the graph, as a subtree 
                seg_target_ids = [edge.target for edge in self.graph.edges if edge.source == targetVertex_id]
                # Handle all situations where the vertex has more than one outgoing edge
                for id_firstVertex_eachBranch in seg_target_ids:
                    firstVertex_eachBranch : Vertex = self.graph.vertices[id_firstVertex_eachBranch]
                    # each branch
                    # Check if the next vertex of the first vertex in the branch is a merging exclusive gateway (meg), if not, add a SEQUENCE block
                    id_nextVertex_of_firstVertex_eachBranch = [edge.target for edge in self.graph.edges if edge.source == id_firstVertex_eachBranch][0]
                    nextVertex_of_firstVertex_eachBranch : Vertex = self.graph.vertices[id_nextVertex_of_firstVertex_eachBranch]
                    if nextVertex_of_firstVertex_eachBranch.T_v != 'meg':
                        hpst['SEQUENCE_Subtree_'+id_firstVertex_eachBranch] = {'node_id': 'SEQUENCE_Subtree_'+id_firstVertex_eachBranch, 'elem_name': 'SEQUENCE_Subtree_'+ firstVertex_eachBranch.elem_name, 'type': 'SEQUENCE', 'children': [], 'parent': 'XOR_Subtree_'+targetVertex_id}
                        hpst['XOR_Subtree_'+targetVertex_id]['children'].append('SEQUENCE_Subtree_'+id_firstVertex_eachBranch)

                        # In case, remove repeated elements in 'hpst['XOR_Subtree_'+targetVertex_id]['children']'                         
                        hpst['XOR_Subtree_'+targetVertex_id]['children'] = list(set(hpst['XOR_Subtree_'+targetVertex_id]['children']))                        
                            
                        if targetVertex_id not in hpst:
                            # Add SMi (i.e., the splitting exclusive gateway) with SEQUENCE block to HPST as described in Table 2 of the paper     
                            hpst[targetVertex_id] = {'node_id': targetVertex_id, 'elem_name': targetVertex.elem_name, 'type': 'Leaf', 'children': [], 'parent': 'XOR_Subtree_'+targetVertex_id}
                            hpst['XOR_Subtree_'+targetVertex_id]['children'].append(targetVertex_id)

                            # In case, remove repeated elements in 'hpst['XOR_Subtree_'+targetVertex_id]['children']'                         
                            hpst['XOR_Subtree_'+targetVertex_id]['children'] = list(set(hpst['XOR_Subtree_'+targetVertex_id]['children']))                                                              
                                
                        add_to_hpst(targetVertex_id, 'SEQUENCE_Subtree_'+id_firstVertex_eachBranch)  
                    else :
                            
                        if targetVertex_id not in hpst:                                
                            # Adding SMi (i.e., the splitting exclusive gateway) with SEQUENCE block to HPST for Table.2 in the paper      
                            hpst[targetVertex_id] = {'node_id': targetVertex_id, 'elem_name': targetVertex.elem_name, 'type': 'Leaf', 'children': [], 'parent': 'XOR_Subtree_'+targetVertex_id}
                            hpst['XOR_Subtree_'+targetVertex_id]['children'].append(targetVertex_id)  

                            # In case, remove repeated elements in 'hpst['XOR_Subtree_'+targetVertex_id]['children']'                         
                            hpst['XOR_Subtree_'+targetVertex_id]['children'] = list(set(hpst['XOR_Subtree_'+targetVertex_id]['children']))            

                        # Add the first vertex of each branch to HPST                      
                        hpst[id_firstVertex_eachBranch] = {'node_id': id_firstVertex_eachBranch, 'elem_name': firstVertex_eachBranch.elem_name,'type': 'Leaf', 'children': [], 'parent': 'XOR_Subtree_'+targetVertex_id}
                        hpst['XOR_Subtree_'+targetVertex_id]['children'].append(id_firstVertex_eachBranch)

                        # In case, remove repeated elements in 'hpst['XOR_Subtree_'+targetVertex_id]['children']'                         
                        hpst['XOR_Subtree_'+targetVertex_id]['children'] = list(set(hpst['XOR_Subtree_'+targetVertex_id]['children']))     

                        add_to_hpst(targetVertex_id, 'XOR_Subtree_'+targetVertex_id)                            

                
                                        
            # Handle the case where the targetVertex is a splitting parallel gateway (spg)
            # Add AND block node and the targetVertex_id (i.e., SMi, the splitting parallel gateway)
            elif targetVertex.T_v == 'spg' and targetVertex_id not in hpst:
                #Adding AND subtree node to HPST
                hpst['AND_Subtree_'+targetVertex_id] = {'node_id': 'AND_Subtree_'+targetVertex_id, 'elem_name': 'AND_Subtree_'+ targetVertex.elem_name, 'type': 'AND', 'children': [], 'parent': pidSV}
                hpst[pidSV]['children'].append('AND_Subtree_'+targetVertex_id)

                # In case, remove repeated elements in 'hpst[pidSV]['children']'                         
                hpst[pidSV]['children'] = list(set(hpst[pidSV]['children']))                     
                
                # If the splitting parallel gateway's branch contains more than one activity, then the SEQUENCE block needs to be added, and then connect with its descendant. Otherwise, just connect the single vertex in the branch to hpst as a leaf node.                        
                # Connecting multiple subtrees (targetVertices) of an AND node as a subtree, i.e., multiple sequence flows split by the splitting gateway in graph as a subtree  
                spg_target_ids = [edge.target for edge in self.graph.edges if edge.source == targetVertex_id]
                # handle all situations for the vertex has out-degree more than 1                          
                for id_firstVertex_eachBranch in spg_target_ids:
                    firstVertex_eachBranch : Vertex = self.graph.vertices[id_firstVertex_eachBranch]
                    
                    # each branch
                    # Check if the next vertex of the first vertex in the branch is a merging parallel gateway (mpg), if not, add a SEQUENCE block
                    id_nextVertex_of_firstVertex_eachBranch = [edge.target for edge in self.graph.edges if edge.source == id_firstVertex_eachBranch][0]
                    nextVertex_of_firstVertex_eachBranch : Vertex = self.graph.vertices[id_nextVertex_of_firstVertex_eachBranch]
                    if nextVertex_of_firstVertex_eachBranch.T_v != 'mpg':
                        hpst['SEQUENCE_Subtree_'+id_firstVertex_eachBranch] = {'node_id': 'SEQUENCE_Subtree_'+id_firstVertex_eachBranch, 'elem_name': 'SEQUENCE_Subtree_'+ firstVertex_eachBranch.elem_name, 'type': 'SEQUENCE', 'children': [], 'parent': 'AND_Subtree_'+targetVertex_id}
                        hpst['AND_Subtree_'+targetVertex_id]['children'].append('SEQUENCE_Subtree_'+id_firstVertex_eachBranch)

                        # In case, remove repeated elements in 'hpst['AND_Subtree_'+targetVertex_id]['children']'                         
                        hpst['AND_Subtree_'+targetVertex_id]['children'] = list(set(hpst['AND_Subtree_'+targetVertex_id]['children']))  
                            
                        if targetVertex_id not in hpst:                                
                            # Add SMi (i.e., the splitting parallel gateway) with SEQUENCE block to HPST as described in Table 2 of the paper      
                            hpst[targetVertex_id] = {'node_id': targetVertex_id, 'elem_name': targetVertex.elem_name, 'type': 'Leaf', 'children': [], 'parent': 'AND_Subtree_'+targetVertex_id}
                            hpst['AND_Subtree_'+targetVertex_id]['children'].append(targetVertex_id)                                                            

                            # In case, remove repeated elements in 'hpst['AND_Subtree_'+targetVertex_id]['children']'                         
                            hpst['AND_Subtree_'+targetVertex_id]['children'] = list(set(hpst['AND_Subtree_'+targetVertex_id]['children']))   

                        add_to_hpst(targetVertex_id, 'SEQUENCE_Subtree_'+id_firstVertex_eachBranch)  
                    else:                     
                            
                        if targetVertex_id not in hpst:                                
                            # Adding SMi (i.e., the splitting parallel gateway) with SEQUENCE block to HPST for Table.2 in the paper      
                            hpst[targetVertex_id] = {'node_id': targetVertex_id, 'elem_name': targetVertex.elem_name, 'type': 'Leaf', 'children': [], 'parent': 'AND_Subtree_'+targetVertex_id}
                            hpst['AND_Subtree_'+targetVertex_id]['children'].append(targetVertex_id)      

                            # In case, remove repeated elements in 'hpst['AND_Subtree_'+targetVertex_id]['children']'
                            hpst['AND_Subtree_'+targetVertex_id]['children'] = list(set(hpst['AND_Subtree_'+targetVertex_id]['children']))                                
                                
                        # Add the first vertex of each branch to HPST
                        hpst[id_firstVertex_eachBranch] = {'node_id': id_firstVertex_eachBranch, 'elem_name': firstVertex_eachBranch.elem_name, 'type': 'Leaf', 'children': [], 'parent': 'AND_Subtree_'+targetVertex_id}
                        hpst['AND_Subtree_'+targetVertex_id]['children'].append(id_firstVertex_eachBranch)

                        # In case, remove repeated elements in 'hpst['AND_Subtree_'+targetVertex_id]['children']'
                        hpst['AND_Subtree_'+targetVertex_id]['children'] = list(set(hpst['AND_Subtree_'+targetVertex_id]['children']))

                        add_to_hpst(targetVertex_id, 'AND_Subtree_'+targetVertex_id) 
                                
            else :                       
                # End of XOR or AND gateway block (subtree), the merging  parallel/splitting gateway be added to hpst (i.e., SMi of parallel/splitting gateway, in Table 2 of the paper)
                # The targetVertex must not be spg or seg
                if 'm' in targetVertex.T_v and targetVertex_id not in hpst:
                    incoming_ids = [edge.source for edge in self.graph.edges if edge.target == targetVertex_id]
                    contain_SEQUENCE_block = 0 
                    for incoming_id in incoming_ids:
                        incoming_incoming_ids = [edge.source for edge in self.graph.edges if edge.target == incoming_id]
                        for incoming_incoming_id in incoming_incoming_ids:
                            incoming_incoming_Vertex : Vertex = self.graph.vertices[incoming_incoming_id]
                            if 'g' not in incoming_incoming_Vertex.T_v:
                                contain_SEQUENCE_block = 1   
                    if contain_SEQUENCE_block == 1:    
                        # If the current incoming path (from the current level's splitting to merging) contains more than one node, i.e., it includes a SEQUENCE block
                        hpst[targetVertex_id] = {'node_id': targetVertex_id, 'elem_name': targetVertex.elem_name, 'type': 'Leaf', 'children': [], 'parent': hpst[pidSV]['parent']}
                        hpst[hpst[pidSV]['parent']]['children'].append(targetVertex_id)   
                        add_to_hpst(targetVertex_id, hpst[pidSV]['parent'])
                    else:
                        # If the current incoming path (from the current level's splitting to merging) contains only one node or more, i.e., it does not include a SEQUENCE block
                        hpst[targetVertex_id] = {'node_id': targetVertex_id, 'elem_name': targetVertex.elem_name, 'type': 'Leaf', 'children': [], 'parent': pidSV}
                        hpst[pidSV]['children'].append(targetVertex_id)   
                        add_to_hpst(targetVertex_id, pidSV)                                                                                                                                            
                else:
                    # Normal case: for example, from activity to activity, or from event to activity
                    # If the targetVertex (in the graph) is an 'oth' vertex, and the sourceVertex (in the graph) is not any gateways ('seg, spg, meg, mpg'), it means the targetVertex and sourceVertex share the same parent node in HPST.
                    if targetVertex_id not in hpst: 
                        hpst[targetVertex_id] = {'node_id': targetVertex_id, 'elem_name': targetVertex.elem_name, 'type': 'Leaf', 'children': [], 'parent': pidSV}
                        hpst[pidSV]['children'].append(targetVertex_id)
                    add_to_hpst(targetVertex_id, pidSV)
                
                                            
        add_to_hpst(start_vertex.vertex_id, 'TreeRoot')
        return hpst

    # The return value of build_horizon_space is horizon_space. If the elements in it are of type str, it represents a one-dimensional time step (i.e., a one-dimensional state consisting of a single activated vertex) v.
    # If the elements are of type tuple, (each element in the tuple can be of type str (corresponding to a single activated vertex) or a nested tuple), 
    # it represents a multidimensional time step state (i.e., a multidimensional state consisting of multiple activated vertices).
    def build_horizon_space(self, hpst: Dict[str, Dict], root_id: str = 'TreeRoot') -> Set[Union[str, tuple]]:
        """
        Build the horizon space based on the rules defined in the HPST and create a lookup dictionary for fast search.

        :param hpst: Hierarchical Process State Tree (HPST) representing the process structure.
        :param root_id: The root node ID of the HPST.

        """
        horizon_space = set()
       

        def traverse(node_id: str):
            node = hpst[node_id]
            if self.is_leaf(node):
                horizon_space.add(node_id)
               
            elif self.is_sequence(node):
                for child_id in node['children']:
                    traverse(child_id)
            elif self.is_xor(node):
                for child_id in node['children']:
                    traverse(child_id)
            elif self.is_and(node):
                children_ids = [child_id for child_id in node['children'] if not (child_id.startswith('Gateway') and hpst[child_id]['type'] == 'Leaf')]
                # Calculate the Cartesian product for each child node and compute its horizon space if it is not a leaf node
                children_horizon_spaces = []
                for child_id in children_ids:
                    if self.is_leaf(hpst[child_id]):
                        children_horizon_spaces.append([child_id])
                    else:
                        # Pass the subtree to build_horizon_space
                        child_horizon_space = self.build_horizon_space(self.extract_subtree(hpst, child_id), child_id)
                        children_horizon_spaces.append(list(child_horizon_space))
                cartesian_product = self.list_cartesian_product_horizon(children_horizon_spaces)
                for product in cartesian_product:
                    # Horizon_space is the 'set' datatype whose elements, if str datatype, indicate that it is a one-dimensional time step (i.e., a one-dimensional state consisting of a single activated vertex) v. 
                    # Conversely, if it is the 'tuple' datatype，and each element in the tuple must be of a str datatype (corresponding to a single activated vertex), 
                    # it indicates that it is a multidimensional time-step state (i.e., a multidimensional state consisting of multiple activated vertices). 
                    horizon_space.add(tuple(product))
                  

                sm_ids = [child_id for child_id in node['children'] if child_id.startswith('Gateway') and hpst[child_id]['type'] == 'Leaf']
                horizon_space.update(sm_ids)

        traverse(root_id)
        return horizon_space
    
    def build_horizon_space_dict(self, horizon_space: set) -> dict:
        """
        Build a dictionary that maps a sorted list representation of each horizon_space item 
        to its corresponding original item.

        :param horizon_space: A set containing items which are either strings (1-dimensional states)
                            or tuples (multi-dimensional states) or nested tuple structures.
        :return: A dictionary where keys are sorted list representations and values are the original items.
        """
        horizon_space_dict = {}

        for item in horizon_space:
            # Convert the item to a flat list
            item_as_list = self.convert_horizon_item_to_list(item)
            
            # Sort the list to make sure order does not affect the lookup
            sorted_item_as_tuple = tuple(sorted(item_as_list))
            
            # Store the original item in the dictionary with the sorted list as the key
            horizon_space_dict[sorted_item_as_tuple] = item

        return horizon_space_dict
    
        '''
        Usage example:
        horizon_space_example = {"Task1", ("Task2", ("Task3", "Task4")), ("Task5", ("Task6", ("Task7", "Task8")))}
        horizon_space_dict = build_horizon_space_dict(horizon_space_example)
        print("Horizon Space Dictionary:", horizon_space_dict)
        '''

    
    def convert_horizon_item_to_list(self, horizon_item: Union[str, tuple]) -> list:
        """
        Convert a horizon space item, which could be multi-dimensional and nested, to a flat list.

        :param horizon_item: An item from the horizon space, which can be a string (one-dimensional state), 
                            a tuple (multi-dimensional state), or a nested tuple structure.
        :return: A flat list representation of the horizon item.
        """
        if isinstance(horizon_item, str):
            # Base case: if the item is a string, return it as a single-element list
            return [horizon_item]
        
        elif isinstance(horizon_item, tuple):
            # Recursive case: if the item is a tuple, flatten each element inside it
            flat_list = []
            for element in horizon_item:
                # Recursively flatten each element of the tuple
                flat_list.extend(self.convert_horizon_item_to_list(element))
            return flat_list
        
        else:
            # Raise an error if the input is not a valid horizon space item
            raise ValueError("Invalid horizon space item. Must be a string or tuple.")

        '''
        Example case:
        horizon_space_example = {"Task1", ("Task2", ("Task3", "Task4")), ("Task5", ("Task6", ("Task7", "Task8")))}
        converted_list = [convert_horizon_item_to_list(item) for item in horizon_space_example]
        print("Converted Horizon Space Items:", converted_list)
        '''


    def find_in_horizon_space_dict(self, horizon_space_dict: dict, target_list: list) -> Union[str, tuple, None]:
        """
        Find the corresponding item in horizon_space based on the given list, ignoring the order of elements.

        :param horizon_space_dict: A dictionary where keys are list representations (sorted as tuples)
                                and values are the original items.
        :param target_list: The list to match against the keys in horizon_space_dict.
        :return: The corresponding item from horizon_space_dict or None if no match is found.
        """
        # Sort the target list to ensure order does not affect the lookup
        sorted_target = tuple(sorted(target_list))
        
        # Search for the sorted target in the dictionary
        return horizon_space_dict.get(sorted_target, None)

        '''
        usage example:
        usage example:
        # Example: Search regardless of order
        target_list_example = ["Task4", "Task3", "Task2"]
        matching_item = find_in_horizon_space_dict(horizon_space_dict, target_list_example)
        print("Matching Horizon Space Item:", matching_item)
        '''

    def extract_subtree(self, hpst: Dict[str, Dict], root_id: str) -> Dict[str, Dict]:
        """
        Extract the subtree from hpst starting at root_id.
        """
        subtree = {}

        def extract(node_id: str):
            subtree[node_id] = hpst[node_id]
            for child_id in hpst[node_id]['children']:
                extract(child_id)

        extract(root_id)
        return subtree

    def is_leaf(self, node: Dict) -> bool:
        """
        Check if a node is a leaf.
        """
        return node['type'] == 'Leaf'

    def is_sequence(self, node: Dict) -> bool:
        """
        Check if a node is a sequence.
        """
        return node['type'] == 'SEQUENCE'

    def is_xor(self, node: Dict) -> bool:
        """
        Check if a node is an XOR gateway.
        """
        return node['type'] == 'XOR'

    def is_and(self, node: Dict) -> bool:
        """
        Check if a node is an AND gateway.
        """
        return node['type'] == 'AND'
    
    
    def list_cartesian_product_horizon(self, lists: List[List[str]]) -> List[List[str]]:
        return list(itertools.product(*lists))
    
    
    def get_horizon_space(self) -> Set[str]:
        return self.horizon_space


    
    ##### hash function for tensor #####
    def string_to_hash_int64(self, s: str) -> int:
        hash_value = FarmHash64(s)

        # The hash value is converted to a string and truncated to an 18-digit decimal integer.
        hash_str = str(hash_value)[:18]

        # Convert back to integer
        truncated_hash = int(hash_str)

        # Check if it is within the int64 range
        if -9223372036854775808 <= truncated_hash <= 9223372036854775807:
            return truncated_hash
        else:
            raise ValueError("Hash value exceeds int64 range")
    
    def convert_tensor_to_strings(self, filtered_tensor: torch.Tensor, int64_to_string_map: dict, wait_actions_int64_to_string_map: dict) -> List[str]:
        # 将 wait_actions_int64_to_string_map 和 int64_to_string_map 合并
        combined_map = {**wait_actions_int64_to_string_map, **int64_to_string_map}

        # Initialize a list to store the string representations
        result = []

        # If the size of filtered_tensor is [1]
        if filtered_tensor.dim() == 1:
            string_row = [combined_map[int(val.item())] for val in filtered_tensor]
            result= string_row
        else:
            # Iterate over each row of filtered_tensor
            for row in filtered_tensor:
                # Convert each integer in the row to its corresponding string representation
                string_row = [combined_map[int(val.item())] for val in row]
                result.append(string_row)
        
        return result


    ###### State space real-time parsing ######

    # the 'initialize_state' function should be called at the beginning of each episode (i.e., the '__init__' of PMDPEnviroment ) to initialize the state of the environment
    def initialize_state(self):
        state = {}
        for vertex in self.graph.vertices.values():
            for var in vertex.C_v + vertex.U_v:
                state[var.name] = None  # Initial state, all variables are unassigned
                state[f'{var.name}_direction'] = var.direction
        return state
    

    def get_current_state_Q(self):
        current_state_without_HorizonAtion = {}
        for state in self.current_state:
            if 'Horizon_Action' not in state and '_direction' not in state:
                current_state_without_HorizonAtion[state] = self.current_state[state]
        return current_state_without_HorizonAtion
    

    def get_current_context_Q(self):
        context_Q = {}
        for state in self.current_state:
            if 'Horizon_Action' not in state and '_direction' not in state:
                context_Q[state] = self.current_state[state]
        for kpi in self.graph.KPIs:
            context_Q[kpi.name] = self.context[kpi.name]

        return context_Q

    
    # The input of 'parse_current_state' is the current time-step v (i.e., active vertex or tertices), which can be a single activated vertex or a concurrent block
    # The return of 'parse_current_state' is a dict, in which the key is the variable name and the value is the variable value
    def parse_current_state(self, v: Union[str, tuple]) -> Dict[str, any]:
        """
        Parse the state of the current time-step v, considering concurrent blocks and \(\mathbbm{v}_h\).
        """
        if isinstance(v, str):
            # Single activated vertex, parse its state directly
            return self.parse_state_for_vertex(v)
        elif isinstance(v, tuple):
            # Concurrent block or nested concurrent block, recursively parse the state of each layer
            return self.parse_state_for_vertices(v)

    def parse_state_for_vertex(self, vertex_id: str) -> Dict[str, any]:
        """
        Parse the state of a single vertex.
        """
        vertex = self.graph.vertices[vertex_id]
        relevant_vars = []
        
        # Find the observable variables on the relevant paths (i.e., path-aware PMDP) under the current time-step (i.e., activated vertex v)
        for one_path_observed_set in vertex.observed_set:
            all_vars_exist = True
            for var in one_path_observed_set:
                if self.current_state[var.name] is None:
                    all_vars_exist = False
                    break
            if all_vars_exist:
                for var in one_path_observed_set:
                    relevant_vars.append(var.name)
        
        # Based on the current oberserved variables set (relevant_vars) of v, retrieve the specific state from the real-time multidimensional state instance
        current_state_for_vertex = {var_name: self.current_state[var_name] for var_name in relevant_vars}
        # The current_state_for_vertex will be a dictionary, where the key is the variable name and the value is the variable value
        return current_state_for_vertex

    def parse_state_for_vertices(self, vertices: tuple) -> Dict[str, any]:
        """
        Parse the state of parallel or nested parallel structures, including the corresponding \(\mathbbm{v}_h\).
        """
        combined_state = {}

        # Get the splitting parallel gateway (\(\mathbbm{v}_h\)) of the current layer of concurrent blocks
        vh = self.get_vh(vertices)
        if vh:
            vh_state = self.parse_state_for_vertex(vh)
            combined_state.update(vh_state)

        for v in vertices:
            if isinstance(v, tuple):  # Nested concurrent structure, i.e., there be a tuple in the tuple
                nested_state = self.parse_state_for_vertices(v)
                combined_state.update(nested_state)
            else:  # Single vertex, actually the v will be a str
                vertex_state = self.parse_state_for_vertex(v)
                combined_state.update(vertex_state)

        # The combined_state will be a dictionary, where the key is the variable name and the value is the variable value
        return combined_state

    def get_vh(self, vertices: tuple) -> str:
        """
        Get the splitting parallel gateway (\(\mathbbm{v}_h\)) of the current layer of concurrent blocks.
        The meaning of H_v is, if H_v is a stack, each element is like ($v_i$,$k$,$n$), it indicates that $v_i$ splits $n$ threads, and $v_j$ is associated with the $k$th thread of the concurrency block $v_i$.
        """
        for v in vertices:
            if isinstance(v, str):  # Find the single vertex of the current layer
                vertex = self.graph.vertices[v]
                if vertex.H_v:
                    return vertex.H_v[-1][0]  # Get the splitting parallel gateway of the current layer, i.e., top
        return None    
    

    ##### Action space real-time parsing #####

    def get_all_active_vertices(self, v: Union[str, tuple]) -> List[str]:
        """
        Recursively parse the nested structure of v to get all vertices.
        """
        if isinstance(v, str):
            return [v]
        elif isinstance(v, tuple):
            vertices = []
            for item in v:
                vertices.extend(self.get_all_active_vertices(item))
            return vertices
        else:
            raise ValueError("Invalid type for v")

    # The 'parse_variable_domain' function is used to parse the variable domain of the controllable variable of the vertex
    # If the vertex has controllable variables, the 'wait' action is added to the action space of each variable domain
    # If the vertex does not have any controllable variables, the 'finish' action is added to the action space of the vertex
    # Except the 'empty...' vertex, the 'wait' and 'finish' actions are only one can be selected in the action space of the vertex
    # Because if the vertex has no controllable variables, we don't need to wait. We can directly transite to next vertex
    # For each 'wait' action, and use vertex_id as the wait action name (i.e., 'vertex_id=wait' in wait_actions_int64_to_string_map
    def get_action_space(self, v: Union[str, tuple]) -> torch.Tensor:
        # Initialize an empty tensor list
        current_action_spaces = []
        # Initialize an empty list for the 'wait' or 'none' domain abuot each vertex
        vertices_with_none_actions = []

        # Create a mapping for int64 to string conversion if necessary
        # this dict only contain controllable variables
        int64_to_string_map = {}

        # Get all active vertices list in the current state
        vertices_ids = self.get_all_active_vertices(v)

        wait_actions_TagindexTensor_list = []
        Tag_index_inTensor = 0
        wait_actions_int64_to_string_map = {}

        # Iterate through each active vertex to collect action spaces
        for vertex_id in vertices_ids:
            controllable_vars_dict = self.get_controllable_vars(vertex_id)
            # Initialize an empty list for the parised variable domain    
            var_domain_list = []
            temp_tag = []

            if controllable_vars_dict or vertex_id.startswith('empty'):
                temp_tag.append(Tag_index_inTensor)

            # Has more than one controllable variables for the vertex
            # The 'wait' action is added to the action space of each variable domain if the vertex has controllable variables, 
            # only when all controllable variables in the vertes are set to 'wait' is legal, we will filter the illegal actions by 'filter_tensor'
            for var_name, var_vertexId_name_domain_controlType in controllable_vars_dict.items():
                # Parse the variable domain, it will contain a 'wait' action, and use vertex_id as the wait action name (i.e., 'vertex_id=wait' in wait_actions_int64_to_string_map)
                var_domain_list.append(self.parse_variable_domain(vertex_id, var_name, var_vertexId_name_domain_controlType[2], int64_to_string_map, wait_actions_int64_to_string_map))
                Tag_index_inTensor += 1   

            # for the 'empty...' vertex, add the 'wait' to the action space of the vertex
            # 'empty...' vertex has no controllable and uncontrollable variables, so this call 'parse_variable_domain' will only add the 'wait' action
            
            if vertex_id.startswith('empty'):
                # the 'var_domain_list' will be none, before calling the 'parse_variable_domain' function, we have added 'wait' as an action
                var_domain_list.append(self.parse_variable_domain(vertex_id, '', [], int64_to_string_map, wait_actions_int64_to_string_map))
                Tag_index_inTensor += 1 

            if controllable_vars_dict or vertex_id.startswith('empty'):
                temp_tag.append(Tag_index_inTensor)
                wait_actions_TagindexTensor_list.append(temp_tag)

            # Not any controllable variables for the vertex
            # If the vertex does not have any controllable variables, even the vertex has uncontrollable variables, add 'finish' as an action for the vertex means the vertex is executed
            # because we assume those variables are independent or have a joint distribution in same vertex for some controllable and uncontrollbale variables, 
            # so if the vertex does not have any controllable variables, even the vertex has uncontrollable variables, we can directly add 'finish' as an action for the vertex, 
            # add 'wait' is not necessary and without any meaning, because by  the 'finish' we can get more information in new state. Rather, if have 'wait', nothing will happen for the vertex
            # Additional, the 'empty' vertex has no controllable and uncontrollable variables, so we add 'finish' as an action for the vertex, before the 'empty' vertex, we have added 'wait' as an action for the vertex
            if not controllable_vars_dict or vertex_id.startswith('empty'):
                hashed_value = self.string_to_hash_int64(f"{vertex_id}=finish")
                int64_to_string_map[hashed_value] = f"{vertex_id}=finish"     
                vertices_with_none_actions.append(hashed_value)
                if vertex_id.startswith('empty'):
                    var_domain_list[0].append(hashed_value)
                else:
                    var_domain_list.append([hashed_value])


            # append the parsed variable domain to the torch list
            for item in var_domain_list:
                current_action_spaces.append(torch.tensor(item, dtype=torch.int64, device=self.device))

        if len(current_action_spaces) == 1 and len(vertices_ids) == 1 and current_action_spaces[0].size()[0] > 1:
            # If there is only one vertex and one controllable variable, return the action space directly
            action_space = current_action_spaces[0].clone().detach().unsqueeze(1)
        else:
            # Calculate the Cartesian product of all action spaces of variables, that is, the action space of the current time-step (i.e., current state)
            # the 'torch.cartesian_prod' function can only take 1-D tensors as input, that is, the len(current_action_spaces) should larger than 1
            action_space = torch.cartesian_prod(*current_action_spaces)


        # because the 'wait' should be at the vertex level to construct cartesian product, due to the 'torch.cartesian_prod' function can only take 1-D tensors as input
        # so we add the 'wait' to each variable domain (i.e., variable level), so will result some redundant (illegal) actions in the action space
        # we need to filter these redundant actions 
        filtered_action_space = self.filter_tensor(action_space, wait_actions_TagindexTensor_list, wait_actions_int64_to_string_map)

        # debug print
        #print("action_space size: ", action_space.size())
        #print("filtered_action_space size: ", filtered_action_space.size())
        #original_action_space = self.convert_tensor_to_strings(filtered_action_space, wait_actions_int64_to_string_map, int64_to_string_map)
        #print(f"The size of original_action_space: {len(original_action_space)}, {len(original_action_space[0])}")
        #print("-----------------")

        return filtered_action_space, int64_to_string_map, wait_actions_int64_to_string_map

    
    # The 'isin' function is used to check if the elements in the 'elements' tensor are in the 'test_elements' tensor 
    def isin(self, elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
        elements = elements.unsqueeze(-1)
        test_elements = test_elements.unsqueeze(0)
        result = elements == test_elements
        return result.any(dim=-1)

    # The 'filter_tensor' function is used to filter the tensor based on the given column ranges and the wait_actions_int64_to_string_map
    def filter_tensor(self, tensor: torch.Tensor, col_ranges: list, wait_actions_int64_to_string_map: dict) -> torch.Tensor:
        # Initialize the bool index based on 'row', initially all rows are retained
        row_mask = torch.ones(tensor.size(0), dtype=torch.bool, device=tensor.device)
        
        wait_action_keys = torch.tensor(list(wait_actions_int64_to_string_map.keys()), device=tensor.device)
        
        for col_range in col_ranges:
            col_start, col_end = col_range

            # Check the dimensionality of the tensor
            if tensor.dim() == 1:
                sub_tensor = tensor[col_start:col_end]
            else:
                sub_tensor = tensor[:, col_start:col_end]

            # Check if any element in the given columns of each row is in wait_action_keys
            condition = self.isin(sub_tensor, wait_action_keys)

            # If the row contains at least one element in wait_action_keys
            has_wait_action = condition.any(dim=1) if tensor.dim() > 1 else condition.any()
            
            if has_wait_action.any():
                # Check if all elements belong to wait_action_keys
                all_wait = condition.all(dim=1) if tensor.dim() > 1 else condition.all()

                # Check if all elements belong to wait_action_keys and are the same
                if tensor.dim() == 1:
                    all_same_wait = all_wait & (sub_tensor == sub_tensor[0])
                else:
                    all_same_wait = all_wait & (sub_tensor.eq(sub_tensor[:, 0].unsqueeze(1)).all(dim=1))
                
                # The rows that need to be removed are those that have wait_action but not all_same_wait
                rows_to_delete = has_wait_action & ~all_same_wait

                # Update the row mask to keep only the needed rows
                row_mask = row_mask & ~rows_to_delete

        # Use bool tensor as index to filter the tensor
        return tensor[row_mask] if tensor.dim() > 1 else tensor[row_mask.nonzero(as_tuple=True)[0]]


    def get_controllable_vars(self, vertex_id: str) -> Dict[str, List]:
        if vertex_id in self.graph.vertices:
            return {var.name: [var.vertex_id, var.name, var.domain, var.controlType] for var in self.graph.vertices[vertex_id].C_v}
        return {}
    
    # this function only for the controllable variables
    def parse_variable_domain(self, vertex_id: str, var_name: str, var_domain: list, int64_to_string_map: Dict[int, str], wait_actions_int64_to_string_map: Dict[int, str]) -> List:
        parsed_values = []
        for item in var_domain:
            if ',' not in item:
                # If not a vector (i.e., a number or character), item will be like: '[0]' or '[character]', first remove the symbols: [ and ] and then concatenate it with the variable name as a string
                item = item.strip('[]')

            # After processing item, or if it is a vector, item will be like: '[0,1,2]', directly concatenate item with variable name as a string
            # Add the variable name to each value and convert it to a string
            full_item = f"{var_name}={item}"
            # Use hashing to map strings to unique integers
            hashed_value = self.string_to_hash_int64(full_item)
            # Store the mapping between hash values and original strings in a dictionary
            int64_to_string_map[hashed_value] = full_item
            # Add the hash value to the list
            parsed_values.append(hashed_value)

        # Add 'wait' as an action state for the vertex, if the vertex has at least one controllable variable, or the vertex is the 'empty...' vertex (i.e., for waiting synchronization)
        hashed_value = self.string_to_hash_int64(f"{vertex_id}=wait")
        wait_actions_int64_to_string_map[hashed_value] = f"{vertex_id}=wait"    
        parsed_values.append(hashed_value)

        return parsed_values

    # When the agent selects an action in PMDP, the 'parse_and_update_state' function should be called to parse the action and update the current state instance
    def parse_and_update_state(self, action: torch.Tensor, int64_to_string_map: Dict[int, str], wait_actions_int64_to_string_map: Dict[int, str]) -> list[List[str], List[str], List[str]]:
        """
        Parses a specific action and updates the current state instance based on the parsed variables.

        :param action: A row from the filtered_action_space tensor (a specific action).
        :param int64_to_string_map: Dictionary mapping int64 hashed values to their original string representations.
        :param wait_actions_int64_to_string_map: Dictionary mapping int64 hashed wait actions to their original string representations.
        """
        # Combine both maps for easy lookup
        combined_map = {**int64_to_string_map, **wait_actions_int64_to_string_map}

        # Initialize lists to store updated variables, wait variables, and finished vertices
        updated_vars = []
        wait_vertices = []
        finished_vertices = []

        # Parse the action into its original string content
        parsed_action = [combined_map[int(val.item())] for val in action]

        
        if self.debug_mode:
            print(f"Action: {parsed_action}")   

        # These variable is related to the 'wait' operation, so need make sure its value is 'None' in the current state
        wait_variables = []


        # Update the current state instance
        for action_item in parsed_action: 
            
            # Parsing action strings
            var_name, value = action_item.split('=')

            if value == 'finish':
                # value will be like 'var_name=finish', in whchi the var_name is the vertex_id
                # Handle the finish operation of vertex
                # That is, there are no controllable variables under this vertex, and the finish operation is executed, which is used to indicate that the vertex has been executed.
                # the var_name is the vertex_id, because the vertex has no controllable variables, and the action is 'finish'
                finished_vertices.append(var_name)
            elif value == 'wait':
                # The 'var_name' in 'wait' case, 'var_name' is the vertex_id, not the controllable or uncontrollable variable name
                # If having one varibale in the vertex is set to 'wait', which means all variables in same vertex will be set to 'wait'
                # Handle wait operation
                # here 'var_name' is the vertex_id, not the controllable or uncontrollable variable name
                
                # 2024-10-01 'subStateSpace' wait state and action are be deactived,  the following code, never is executed if deactive
                # the horizon 'wait' action is not handled in here
                # It may happen that a controllable variable under a vertex is set to wait, and other variables are assigned values. This is because these variables belong to different subStateSpaces.
                # PMDP will transfer itself, and the active node of the next time step still contains the vertex
                # But we need to set all variables under this vertex to 'None' initial value, making other variable assignments invalid
                wait_vertices.append(var_name)
                for var in self.graph.vertices[var_name].C_v:
                    # var.name is the name of controllable variable
                    self.current_state[var.name] = None
                    self.context[var.name] = None
                    wait_variables.append(var.name)
            else:
                # value will be like 'var_name=value', in which the var_name is the name of controllable variable
                # Handle ordinary assignment operations
                #value, direction = value.split('#')
                if var_name not in wait_variables:
                    # The value of the variable is updated only when all controllable variables in the vertex to which the variable belongs are not set to wait variables.
                    # If a controllable variable in the vertex to which the variable belongs is set to wait, other variable assignments are invalid.
                    self.current_state[var_name] = value
                    # context consist of all variables and KPIs, current_state only consist of variables
                    self.context[var_name] = self.evaluate_expression(value)
                    updated_vars.append(var_name)

        # debug print to verify the updated state
        #print(f"Updated current state: {self.current_state}")
     
        return updated_vars, wait_vertices, finished_vertices


    ##### State Traisition Function #####

    # The 'get_uncontrollable_vars' function is used to get the uncontrollable variables for the activated vertices
    def get_uncontrollable_vars(self, activated_vertices: List[str]) -> List[Variable]:
        uncontrollable_vars = []
        for vertex_id in activated_vertices:
            if vertex_id in self.graph.vertices:
                uncontrollable_vars.extend([var for var in self.graph.vertices[vertex_id].U_v])
        return uncontrollable_vars
        

    def parse_uncontrollable_variable_domain(self, variable: Variable) -> dict:
        """
        Parses the domain of an uncontrollable variable and extracts the value-probability pairs.

        :param variable: The uncontrollable variable to parse.
        :return: A dictionary where the keys are the parsed values and the values are their corresponding probabilities.
        """
        parsed_domain = {}

        # the value_prob_pair is like ['[0]', '0.5'], ['[1]', 0.5], or [[0,1], 0.5], [[1,2], 0.5], etc.
        for value_prob_pair in variable.domain:
            value_str = value_prob_pair[0]
            prob = value_prob_pair[1]

            # Check if the value is a vector or a single value
            if ',' not in value_str:
                # Remove brackets for non-vector values
                value = value_str.strip('[]')
            else:
                # Keep the value as is for vectors
                value = value_str

            # Store the value and its probability in the dictionary
            parsed_domain[value] = prob

        return parsed_domain
    
    #  The 'check_pathTag_variable_value' function is used to check if the pathTag variable value (i.e., a vector) is subject to the rules of the PMDP
    def check_pathTag_variable_value(self, pathTag_variable_value: str) -> bool: 
        # Remove parentheses and split pathTag_variable_value by commas
        values = pathTag_variable_value.strip('[]').split(',')

        # Convert values to floats and calculate their sum
        values_sum = sum(float(value) for value in values)

        # Check if the sum of values is equal to 1
        if values_sum != 1:
            raise ValueError("pathTag_variable_value should be a vector with elements (each element should be 0 or 1), and the basis of the vector should be 1 (i.e., the sum of all elements should be 1)")
        else:
            return True    
        
    # The 'external_function' function is used to get the value for an uncontrollable variable with an unknown probability distribution
    #  The 'external_function' function is a placeholder for an external function that can be called to get the value for an uncontrollable variable with an unknown probability distribution
    def get_uncontrollable_unknown_var_value(self, var_name: str, current_state_instance: Dict[str, any], uncontrol_var_unknow_distributions:dict, mode = 'any') -> any:
        """
        Get the value for an uncontrollable variable with an unknown probability distribution.
        """
        # the dats structure of variables in U_v (when prob is 'unknown'), the len(domain_list) >= 1 that can be a any value like domain_list = [any value, 'unknown']
        # the prob part set to 'unknown' and the len(domain_list) >= 1 is necessary, but the value part can be any value, and the len(domain_list) >= 1

        # Call an external function to get the value for the variable
        # Anotate the code below to call the external function 
        #result = UVSampler.sample_based_on_datset_and_current_state(self.dataset, var_name, current_state_instance)[var_name]
        
        if mode == 'QWS':
            conVar = var_name.split('_')[0]
            unconVar = var_name.split('_')[1]
            ## The unconVar name is from 1, so we need to minus 1 to get the index of the unconVar in the activity
            unconVar_key = int(re.findall(r'\d+', unconVar)[-1]) - 1
            # var_key is the index of the controllable variable in the activity
            var_key = int(re.findall(r'\d+', conVar)[-1])
            selected_candidate_service = int(current_state_instance[f'c{var_key}'])
            #if var_key == 10:
                #print(f"selected_candidate_service: {selected_candidate_service}\n")
            candidate_services = uncontrol_var_unknow_distributions[var_key]

            """
            Given a candidate result tuple of the form:
                (candidate_index, [ (row index, probability dict), ... ])
            For each row, sample a value using its probability distribution.
            Return a list of sampled values (one per row).
            """
            candidate_distribution = next(item for item in candidate_services if item[0] == selected_candidate_service)
            cand_idx, row_dists = candidate_distribution
            # Create a mapping from row index to its probability dictionary.
            row_dict = {row_idx: prob_dict for row_idx, prob_dict in row_dists}
            samples = {}
            target_rows = [unconVar_key, unconVar_key]
            for row in target_rows:
                if row not in row_dict:
                    samples[row] = None
                else:
                    prob_dict = row_dict[row]
                    # If the probability distribution is empty, use a default value (here 0.0).
                    if not prob_dict:
                        samples[row] = 0.0
                    else:
                        candidates = list(prob_dict.keys())
                        weights = list(prob_dict.values())
                        samples[row] = random.choices(candidates, weights=weights, k=1)[0]
                        if samples[row] == 0.0:
                            samples[row] = 0.1
            result = str(samples[unconVar_key])
            return result
        
        elif mode == 'travel_agency':
            if var_name == 'HC' or var_name == 'HP':
                CI = int(current_state_instance['CI'])
                CO = int(current_state_instance['CO'])
                H = int(current_state_instance['H'])
                if CO > CI and current_state_instance[var_name] == None:
                    data_id = f"H_{CI}_{CO}_{H}"
                    ### sample the price and hc from the distribution
                    price_hc_tuple = sample_from_distribution(data_id, uncontrol_var_unknow_distributions)
                    current_state_instance['HP'] = str(int(price_hc_tuple[0]))
                    current_state_instance['HC'] = str(int(price_hc_tuple[1]))
                    result = current_state_instance[var_name]
                elif CO > CI and current_state_instance[var_name] != None:
                    result = current_state_instance[var_name]
                else:
                    #### unligal state, the CO should be larger than CI
                    if var_name == 'HC':
                        result = '0'
                    elif var_name == 'HP':
                        result = '99999'
            elif var_name == 'FP' or var_name == 'FT':
                D = int(current_state_instance['D'])
                R = int(current_state_instance['R'])
                F = int(current_state_instance['F'])
                if R > D and current_state_instance[var_name] == None:
                    data_id = f"F_{D}_{R}_{F}"
                    ### sample the price and hc from the distribution
                    price_time_tuple = sample_from_distribution(data_id, uncontrol_var_unknow_distributions)
                    current_state_instance['FP'] = str(int(price_time_tuple[0]))
                    current_state_instance['FT'] = str(int(price_time_tuple[1]))
                    result = current_state_instance[var_name]
                elif R > D and current_state_instance[var_name] != None:
                    result = current_state_instance[var_name]
                else:
                    #### unligal state, the R should be larger than D
                    if var_name == 'FT':
                        result = '99999'
                    elif var_name == 'FP':
                        result = '99999'
            elif var_name == 'TP' or var_name == 'RT':
                D = int(current_state_instance['D'])
                R = int(current_state_instance['R'])
                T = int(current_state_instance['T'])
                if R > D and current_state_instance[var_name] == None:
                    data_id = f"T_{D}_{R}_{T}"
                    ### sample the price and hc from the distribution
                    price_time_tuple = sample_from_distribution(data_id, uncontrol_var_unknow_distributions)
                    current_state_instance['TP'] = str(int(price_time_tuple[0]))
                    current_state_instance['RT'] = str(int(price_time_tuple[1]))
                    result = current_state_instance[var_name]
                elif R > D and current_state_instance[var_name] != None:
                    result = current_state_instance[var_name]
                else:
                    #### unligal state, the R should be larger than D
                    if var_name == 'RT':
                        result = '99999'
                    elif var_name == 'TP':
                        result = '99999'
            else:
                raise ValueError(f"The 'Unknown Distribution' of {var_name} is not in the travel agency dataset")
            
            return result
        
        else:

            # Use the debug code below to simulate the external function
            result = np.random.normal(loc=200, scale=100)
            result = round(result) 
            if result < 0:
                result = 0

            result = str(np.clip(int(round(result / 10) * 10), 0, 200))
            if var_name == 'HC':
                result = '4'
              

            return result
                                 
    # The 'parse_nexeStates' function is used to execute the state transition for the PMDP based on the current state, activated vertices, and the parsed action results
    # it will return the next activated vertices, and update the 'current_state' instance

    # the 'updated_vars' is the list of variables that have been updated by the action
    # the 'wait_vertices' is the list of vars that are set to 'wait' by the action, if any one variable be set 'wait', which means all variables in the same vertex will be set to 'wait'
    # the 'finished_vertices' is the list of vertices that have finished executing, that is, the vertex has no controllable variables, and the action is 'finish'
    # The 'parse_nexeStates' function is model-free, so we do not need to calculate all transition probability in all trajectories, we only need to calculate what is the next in the current trajectory
    def parse_nexeStates(self, activated_vertices: Union[str, tuple], updated_vars: List[str], wait_vertices: List[str], finished_vertices: List[str], 
                         uncontrol_var_unknow_distributions, mode, parallel_asynchronous_prob = 1, frequency_prob_dict = {}, MDP_example_mode = '') -> list[str]:
        """
        Executes the state transition for the PMDP based on the current state, activated vertices, and the parsed action results.

        :param current_state: The current state of the PMDP as a dictionary.
        :param activated_vertices: The list of currently activated vertices.
        :param updated_vars: The list of variables that have been updated by the action.
        :param wait_vertices: The list of vertices that are set to 'wait' by the action.
        :param finished_vertices: The list of vertices that have finished executing.
        :param graph: The PMDP graph containing vertices and edges.

        :return: the next state.

        """
        # the 'activated_vertices' is the current time-step v, which can be a single activated vertex or a concurrent block via a series of nested tuples
        activated_vertices_list = []
        activated_vertices_list = self.convert_horizon_item_to_list(activated_vertices)
      

        # uncontrollable_vars data structure is list of Variable class instances, each variable only belongs to one vertex
        # each vetex in 'activated_vertices_list' belong to a concurrent thread
        uncontrollable_vars = self.get_uncontrollable_vars(activated_vertices_list)
        Var_with_distrubution = {}
        next_activated_vertices_list = []
        #transition_probability = 1.0  # Initialize the transition probability
        
        # Parse the values and domains of uncontrollable variables
        for var in uncontrollable_vars:
            # The dict is like {'var_name': {'value1': prob1, 'value2': prob2, ...}}
            # prob1, prob2, ... sum up to 1 (if these probs is not 'unknown'), which has been checked in the 'Convert_BPMN_into_Graph.py'
            Var_with_distrubution.update({var.name: self.parse_uncontrollable_variable_domain(var)})

            # next_state[var] = parsed_values


        # Handle state transitions (the new state not been handled in here, rather the end of 'parse_nexeStates' function) based on the type of the vertex, to get the next activated vertices
        for vertex_id in activated_vertices_list:     
            vertex = self.graph.vertices[vertex_id]
            
            if vertex.T_v == 'seg' and (vertex_id not in wait_vertices):
                # i.e., the fomula (6) in paper, but especially for the splitting exclusive gateways 
                # Handle splitting exclusive gateways
                outgoing_edges = [edge for edge in self.graph.edges if edge.source == vertex_id]
                # get the name of pathTag, the 'pathTag_variabl' will be like 'pathTag_variable_name=pathTag_variable_value'
                # The 'pathTag_variable_value' is the vector, for example, if path has two, the pathTag_variable_value will be like '[0,1]' or '[1,0]'
                pathTag_variable_name = outgoing_edges[0].pathTag.split('=')[0]

                # Check if the pathTag variable value is subject to the rules of the PMDP
                # If pass, transition to the next_activated_vertices, that current edge.target 
                if pathTag_variable_name in self.current_state:
                # pathTag_variable_name must be defined in the PMDP instance as a variable
                    if self.current_state[pathTag_variable_name] != None:
                    # the pathTag variable is controllable or uncontrollable, but its value has been set in before or current action, so we can determine the next activated vertices in the current state instance
                        for edge in outgoing_edges:
                            # the pathTag value is used to decide which path is selected
                            pathTag_variable_name, pathTag_variable_value = edge.pathTag.split('=')
                            # the paths from 'seg' will be only one path in each trajectory, so we can break the loop after we get which path is selected 
                            if self.current_state[pathTag_variable_name] == pathTag_variable_value and self.check_pathTag_variable_value(pathTag_variable_value):
                                # corresponding to set the edge probability of ep_ij as '1' in current trajectory, the ep_ij is the formula (6) in paper
                                # the definition of ep_ij in the formula (6) is the model-based transition probability
                                # the code is model-free, so we do not need to calculate all transition probability in all trajectories, we only need to calculate the transition probability in the current trajectory
                                next_activated_vertices_list.append(edge.target)
                                break
                    elif any(var.name == pathTag_variable_name for var in vertex.U_v):
                        # Handle uncontrollable variables (only the pathTag variable) that are defined in currrent time-step, i.e., the uncontrollable variable of vertex_id (i.e., the splitting exclusive gateway)
                        # we need to sample a value for the pathTag variable
                        for var in vertex.U_v:
                            if var.name == pathTag_variable_name:
                                if var.name in Var_with_distrubution:
                                    # Get the probability distribution for the variable, that will a dict like {'value1': prob1, 'value2': prob2, ...}
                                    # The prob1, prob2, ... sum up to 1 (if these probs is not 'unknown'), which has been checked in the 'Convert_BPMN_into_Graph.py'
                                    distribution = Var_with_distrubution[var.name]

                                    # Check if the probability distribution is 'unknown'
                                    if 'unknown' in distribution.values():
                                        # The code version of 19-11-2024 
                                        # Never set pathTag variable to 'unknown' in the BPMN model, so this code will never be executed
                                        # Call an external function to get the value for the variable
                                        sampled_value = self.get_uncontrollable_unknown_var_value(var.name, self.current_state, uncontrol_var_unknow_distributions, mode)
                                    else:
                                        # Sample a value based on the probability distribution
                                        # The np.random.choice function is used to sample a value from the distribution
                                        sampled_value = np.random.choice(list(distribution.keys()), p=list(distribution.values()))
                                    # Update the sampled pathTag value in the current state instance
                                    self.current_state[var.name] = sampled_value
                                    self.context[var.name] = self.evaluate_expression(sampled_value)
                                    # Update the transition probability based on the sampled value
                                    ###### transition_probability *= distribution[sampled_value]
                                    next_activated_vertices_list.append(edge.target)
                                else:
                                    raise ValueError(f"The distribution of variable {var.name} is not defined in the PMDP instance")
                                # the paths from 'seg' will be only one path in each trajectory, so we can break the loop after we get which path is selected 
                                break 
                    else:
                        raise ValueError(f"The splitting exclusive gateway {vertex_id} need a pathTag variable {pathTag_variable_name} that is defined after {vertex_id}, please revise it to be defined before or at {vertex_id}")
                else:
                    raise  ValueError(f"The splitting exclusive gateway {vertex_id} need a pathTag variable {pathTag_variable_name} that is not defined in the PMDP instance, please check the BPMN model")  
            
            # if the vertex need to wait, the current active vertex will be next active vertex, which will be handle in the last situation of this IF-ELSE statement
            elif vertex.T_v == 'spg' and (vertex_id not in wait_vertices):
                # i.e., the fomula (7) in paper
                # Handle splitting parallel gateways
                outgoing_edges = [edge for edge in self.graph.edges if edge.source == vertex_id]
                for edge in outgoing_edges:
                    next_activated_vertices_list.append(edge.target)
            
            # checking whether the target vertex of current vertex is a 'mpg' vertex
            # bacause each 'mpg' incoming must come from a 'empty...' vertex
            elif 'empty' in vertex_id and vertex_id in finished_vertices:
                
                # the len(target_edge_list) must be 1, because the 'empty....' vertex will be only one target vertex 'mpg'
                target_edge_list = [edge for edge in self.graph.edges if edge.source == vertex_id]
                # Handle synchronization points (merge points) for parallel gateways
                # get all 'edges' has the same target vertex for current vertex_id
                edges = [edge for edge in self.graph.edges if edge.target == target_edge_list[0].target]
                # because the lasts vertex of each thread will be 'empty....' vertex without any variables, so it must be set finished if this vertex is activated and all threads are finished
                # synchronize the threads, so the next active vertices will be the target vertex of the current vertex_id
                if all(edge.source in finished_vertices for edge in edges):
                    # when all threads are finished, the target vertex of the current vertex_id will be activated
                    next_activated_vertices_list.append(target_edge_list[0].target)
                    # in case, list remove duplicate elements
                    # because each 'empty....' will be traversed, so the target vertex 'mpg' will be added to the next_activated_vertices_list multiple times, so we need to remove the duplicate elements
                    next_activated_vertices_list = list(set(next_activated_vertices_list))                    
                else:
                    # even the vertex action is 'finished', the 'finished' will be seen as 'wait'
                    # 'Self Transition' for the 'empty....' vertex, wait synnchronization, so the current active vertex will be next active vertex
                    next_activated_vertices_list.append(vertex_id)
            
            else:
                # i.e., the fomula (6) in paper, but especially for the ordinary vertex transitions (i.e., not the splitting exclusive gateways)
                # Handle ordinary vertex transitions, or self transitions with 'parallel_asynchronous_prob' for the vertices that need to asynchronization or 'wait(in old version)'
                # the len(outgoing_edges) must be 1, because the splitting exclusive gateways and the splitting parallel gateways have been handled in the above
                outgoing_edges = [edge for edge in self.graph.edges if edge.source == vertex_id]
                for edge in outgoing_edges:
                    # actually, only two situations, all variables in the vertex are set to 'wait' or not any variable in the vertex is set to 'wait'
                    if vertex_id in wait_vertices:
                        # the vertex need to wait, so the current active vertex will be next active
                        # i.e., Self Transition for the vertex
                        next_activated_vertices_list.append(edge.source)
                    else:
                        if len(self.graph.vertices[vertex_id].H_v) > 1:
                            # 'len(self.graph.vertices[vertex_id].H_v) > 1' means this vertex belongs to a concurrent block
                            # parallel_asynchronous_prob '1' means don't self transition, so the vertex will be executed
                            if random.random() < parallel_asynchronous_prob and parallel_asynchronous_prob != 1:
                                # 5% of the vertex will be not executed when the 'action' is selected, for make some states in PMDP about the parallel blocks are asynchronous
                                # Handle asynchronous transitions situations for parallel blocks, that is, model some states in PMDP about the parallel blocks are asynchronous
                                # We set default value of 'parallel_asynchronous_prob' to 0.05, which means 5% of the time the transitions in parallel blocks are asynchronous
                                next_activated_vertices_list.append(edge.source)
                                # Reset the variables in the vertex to 'None' for the next time-step in PMDP
                                for var in self.graph.vertices[vertex_id].C_v + self.graph.vertices[vertex_id].U_v:
                                    # var.name is the name of controllable variable
                                    self.current_state[var.name] = None
                                    self.context[var.name] = None
                            else:
                                # 1 - parallel_asynchronous_prob% (if parallel_asynchronous_prob=1, then 100%) of the vertex will be executed when the 'action' is selected
                                next_activated_vertices_list.append(edge.target)
                        else:
                        # not need to wait, so the vertex will be executed, and the next vertex will be activated
                        # Here also handle those vertices that have no controllable variables (that is, '..._Horizon_Action'), even we have not set the 'finish' action for the vertex
                            next_activated_vertices_list.append(edge.target)
            
        # Handle the new state changed by environment, that is, uncontrollable variables related to the activated_vertices_list, (not next_activated_vertices_list)
        for vertex_id in activated_vertices_list:
            vertex = self.graph.vertices[vertex_id]
            for var in vertex.U_v:
                # Check if the var has uncontrollable variables, and is not excuted wait action (i.e., the vertex is still activated in next time-step)
                if var.name in Var_with_distrubution and vertex_id not in next_activated_vertices_list:
                    distribution = Var_with_distrubution[var.name]
                    if 'unknown' in distribution.values():
                        if 'CSSC_MDP' == MDP_example_mode:
                            if 'rt' in var.name:
                                concrete_service_index = '~' + f"{self.current_state['C' + var.name.replace('rt', '')]}"
                                uncontrollable_var_index = 'U' + var.name.replace('rt', '')
                            if 'tp' in var.name:
                                concrete_service_index = '~' + f"{self.current_state['C' + var.name.replace('tp', '')]}"
                                uncontrollable_var_index = 'U' + var.name.replace('tp', '')
                            #concrete_service_index = '~' + f"{self.current_state['C' + var.name.replace('U', '')]}"
                            ####### only for wsdreamDataset1 for CSSC-MDP #######
                            probabilities, unique_values = frequency_prob_dict[uncontrollable_var_index][concrete_service_index]
                            # Default 'sampled_value' is string and will be converted later, so here for compatibility, we set the 'sampled_value' to string
                            sampled_value = f'{probability_sampling(probabilities, unique_values, sample_size=1, replace=True)[0]}'
                            #sampled_value = probability_sampling(probabilities, unique_values, sample_size=1, replace=True)[0]

                            # Remove the outer square brackets
                            trimmed_str = sampled_value.strip('[]')

                            # Splitting a string
                            rt_value, tp_value = trimmed_str.split(',')
                            tp_value = tp_value.strip()

                            if 'rt' in var.name:
                                var_index = var.name.replace('rt', '')

                                self.current_state[var.name] = rt_value
                                self.context[var.name] = self.evaluate_expression(rt_value)

                                self.current_state[f'tp{var_index}'] = tp_value
                                self.context[f'tp{var_index}'] = self.evaluate_expression(tp_value)
                            if 'tp' in var.name:
                                var_index = var.name.replace('tp', '')

                                self.current_state[var.name] = tp_value
                                self.context[var.name] = self.evaluate_expression(tp_value)

                                self.current_state[f'rt{var_index}'] = rt_value
                                self.context[f'rt{var_index}'] = self.evaluate_expression(rt_value)
                            break
                        else:
                            # Call an external function to get the value for the variable, one by one
                            sampled_value = self.get_uncontrollable_unknown_var_value(var.name, self.current_state, uncontrol_var_unknow_distributions, mode)

                            self.current_state[var.name] = sampled_value
                            self.context[var.name] = self.evaluate_expression(sampled_value)
                    else:
                        # a explicit distribution follows the format described in paper about uncertainty, so we can use the np.random.choice function to sample a value from the distribution
                        # the distribution is assumed discrete probability distribution (a explicit) in here, so we can use the np.random.choice function to sample a value from the distribution
                        sampled_value = np.random.choice(list(distribution.keys()), p=list(distribution.values()))

                        self.current_state[var.name] = sampled_value
                        self.context[var.name] = self.evaluate_expression(sampled_value)
                    #####transition_probability *= distribution[sampled_value]

        for vertex_id in next_activated_vertices_list:
            vertex = self.graph.vertices[vertex_id]
            # if in this next time-step we transite to a 'seg' vertex, we need to set the all variables (incluing controllable and uncontrollable) in those pathes that are not selected to '0' for 'KPIs calculation'
            if vertex.T_v == 'meg' and vertex_id not in activated_vertices_list:
                # observed_set is a list of list, each list is a path, and each path is a list of variables 
                for pathVars in vertex.observed_set:
                    for var in pathVars:
                        # all vars in pathVars is defined be before the 'meg' vertex, and will have a value in the current state instance if the path is selected
                        if self.current_state[var.name] is None:
                            # if the path is not selected, the value of the var (in not selected path) will still be 'None' in the current state instance
                            # set the value of the var to '0', for P-MDP with pathAware when we calculate the KPIs
                            self.current_state[var.name] = 'NotSelected'
                            self.context[var.name] = 'NotSelected'
                            

        # We get the next activated vertices as a list, but we need convertd it to the type compatible with the horizon space, that is str, a tuple for a concurrent block, or nested tuple for the nested concurrent blocks
        result = self.find_in_horizon_space_dict(self.horizon_space_dict, next_activated_vertices_list)
        return result


    ##### The Reward Function #####

    # The 'build_pmdp_context' function is used to build (initialize) the context for the PMDP, including the state instance and KPIs
    def build_pmdp_context(self) -> dict:
        """
        Builds the context for the PMDP, including the state instance and KPIs.

        :return: The context dictionary.
        """
        # get a copy of the current state instance, which involve all variables state in PMDP
        context = self.current_state.copy()
        for kpi in self.graph.KPIs:
            context[kpi.name] = None

        # Add the 'NotSelected' value to the context for the KPIs calculation with pathAware
        context['NotSelected'] = 0
        return context

    # The 'retrieve_VarsOrKPIs_from_expression' function is used to retrieve the variables or KPIs from an expression
    def retrieve_VarsOrKPIs_from_expression(self, expression: str) -> list:
        # The OPERATOR_INFO is a dictionary, which contains the operators and their corresponding information
        operator_keys = OPERATOR_INFO.keys()

        ### Due to the operator '\sqrt' in the expression, is like '\sqrt{...} or \sqrt[...]{...}', 
        # we need to remove the '{' and '}' or '[' and ']' in the expression, those two symbols '{','}' and '(',')' are not the operators and not the variables or KPIs

        sqrt_tag = False
        if r'\sqrt' in expression:
            sqrt_tag = True

        # Replace the operators with spaces, in case non-operators has the same part of name as operators
        for key in operator_keys:
            expression = expression.replace(key, " ")

        expression_list = expression.split()


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
        #for key in self.context.keys():
            #if key in expression_list:
                #VoK.append(key)


        for key in self.context.keys():
            if key in expression_list:
                if key in self.current_state: 
                    ### This key is a variable in a activity              
                    VoK.append(key)
                else:
                    # the key is a KPI expression, we need to retrieve the variables from the expression
                    for kpi in self.graph.KPIs:
                        if kpi.name == key:
                            kpi_expression = kpi.expression
                            VoK.extend(self.retrieve_VarsOrKPIs_from_expression(kpi_expression))

                            ### 2025-03-09 ###
                            #### Some kpis 'k1' is not defined in the activities, those could be used by other kpis 'k2', and 'k3' use 'k2'
                            #### so we all variables and KPIs in the expression of kpi to the VoK list
                            
                            VoK.append(key)
                            #### 2025-03-09 ####

                            break


        return VoK

    # The constraint only all variables in the expression are assigned, the constraint will be set to 'int' stage
    # 'int' stage means the constraint is ready to be assessed (i.e., ready to calculate immediate reward) in the PMDP
    def update_constraints_int_stage(self, constraints: List[Constraint])-> List[Constraint]:
        for constraint in constraints:
            # the all constraint stage will be 'ini' at the beginning of the PMDP
            if constraint.stage == 'ini':
                variables = self.retrieve_VarsOrKPIs_from_expression(constraint.expression)
                # The 'variables' could consist of var and KPIs
                variables_assigned = [var for var in variables if self.context[var] != None]
                # Check if all variables in the constraint are assigned
                if len(variables_assigned) == len(variables):
                    constraint.stage = 'int'
        return constraints
    
    def update_kpis_int_stage(self, kpis: List[KPI]) -> List[KPI]:
        for kpi in kpis:
            # the all kpi stage will be 'ini' at the beginning of the PMDP
            if kpi.stage == 'ini':
                if 'kpt4' in kpi.name and kpi.stage == 'int':
                    pass

                variables = self.retrieve_VarsOrKPIs_from_expression(kpi.expression)
                # The 'variables' could consist of var and other KPIs
                variables_assigned = [var for var in variables if self.context[var] != None]
                # Check if all variables in the constraint are assigned
                if len(variables_assigned) == len(variables):

                    kpi.stage = 'int'

                    
                    #### 2025-03-09 ####
                    #### Some kpis 'k1' is not defined in the activities, those could be used by other kpis 'k2', and 'k3' use 'k2'
                    #### This mean k3 Indirectly using the 'k1', but we can not makesure the 'k1' is evaluated before 'k3'
                    #### So we need to evaluate this kpi in right now
                    if f'{kpi.name}_direction' not in self.context:
                        
                        #### 2025-03-09 #### 
                        variables = self.retrieve_VarsOrKPIs_from_expression(kpi.expression)
                        # The 'variables' could consist of var and other KPIs
                        # or some KPIs just got 'int' but have not been evaluated so far, such as kpi4 = kpi3 + kpi2, and kpi4 and kpi3 just got 'int', 
                        # and in kpis list, kpi4 is before kpi3, so kpi4 will be evaluated first but kpi3 has not been evaluated no value, so kpi3 will be evaluated in the next time-step
                        variables_assigned = [var for var in variables if self.context[var] != None]
                        # Check if all variables in the constraint are assigned
                        if len(variables_assigned) == len(variables):
                        
                            kpi_value, self.dynamic_bounds = kpi.evaluate(self.context, self.dynamic_bounds)
                            self.context[kpi.name] = kpi_value
                    ### 2025-03-09 ####
                    
                    
        return kpis

    # the 'fin' means the constraint and KPI has been assessed, and the constraint and KPI will be set to 'fin' stage
    # Return the updated constraints and KPIs
    def update_constraints_KPIs_fin_stage(self, constraints: List[Constraint], kpis: List[KPI]) -> tuple[List[Constraint], List[KPI],]:
        """
        Sets the given 'int' constraints to the finished stage ('fin').

        :param constraints: The list of constraints and KPIs.

        :return: The updated list of constraints.
        """
        for constraint in constraints:
            if constraint.stage == 'int':
                constraint.stage = 'fin'

        for kpi in kpis:
            if kpi.stage == 'int':
                kpi.stage = 'fin'

        return constraints, kpis
    
    # Return the updated constraints
    # Some kpis in context will be got the value from the 'evaluate' function
    def evaluate_kpis(self, kpis: List[KPI], first = True) -> List[KPI]:
        # update some kpis in 'int' stage
        kpis = self.update_kpis_int_stage(kpis)

        for kpi in kpis:
            if kpi.stage == 'int':

                ######only has those two old code before 2025-03-09 ######
                kpi_value, self.dynamic_bounds = kpi.evaluate(self.context, self.dynamic_bounds)
                self.context[kpi.name] = kpi_value

        # update some kpis to 'fin' stage
        kpis = self.update_constraints_KPIs_fin_stage([], kpis)[1]

        if first:
            # Because some KPIs involve variables that are defined at the current time step, they need to be evaluated again after the first KPIs evaluation (some KPIs may not have been set to the 'int' stage and thus not evaluated).
            kpis = self.evaluate_kpis(kpis, False)

        return kpis
    
    # The 'evaluate_constraints' function is used to evaluate the constraints in the PMDP
    # Will return the current_rewards (immediate) reward for the constraints, the updated constraints in current time-step
    def evaluate_constraints(self, constraints: List[Constraint], cons_type: str , ConstraintState, punish_mode = False, CET_mode = False, evaluated_constraintsName_list = [], current_rewards = 0, first = True) -> tuple[float, List[Constraint]]:
        
        # update some constraints in 'int' stage
        constraints = self.update_constraints_int_stage(constraints)

        for constraint in constraints:
            if constraint.stage == 'int':
                #evaluated_constraintsName_list.append(constraint.name)
                
                # calculat the HC rewards
                # The HC constraint return 'True' or 'False' as result
                if cons_type == 'HC':
                    constraint_result, self.dynamic_bounds = constraint.evaluate(self.context, self.dynamic_bounds, 'HC')
                    if constraint_result == 1 or constraint_result == True:
                        current_rewards += constraint.weight


                        if CET_mode and constraint.weight != 0:
                            # for update the constraint state for CET
                            ConstraintState.constraint_values[constraint.name] = 1
                            evaluated_constraintsName_list.append(constraint.name)
                    else:
                        ## try punishment for hard constraints ##
                        if punish_mode and constraint.weight != 0:
                            current_rewards -= 1
                        ## End try punishment for hard constraints ##
                        if CET_mode and constraint.weight != 0:
                            ConstraintState.constraint_values[constraint.name] = 0
                            #evaluated_constraintsName_list.append(constraint.name)
                        # No punishment
                        pass
                        
                        # have punishment
                        #current_rewards -= constraint.weight
                # calculate the SC rewards
                else:
                    constraint_result, self.dynamic_bounds = constraint.evaluate(self.context, self.dynamic_bounds, 'SC')

                    current_rewards += constraint.weight * constraint_result


                    if 'TP' in constraint.expression:
                        pass

                    if 'TP' in constraint.expression and constraint_result == 1:
                        pass


                    if punish_mode and constraint.weight != 0:
                        #if constraint_result == False and '\max' not in constraint.expression and '\min' not in constraint.expression:
                            #pass
                        if constraint_result == 0 and '\max' not in constraint.expression and '\min' not in constraint.expression:
                            current_rewards -= constraint.weight
                            
                    if CET_mode and constraint.weight != 0:
                        # for update the constraint state for CET
                        if constraint_result == 1 or constraint_result == True:
                            ConstraintState.constraint_values[constraint.name] = 1
                            evaluated_constraintsName_list.append(constraint.name)
                        else:
                            ConstraintState.constraint_values[constraint.name] = round(constraint_result, 3)
                            if round(constraint_result, 3) >= 0.0:
                                evaluated_constraintsName_list.append(constraint.name)
                            else:
                                ConstraintState.constraint_values[constraint.name] = 0
                                #evaluated_constraintsName_list.append(constraint.name)

        evaluated_constraintsName_list = list(set(evaluated_constraintsName_list))

        # update some constraints to 'fin' stage
        constraints = self.update_constraints_KPIs_fin_stage(constraints, [])[0]

        if first:
            # Because some constraints involve variables or KPIs that are defined at the current time step, they need to be evaluated again after the first constraints evaluation (some constraints may not have been set to the 'int' stage and thus not evaluated).
            # Note that the reward value from the first evaluation should be used as the initial reward value for the second evaluation.
            current_rewards, constraints, ConstraintState, evaluated_constraintsName_list = self.evaluate_constraints(constraints, cons_type, ConstraintState, punish_mode, CET_mode, evaluated_constraintsName_list, current_rewards, False)

        # the self.context will be not changed in the 'evaluate_constraints' function
        return current_rewards, constraints, ConstraintState, evaluated_constraintsName_list


    def calculate_reward(self, w_hc: float, w_sc: float, HC_list, SC_list, ConstraintState_list, punish_mode, CET_mode, accumulate_rewards_list, subStateSpace_number = 2) -> list[float]:
        """
        calculate_reward Calculates reward based on the current state instance
        When used in PMDP, the latest 'state instance' obtained after executing the action (i.e., controllable factors) and the state transition (i.e., uncontrollable factors) at each time-step should be called to calculate the reward

        w_hc: The weight of the hard constraint, the default is 0.5.
        w_sc: The weight of the soft constraint, the default is 0.5.
        """

        self.graph.KPIs = self.evaluate_kpis(self.graph.KPIs)
        
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

            sub_HC_Rewards, temp_HC, ConstraintState_list[i], HC_evaluated_constraintsName_list[i] = self.evaluate_constraints(HC_list[i], 'HC', ConstraintState_list[i], punish_mode, CET_mode, [])
            sub_SC_Rewards, temp_SC, ConstraintState_list[i], SC_evaluated_constraintsName_list[i] = self.evaluate_constraints(SC_list[i], 'SC', ConstraintState_list[i], punish_mode, CET_mode, []) 
            evaluated_constraintsName_list[i] = HC_evaluated_constraintsName_list[i] + SC_evaluated_constraintsName_list[i]

            # remove the duplicate elements in the 'evaluated_constraintsName_list'
            evaluated_constraintsName_list[i] = list(set(evaluated_constraintsName_list[i]))

            Rewards_HC_list[i] = sub_HC_Rewards * w_hc
            Rewards_SC_list[i] = sub_SC_Rewards * w_sc
            Rewards_HC_plus_SC_list[i] = Rewards_HC_list[i] + Rewards_SC_list[i]
            All_Rewards_All_subStates += Rewards_HC_plus_SC_list[i]
            updated_stage_HC.extend(temp_HC)
            updated_stage_SC.extend(temp_SC)

            accumulate_rewards_list[i][0] += Rewards_HC_plus_SC_list[i]
            accumulate_rewards_list[i][1] += Rewards_HC_list[i]
            accumulate_rewards_list[i][2] += Rewards_SC_list[i]
            
        self.graph.HC = updated_stage_HC
        self.graph.SC = updated_stage_SC

        
        # The 'Rewards_HC_plus_SC_list' consist of two parts, the first part is the HC rewards, and the second part is the SC rewards, each part is weight i
        return [Rewards_HC_plus_SC_list, Rewards_HC_list, Rewards_SC_list, All_Rewards_All_subStates], accumulate_rewards_list, ConstraintState_list, evaluated_constraintsName_list

        
    def reset_KPIs_Constraints_to_ini_stages(self):
        # Iterate through all KPI objects and reset their stage
        for kpi in self.graph.KPIs:
            kpi.stage = 'ini'

        # Iterate through all Constraint objects and reset their stage
        for hc in self.graph.HC:
            hc.stage = 'ini'
        
        for sc in self.graph.SC:
            sc.stage = 'ini'

    # Used to convert characters into a computable format
    def evaluate_expression(self, value):
        try:
            str_value = str(value)
            # constant
            if re.match(r'\d+(\.\d+)?', str_value):
                return eval(str_value)

            # If the AST is originally a vector, try to convert the string to a list if the string is a vector
            if value.startswith('[') and value.endswith(']'):
                new_value = ''
                vars = value.replace('[', '').replace(']', '').split(',')
                new_value += '['
                for var in vars:
                    var = var.strip()
                    if re.match(r'\d+(\.\d+)?', var):
                        new_value += f"{var},"
                    elif var in self.context:
                        actual_value = self.context[var]
                        new_value += f"{actual_value},"
                    elif any([op in var for op in OPERATOR_INFO.keys()]):
                        # if the var (i.e., KPI sub-expression) is an expression, evaluate it via recursion
                        subKPI_ast =  self.parser.parse(var)
                        actual_value, self.dynamic_bounds = self.sub_kpi_evaluator.evaluate(subKPI_ast, self.dynamic_bounds)
                        new_value += f"{actual_value},"
                    else:
                        raise ValueError(f"The: {value}" + " is not defined in the context")
                new_value = new_value.rstrip(',')  # Remove the trailing comma
                new_value += ']'
                return eval(new_value)

            # If the AST is neither a number nor a list (i.e., the AST is a variable name), get the corresponding value of the AST from the context (self.context).
            # self.context is a dictionary that stores the values of variables
            # If it is a variable in the context, call its own evaluate_expression function to solve it recursively
            value = self.evaluate_expression(self.context[str_value])

            # Check if the input in the context is a vector, that is, check whether the value obtained from the context is a string and is a string surrounded by square brackets (list)
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                return eval(str_value)
            
            return value

        # If the AST is neither a number nor a list, and the corresponding value of the AST cannot be found in the context, return the AST itself
        except (ValueError, KeyError):
            return str_value

    ### Debugging functions ###
    def select_random_row(self, tensor_action):
        """
        Randomly select a row from a 2D tensor.

        Parameters:
        tensor_2d (torch.Tensor): Input 2D tensor.

        Returns:
        torch.Tensor: Randomly selected row.
        """
        if tensor_action.dim() == 2:
            # Get the number of rows in the tensor
            num_rows = tensor_action.size(0)

            # Generate a random row index
            random_row_index = torch.randint(0, num_rows, (1,)).item()

            # Select the random row
            random_row = tensor_action[random_row_index]

            return random_row.squeeze()
        else:
            return tensor_action
        
if __name__ == "__main__":


    ### Build the 'Uncertainty Graph' from BPMN with User-defined NFPs ###
    #### Or load a exsting 'Uncertainty Graph' from file. ###
    path_to_user_defiend_bp = 'BPMN_Models/TarvelAgency/Travel_Agency_BP_Userdefined_NFPs.bpmn'
    uncertrainty_graph = parse_bpmn(path_to_user_defiend_bp)

    
    ### Here to build the PMDP environment from the uncertainty graph ###
    pmdp_env = PMDPEnvironment(uncertrainty_graph)
    pmdp_env.debug_mode = False
    horizon_space = pmdp_env.get_horizon_space()
    pass

