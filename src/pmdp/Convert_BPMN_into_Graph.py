#### A transformation of BPMN with Unser-defiend NFPs to Graph for P-MDP ####
####Update this py 'name' , like 'Uncertainty_graph.py'####

import xml.etree.ElementTree as xmlET
import re
import itertools
import networkx as nx
# Get the project root directory
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)


from Datasets.wsdream.Read_Dataset import compute_frequency_probabilities_domain
from src.pmdp.KPIs_Constraints_Operators_Parse_Evaluation import ExpressionParser, ExpressionEvaluator, OPERATOR_INFO

# Define global dynamic_bounds for all variables
dynamic_bounds = {}

class Variable:
    def __init__(self, name, domain, controlType, direction='positive'):
        self.name = name
        self.domain = domain
        self.direction = direction  # 'positive' for larger is better, 'negative' for smaller is better
        #controlType: 1 for controllable, 2 for uncertain
        self.controlType = controlType
        # The vertex_id is used to identify the vertex to which the variable belongs, but at innitialization, it is set to None
        # it will be set actual 'vertex_id' when we need it, such as in Q-learning, the vertex_id is used to identify the vertex to which the variable belongs,
        self.vertex_id = None
        
    def __repr__(self):
        return f"Variable({self.name}, {self.domain}, {self.vertex_id}, {self.controlType}, {self.direction})"
    
    
class Vertex:
    def __init__(self, vertex_id, T_v, C_v=None, U_v=None, elem_name='', H_v=None):
        self.vertex_id = vertex_id
        self.T_v = T_v  # Vertex type
        self.C_v = C_v if C_v else []  # Controllable variables
        self.U_v = U_v if U_v else []  # Uncertain variables
        self.H_v = H_v if H_v else []  # Vertex hierarchy
        self.observed_set = [set()]  # Observable set
        self.elem_name = elem_name
    
    def __repr__(self):
        return f"Vertex({self.vertex_id}, {self.elem_name}, {self.T_v}, {self.C_v}, {self.U_v}, {self.H_v}, {self.observed_set})"

class Edge:
    def __init__(self, source, target, pathTag=None, ep=1.0,):
        self.source = source
        self.target = target
        self.ep = ep  # Edge probability
        # For the edge from seg to vertex, the pathTag is used to identify the path of the edge, which is used to get the edge probability in the P-MDP section.
        self.pathTag = pathTag  # except for the edge from seg to vertex, it will be default as None

    def __repr__(self):
        return f"Edge({self.source} -> {self.target}, {self.ep})"

class KPI:
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression
        self.stage = 'ini'  

    def evaluate(self, context: dict, dynamic_bounds: dict):

        parser = ExpressionParser(OPERATOR_INFO)
        evaluator = ExpressionEvaluator(context)
        ast = parser.parse(self.expression, category='KPI')
        return evaluator.evaluate(ast, dynamic_bounds)
    
    def __repr__(self):
        return f"KPI({self.name}, {self.expression})"



class Constraint:
    def __init__(self, name, expression, weight):
        self.name = name
        self.expression = expression
        self.weight = weight  
        self.stage = 'ini'  

    def evaluate(self, context: dict, dynamic_bounds: dict, c_category):

        parser = ExpressionParser(OPERATOR_INFO)
        evaluator = ExpressionEvaluator(context)
        ast = parser.parse(self.expression, category=c_category)
        return evaluator.evaluate(ast, dynamic_bounds)
    
    
    def __repr__(self):
        return f"Constraint({self.name}, {self.expression}, state={self.stage}, weight={self.weight})"

    

class HardConstraint(Constraint):
    def __init__(self, name, expression, weight):
        super().__init__(name, expression, weight)

class SoftConstraint(Constraint):
    def __init__(self, name, expression, weight):
        super().__init__(name, expression, weight)



class WeightedDirectedGraph:
    def __init__(self):
        self.vertices = {}
        self.edges = []
        self.KPIs = []
        self.HC = []
        self.SC = []
        #self.dynamic_bounds = {}  
        self.startVertex_id = ""
        
    def add_vertex(self, vertex_id, T_v, C_v=None, U_v=None, elem_name='', H_v=None):
        if C_v == [] and not vertex_id.startswith('empty'):
            # Add a default controllable variable for vertices that do not have any controllable variables
            # This will help to avoid HorionSpace Q become larger
            # We add this, will not reuslt to StateSpace about Q and Q_ become larger, because the '{vertex_id}_Horizon_Action' domain is only one value, and it is not any constraints will use it '{vertex_id}_Horizon_Action'
            # the value in domain '1' is no meaning, just for mean the vertex has been executed, the value can be replaced by any other value, such as '0', '2', '3', etc.
            C_v.append(Variable(f'{vertex_id}_Horizon_Action', ['[1]'], 1, 'positive'))
            
        vertex = Vertex(vertex_id, T_v, C_v, U_v, elem_name, H_v)
        self.vertices[vertex_id] = vertex
        return vertex
    
    def add_edge(self, source, target, pathTag = None, ep = 1.0):
        edge = Edge(source, target, pathTag, ep)
        self.edges.append(edge)
    
    def __repr__(self):
        return f"Graph(vertices={self.vertices}, edges={self.edges}, KPIs={self.KPIs}, HC={self.HC}, SC={self.SC})"

    
    # First set them (i.e., the probability of executing of Exclusive Gateways) all to the default static 1/n
    # Later in the P-MDP section, it also needs to be updated if the exclusion gateway path is not the default 1/n
    def calculate_edge_fromSEG_weights(self):
        # Iterate over all vertices in the graph. self.vertices is a dictionary, the keys are vertex IDs and the values are vertex objects.
        for vertex_id, vertex in self.vertices.items():
            if vertex.T_v == 'seg':
                # Create a list of all edges that have the current seg vertex as their source. 
                # Self.edges is a list of all edges. Filter out all edges by checking if the source attribute of each edge is equal to the ID of the current vertex.
                outgoing_edges = [edge for edge in self.edges if edge.source == vertex_id]
                n = len(outgoing_edges)
                if n > 0:
                    weight = 1.0 / n
                    for edge in outgoing_edges:
                        edge.ep = weight
                        

    # Initial the hierarchy stack (H_v) for each vertex based on the rules specified.
    def initial_vertex_hierarchy(self):

       
        G = nx.DiGraph()
        
        for edge in self.edges:
            source_vertex, target_vertex = edge.source, edge.target
            G.add_edge(source_vertex, target_vertex)
        
        # Get the topological sort of vertices
        try:
            sorted_vertices = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise ValueError("The graph contains a cycle, which is not allowed for topological sorting.")

        # Create a mapping from vertex to its sorted index
        vertex_to_index = {vertex: index for index, vertex in enumerate(sorted_vertices)}
        
        # Get the topological sorting of edges based on the topological sorting of vertices
        sorted_edges = sorted(self.edges, key=lambda edge: (vertex_to_index[edge.source], vertex_to_index[edge.target]))


        spg_thread_counter = {}  # Dictionary to keep track of thread indices for each spg vertex
        
         # Initialize H_v for all 'ste' vertices
        for vertex in self.vertices.values():
            if vertex.T_v == 'ste':
                vertex.H_v.append(('*', 1, 1))

        # Traverse the vertices in topological order
        for edge in sorted_edges:
            source_vertex = self.vertices[edge.source]
            target_vertex = self.vertices[edge.target]
            if not target_vertex.H_v:
                # Perform subsequent operations to set target_vertex.H_v

                # Rule 1: If T_v_i != spg and T_v_j != mpg, copy H_v_i to H_v_j
                if source_vertex.T_v != 'spg' and target_vertex.T_v != 'mpg':
                    target_vertex.H_v = source_vertex.H_v.copy()
                
                # Rule 2: If T_v_i == spg, copy H_v_i to H_v_j, then push (v_i, k, n) to H_v_j
                elif source_vertex.T_v == 'spg':
                    target_vertex.H_v = source_vertex.H_v.copy()
                    if source_vertex.vertex_id not in spg_thread_counter:
                        spg_thread_counter[source_vertex.vertex_id] = 0
                    spg_thread_counter[source_vertex.vertex_id] += 1
                    thread_index = spg_thread_counter[source_vertex.vertex_id]
                    n = len([e for e in self.edges if e.source == source_vertex.vertex_id])
                    target_vertex.H_v.append((source_vertex.vertex_id, thread_index, n))
                
                # Rule 3: If T_v_j == mpg, copy H_v_i to H_v_j, then pop from H_v_j
                elif target_vertex.T_v == 'mpg':
                    target_vertex.H_v = source_vertex.H_v.copy()
                    if target_vertex.H_v:
                        target_vertex.H_v.pop()
    
    # Temporal and Logical Dependencies of Vertices.
    def calculate_observed_sets(self):
        """
        Calculate the observed set for each vertex starting from 'ste' vertices.
        """
        visited = set()
        for vertex_id, vertex in self.vertices.items():
            if vertex.T_v == 'ste':
                self.build_observed_set(vertex_id, visited)

    def build_observed_set(self, vertex_id, visited):
        """
        Build the observed set for a vertex and propagate to its successors.
        """
        if vertex_id in visited:
            return

        vertex = self.vertices[vertex_id]
        vertex.observed_set = self.calculate_observed_set(vertex_id)

        visited.add(vertex_id)

        # Propagate the observed set to successor vertices
        outgoing_edges = [edge for edge in self.edges if edge.source == vertex_id]
        for edge in outgoing_edges:
            target_vertex_id = edge.target
            target_vertex = self.vertices[target_vertex_id]

            if target_vertex.T_v in ['meg', 'mpg']:
                # Check if all predecessors have been visited
                incoming_edges = [e for e in self.edges if e.target == target_vertex_id]
                if all(pred.source in visited for pred in incoming_edges):
                    self.build_observed_set(target_vertex_id, visited)
            else:
                self.build_observed_set(target_vertex_id, visited)

    def calculate_observed_set(self, vertex_id):
        vertex = self.vertices[vertex_id]
        if vertex.T_v == 'ste':
            return [set()]  # Rule 1, initialize with an empty set
        elif vertex.T_v in ['oth', 'seg', 'spg']:
            incoming_edges = [edge for edge in self.edges if edge.target == vertex_id]
            if len(incoming_edges) == 1:
                source_vertex = self.vertices[incoming_edges[0].source]
                if source_vertex.T_v != 'spg':
                    return [sigma.union(set(source_vertex.C_v)).union(set(source_vertex.U_v))
                            for sigma in source_vertex.observed_set]  # Rule 2
                else:
                    return [set()]  # Initialize with an empty set for parallel gateways
            else:
                raise Exception(f"Element {vertex_id} has more than one incoming sequence flow, which is not compliant with BPMN standards.")  
        elif vertex.T_v == 'meg':
            incoming_edges = [edge for edge in self.edges if edge.target == vertex_id]
            from_different_paths_observed_set = []
            for edge in incoming_edges:
                source_vertex = self.vertices[edge.source]
                observed_set_from_source = [sigma.union(set(source_vertex.C_v)).union(set(source_vertex.U_v))
                                            for sigma in source_vertex.observed_set]
                from_different_paths_observed_set.extend(observed_set_from_source)
            return from_different_paths_observed_set  # Rule 4
        elif vertex.T_v == 'mpg':
            incoming_edges = [edge for edge in self.edges if edge.target == vertex_id]
            # The GetTop operation on the $H_v$ of any vertex within the block, in here select the vertex with index '0'
            spg_vertex_id = self.vertices[incoming_edges[0].source].H_v[-1][0]  # GetTop(H_v)
            spg_vertex_observed_set = self.vertices[spg_vertex_id].observed_set
            
            # Collect observed sets from all incoming edges
            incoming_observed_sets = []
            for edge in incoming_edges:
                source_vertex = self.vertices[edge.source]
                observed_set_from_source = [sigma.union(set(source_vertex.C_v)).union(set(source_vertex.U_v))
                                            for sigma in source_vertex.observed_set]
                incoming_observed_sets.append(observed_set_from_source)
            
            # Compute the Cartesian product of all observed sets from different paths
            cartesian_product = list(itertools.product(*incoming_observed_sets))
            combined_observed_set = [set().union(*product) for product in cartesian_product]
            
            # Compute the Cartesian product of spg_vertex_observed_set and combined_observed_set
            final_cartesian_product = list(itertools.product(spg_vertex_observed_set, combined_observed_set))
            final_observed_set = [set().union(*product) for product in final_cartesian_product]
            
            return final_observed_set  # Rule 5
    # delete some unrelated vertices about NR and NFR in G
    def clean_unconnected_vertices(self):
        """
        Remove vertices that are not connected to any edge.
        """
        connected_vertices = set()
        for edge in self.edges:
            connected_vertices.add(edge.source)
            connected_vertices.add(edge.target)
        
        self.vertices = {vertex_id: vertex for vertex_id, vertex in self.vertices.items() if vertex_id in connected_vertices}
        
        
    def parse_graph_KPIsandConstraints(self, root, ns, annotations):
        """
        Parse the KPIs, Hard Constraints, and Soft Constraints for the graph from BPMN data.
        """
        # Traverse all 'bpmn:startEvent' elements
        for start_event in root.findall('.//bpmn:startEvent', ns):
            # Traverse all 'bpmn:dataOutputAssociation' elements within the start event
            for data_output_association in start_event.findall('bpmn:dataOutputAssociation', ns):
                target_ref = data_output_association.find('bpmn:targetRef', ns).text
                # Traverse all 'bpmn:dataObjectReference' elements to find the matching reference
                for data_object in root.findall('.//bpmn:dataObjectReference', ns):
                    if data_object.attrib['id'] == target_ref:
                        # Check the name attribute to identify KPIs, HardConstraints, or SoftConstraints
                        if data_object.attrib['name'] in ['KPIs', 'HardConstraints', 'SoftConstraints']:
                            data_object_id = data_object.attrib['id']
                            # Traverse all 'bpmn:association' elements
                            for association in root.findall('.//bpmn:association', ns):
                                if association.attrib['sourceRef'] == data_object_id:
                                    text_annotation_id = association.attrib['targetRef']
                                    if text_annotation_id in annotations:
                                        # Parse and add KPIs, HardConstraints, or SoftConstraints
                                        if data_object.attrib['name'] == 'KPIs':
                                            _, _, KPIs, _, _= annotations[text_annotation_id]
                                            self.KPIs.extend(KPIs)
                                        elif data_object.attrib['name'] == 'HardConstraints':
                                            _, _, _, HC, _= annotations[text_annotation_id]
                                            self.HC.extend(HC)
                                        elif data_object.attrib['name'] == 'SoftConstraints':
                                            _, _, _, _, SC = annotations[text_annotation_id]
                                            self.SC.extend(SC)


        
def parse_text_annotation(text):
    """
    Parse the text of a BPMN textAnnotation element to extract controllable (C_v) and uncertain (U_v) variables.

    The text could be like "C={c1#positivve:{[A,A2],[B],[C],[D],[E],[F]}}||U={u1#negative:{([1],0.2),([2],0.6),([3],0.1),([4],0.1)}}" the Signature '#' means direction of variable for mm, default will be set positive (if without '#....')
    or like "C={c2:{[1],[2],[3],[4],[5],[6]}}||U={u2:{([0.4,0.6],0.7),([0.5,0.5],0.3)}|u3:{([500],1)}}"
    the special and fixed 'Signatures' are '||','|','=',':','[]','()' and, '{}', where '()' is special for uncertain U_v variables in that it is a probability distribution
    the each value of variables use Signature in two kind of '[value1]' or '[value1,value2,...]', the first mean the variable is a normal variable, the second ([value1,value2,...]) mean the variable is a 'verctor'
    """

    """
    Examples
    text_annotation_text = "C={c1:{[A],[B],[C]}}||U={u1:{([1],0.2),([2],0.6),([3],0.2)}}"
    kpi_annotation_text = "KPIs::kpi1:x+y|kpi2:(z+y-h)/3"
    hc_annotation_text = "HC::hc1#0.5:kpi2<10|hc2#0.5:MAX(z+y)"
    sc_annotation_text = "SC::sc1#0.6:kpi1>5|sc2#0.4:MIN(x+y)"
    """
    C_v = []
    U_v = []
    KPIs = []
    HC = []
    SC = []
    
    
    if "C=" in text and "||U=" in text:
        C_part, U_part = text.split("||")
        C_part = C_part.split("=")[1][1:-1]
        U_part = U_part.split("=")[1][1:-1]
        
        # Parse controllable variables
        for var in C_part.split("|"):
            if ':' in var:
                direction = "positive"
                if "#" in var:
                    name_direction, domain = var.split(":")
                    name, direction = name_direction.split("#")
                else:
                    name, domain = var.split(":")
                domain = domain.strip("{}")
                domain_list = re.findall(r'\[.*?\]', domain)
                C_v.append(Variable(name.strip(), domain_list, 1, direction.strip()))
                
        # Parse uncertain variables
        for var in U_part.split("|"):
            if ':' in var:
                direction = "positive"
                if "#" in var:
                    name_direction, domain = var.split(":")
                    name, direction = name_direction.split("#")
                else:
                    name, domain = var.split(":")
                domain = domain.strip("{}")
                domain_list = []
                # Use regex to extract value-probability pairs
                value_prob_pattern = r"\((\[[^\]]+\]),([^\)]+)\)"
                value_prob_matches = re.findall(value_prob_pattern, domain)
                total_prob = 0.0
                unknown_prob = False
                for value, prob in value_prob_matches:
                    if prob.strip() == 'unknown':
                        '''
                        'unknown' denotes that the probability distribution is unknown and is used for model-free RL scenarios (some variables with unknown probability distribution), 
                        and also for modelling the Candidate Service of a Component Service, 
                        where a Component Service has multiple Candidate Services and each Candidate Service has multiple QoS variables, 
                        and the values of the QoS Variables of each Candidate Service (the probability distribution is the joint probability associated with the specific selected Candidate Service, even other  variables in predecessor vertices
                        and thus is denoted by unknown, and the specific probability distributions (joint) will be learnt and acquired when interacting with the P-MDP and external environment. 
                        Rather than being simply an independent distribution used for independent variables).
                        '''
                        # the dats structure of variables in U_v (when prob is 'unknown'), the len(domain_list) >= 1 that can be a any value like domain_list = [any value, 'unknown']
                        # the prob part set to 'unknown' and the len(domain_list) >= 1 is necessary, but the value part can be any value, and the len(domain_list) >= 1
                        unknown_prob = True
                        domain_list.append([value, prob])
                    else:
                        prob = float(prob)
                        total_prob += prob
                        domain_list.append([value, prob])
                # Check if total probability equals 1
                if not unknown_prob and abs(total_prob - 1.0) > 1e-6:
                    raise ValueError(f"Total probability for variable {name} does not sum to 1: {total_prob}")
                U_v.append(Variable(name.strip(), domain_list, 2, direction.strip()))   
                
    else:
        # Parse KPIs, Hard Constraints, Soft Constraints
            if "KPIs::" in text:
                KPI_part = text.split("KPIs::")[1].strip()
                for kpi in KPI_part.split("|"):
                    name, expr = kpi.split("=")
                    KPIs.append(KPI(name.strip(), expr.strip()))
            elif "HC::" in text:
                HC_part = text.split("HC::")[1].strip()
                if HC_part != "":
                    total_hc_weight = 0.0
                    for hc in HC_part.split("|"):
                        name_weight, expr = hc.split(":")
                        name, weight = name_weight.split("#")
                        weight = float(weight)
                        total_hc_weight += weight
                        HC.append(HardConstraint(name.strip(), expr.strip(), weight))
                    if abs(total_hc_weight - 1.0) > 1e-6:
                        raise ValueError("Constraint weights for HC are incorrect, the sum does not equal 1")
            elif "SC::" in text:
                SC_part = text.split("SC::")[1].strip()
                if SC_part != "":
                    total_sc_weight = 0.0
                    for sc in SC_part.split("|"):
                        name_weight, expr = sc.split(":")
                        name, weight = name_weight.split("#")
                        weight = float(weight)
                        total_sc_weight += weight
                        SC.append(SoftConstraint(name.strip(), expr.strip(), weight))
                    if abs(total_sc_weight - 1.0) > 1e-6:
                        raise ValueError("Constraint weights for SC are incorrect, the sum does not equal 1")

    return C_v, U_v, KPIs, HC, SC
      

def insert_empty_vertex(graph, source, target):
    """
    Insert an empty vertex of type 'oth' between source and target vertices.
    """
    empty_vertex_id = f"empty_{source}_{target}"
    graph.add_vertex(empty_vertex_id, 'oth')
    graph.add_edge(source, empty_vertex_id)
    graph.add_edge(empty_vertex_id, target)
    
    
def parse_bpmn(file_path, n_abstract_number = 30, total_candidates_num = 3, CSSC_MDP_example_mode = False, NMDP_example_mode = False,):

    tree = xmlET.parse(file_path)
    root = tree.getroot()

    # Namespace dictionary to handle XML namespaces in BPMN
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    graph = WeightedDirectedGraph()
    annotations = {}
    
    # Parsing text annotations
    for text_annotation in root.findall('.//bpmn:textAnnotation', ns):
        text = text_annotation.find('bpmn:text', ns).text 
        parsed_data = parse_text_annotation(text)

        if NMDP_example_mode:
            if len(parsed_data[0]) == 1:
                '''
                Norm_state_values = ['[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3]','[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]','[0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3]',
                                    '[0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3]','[0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3]','[0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3]',
                                    '[0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3]','[0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 3]','[0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1]','[0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 2]','[0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3]']
                '''
                Norm_state_values = ['[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]']
                parsed_data[0][0].domain.extend(Norm_state_values)
            elif len(parsed_data[0]) != 0:
                raise ValueError("The number of controllable variables in the NMDP example should be 1.")
            
        annotations[text_annotation.attrib['id']] = parsed_data

    ##### Only is used in CSSC-MDP Example #####
        #- frequency_prob_dict (dict): A dictionary of frequency probabilities for each variable.
    #- activity_variable_domain_string_list (list): Activity variable C and U domain string list
    if CSSC_MDP_example_mode:
        _, activity_variable_doamin_string_list = compute_frequency_probabilities_domain(n_abstract_number, k = total_candidates_num, decimal = 1)
    
    ##### Only is used in CSSC-MDP Example #####

    # Parsing vertices
    for process in root.findall('bpmn:process', ns):
        if CSSC_MDP_example_mode:
            CSSC_index = 0
        for elem in process:
            if 'id' in elem.attrib:
                elem_id = elem.attrib['id']
                try:
                    elem_name = elem.attrib['name']
                except:
                    elem_name = elem_id
                C_v = []
                U_v = []
                # Check if there's an association for the current element
                for association in root.findall('.//bpmn:association', ns):
                    if association.attrib['sourceRef'] == elem_id and "DataObjectReference" not in association.attrib['sourceRef']:
                        target_id = association.attrib['targetRef']
                        if target_id in annotations:
                            # actually here is only got text of C_v and U_v 
                            C_v, U_v, _, _, _= annotations[target_id]
                
                # Only is used in CSSC-MDP Example
                # Get the domains of C_v and U_v from activity_variable_doamin_string_list
                if CSSC_MDP_example_mode and elem.tag == '{http://www.omg.org/spec/BPMN/20100524/MODEL}task' and elem.tag != '{http://www.omg.org/spec/BPMN/20100524/MODEL}startEvent' and elem.tag != '{http://www.omg.org/spec/BPMN/20100524/MODEL}endEvent':
                   
                    C_v, U_v, _, _, _= parse_text_annotation(activity_variable_doamin_string_list[CSSC_index])
            
                    CSSC_index += 1
                            

                if elem.tag == '{http://www.omg.org/spec/BPMN/20100524/MODEL}startEvent':
                    graph.add_vertex(elem_id, 'ste', C_v, U_v, elem_name)
                    # Record the id of the StartEvent of the BPMN to be used as the root of the tree for the HPST in the P-MDP
                    graph.startVertex_id = elem_id
                elif elem.tag == '{http://www.omg.org/spec/BPMN/20100524/MODEL}endEvent':
                    graph.add_vertex(elem_id, 'oth', C_v, U_v, elem_name)
                elif elem.tag == '{http://www.omg.org/spec/BPMN/20100524/MODEL}task':
                    graph.add_vertex(elem_id, 'oth', C_v, U_v, elem_name)
                elif elem.tag == '{http://www.omg.org/spec/BPMN/20100524/MODEL}exclusiveGateway':
                    incoming_flows = elem.findall('bpmn:incoming', ns)
                    outgoing_flows = elem.findall('bpmn:outgoing', ns)
                    if len(incoming_flows) > 1 and len(outgoing_flows) == 1:
                        graph.add_vertex(elem_id, 'meg', C_v, U_v, elem_name)
                    elif len(incoming_flows) == 1 and len(outgoing_flows) > 1:
                        graph.add_vertex(elem_id, 'seg', C_v, U_v, elem_name)
                    elif len(incoming_flows) > 1 and len(outgoing_flows) > 1:
                        raise Exception(f"Gateway modeling exception at {elem_id}")
                elif elem.tag == '{http://www.omg.org/spec/BPMN/20100524/MODEL}parallelGateway':
                    incoming_flows = elem.findall('bpmn:incoming', ns)
                    outgoing_flows = elem.findall('bpmn:outgoing', ns)
                    if len(incoming_flows) > 1 and len(outgoing_flows) == 1:
                        graph.add_vertex(elem_id, 'mpg', C_v, U_v, elem_name)
                    elif len(incoming_flows) == 1 and len(outgoing_flows) > 1:
                        graph.add_vertex(elem_id, 'spg', C_v, U_v, elem_name)
                    elif len(incoming_flows) > 1 and len(outgoing_flows) > 1:
                        raise Exception(f"Gateway modeling exception at {elem_id}")
                elif elem.tag != '{http://www.omg.org/spec/BPMN/20100524/MODEL}sequenceFlow':
                    graph.add_vertex(elem_id, 'oth', C_v, U_v, elem_name)

    # Parsing edges (sequence flows)
    for sequence_flow in root.findall('.//bpmn:sequenceFlow', ns):
        source_ref = sequence_flow.attrib['sourceRef']
        target_ref = sequence_flow.attrib['targetRef']
        
        try:
            edge_name = sequence_flow.attrib['name']
        except:
            edge_name = None

        target_vertex = graph.vertices.get(target_ref)
        
        # Check if target vertex is 'mpg' to insert empty vertices to implement the synchronization semantics of parallel block
        if target_vertex and target_vertex.T_v == 'mpg':
            # Insert an empty vertex
            insert_empty_vertex(graph, source_ref, target_ref)
        else:
            graph.add_edge(source_ref, target_ref, edge_name)

    # Clean unconnected vertices
    # "elif elem.tag ! = '{http://www.omg.org/spec/BPMN/20100524/MODEL}sequenceFlow': graph.add_vertex(elem_id, 'oth', C_v, U_v) "
    # This code in 'parse_text_annotation' will add some other elements as vertices of G, but these vertices do not have edges associated with them, such as the annotation element, so they need to be removed.
    graph.clean_unconnected_vertices()
    
    # Calculate edge weights (i.e., the probability of executing) for 'seg' vertices, as 1/n as default
    # it could be updated in PMDP transition function
    graph.calculate_edge_fromSEG_weights()
    
    # Initial vertex hierarchy
    graph.initial_vertex_hierarchy()
    
     # Calculate observed sets
    graph.calculate_observed_sets()
    
    # Parse graph KPIs and constraints
    graph.parse_graph_KPIsandConstraints(root, ns, annotations)
    
    return graph

if __name__ == "__main__":

    ############################  Use simple example to test the code  ############################
    file_path = 'BPMN_Models/TarvelAgency/Travel_Agency_BP_Userdefined_NFPs.bpmn'
    path_to_user_defiend_bp = parse_bpmn(file_path)
    pass
    
    #you can save it into as a file'
    


