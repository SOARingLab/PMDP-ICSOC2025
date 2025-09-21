import ast
from collections import Counter
import random

def old_get_QWS_distribution(file_path, candidate_number):
    """
    Reads a txt file with every two lines forming a group:
    The first line of numbering information (numbering from 1)
    The second line is specific data in the format [[...], [...], ...]
    For each group of data,
    Divide the 2D list in the second row into candidate_number sub-2D lists according to candidate_number.
    For each sub-2D list: first de-weight each one-dimensional list (converted to a tuple),
    Count the frequency of occurrence of each unique one-dimensional list, and construct probability sampling data (probability = number of occurrences of the one-dimensional list / length of the sub-2D list)
    Returns a dictionary with the key as the group number and the value as a list, each element of which is (sub-list index, sampling probability dictionary)

    Reads a txt file with every two lines forming a group:
    The first line of numbering information (numbering from 1)
    The second line is specific data in the format [[...], [...] ...]
    For each group of data
    Divide the 2D list in the second row into candidate_number sublists according to candidate_number.
    For each sub-tuple: first de-weight each one-dimensional list (converted to a tuple). 
    Count the frequency of occurrence of each unique 1D list and construct a probability sample (probability = number of occurrences of the 1D list / length of the sub-2D list) 
    Return a dictionary with the group number as the key and a list as the value, with each element being (sub-list index, sampling probability dictionary)
    
    """
    results = {}
    with open(file_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Every two lines are grouped
    group_count = len(lines) // 2
    for i in range(group_count):
        # First line: group id information
        group_id_line = lines[2 * i][-2]
        # Second line: data string, in the format "[[...],[...],...]"
        data_line = lines[2 * i + 1]
        try:
            # Safely parse the data using ast.literal_eval
            data_list = ast.literal_eval(data_line)
        except Exception as ex:
            print(f"Failed to parse group {i+1}: {ex}")
            continue

        total_items = len(data_list)
        # Calculate the base size for each sub-2D list (may be uneven for the last one)
        base_size = total_items // candidate_number
        groups = []
        for j in range(candidate_number):
            if j < candidate_number - 1:
                sub_list = data_list[j * base_size:(j+1) * base_size]
            else:
                sub_list = data_list[j * base_size:]
            groups.append(sub_list)
        


        # Build probability distribution for each sub 2D list
        prob_data = []  # Each element is (sub-list index, {candidate: probability})
        for idx, sub in enumerate(groups):
            # Convert each 1D list in sub to tuple (for uniqueness)
            tuple_list = [tuple(item) for item in sub]
            count = Counter(tuple_list)
            total = sum(count.values())
            # Directly use tuples as keys
            prob_dict = {k: v/total for k, v in count.items()}
            prob_data.append((idx, prob_dict))


        # The group number can be extracted from group_id_line or directly used i+1
        results[i+1] = prob_data

    return results


def get_QWS_distribution(file_path, candidate_number):
    """
    Reads a txt file with every two lines forming one group:
      - The first line is the group id information (numbering from 1)
      - The second line is the group data in the format [[...],[...],...,[...]]
    For each group:
      1. Parse the second line to obtain a 2D list.
      2. Divide the original 2D list into 'candidate_number' sub-2D lists (row-wise split).
      3. For each candidate sub-2D list:
         a. Transpose it to obtain a new 2D list (columns become rows), while rounding each element to one decimal.
         b. For each row in the transposed matrix, count the frequency of each unique value and
            compute its probability (frequency divided by total elements in that row).
      4. Return a dictionary where the key is the group number and the value is a list of tuples for each candidate,
         each tuple being (candidate_index, [ (row index, probability distribution dict), ... ]).
    
    Note: candidate_number is used for dividing the original data rows.
    """
    results = {}
    with open(file_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Each two lines form one group.
    group_count = len(lines) // 2
    for i in range(group_count):
        group_num = i + 1
        # First line contains group id info (unused here)
        group_id_line = lines[2 * i]
        # Second line is a string representing a 2D list.
        data_line = lines[2 * i + 1]
        try:
            data_list = ast.literal_eval(data_line)
        except Exception as ex:
            print(f"Failed to parse data for group {group_num}: {ex}")
            continue
        
        # Divide the original 2D list into candidate_number sub-2D lists.
        total_items = len(data_list)
        base_size = total_items // candidate_number
        candidate_groups = []
        for j in range(candidate_number):
            if j < candidate_number - 1:
                sub_list = data_list[j * base_size:(j + 1) * base_size]
            else:
                sub_list = data_list[j * base_size:]
            candidate_groups.append(sub_list)
        
        # Process each candidate sub-2D list:
        candidate_results = []  # Will hold (candidate index, [ (row index, probability dict), ... ])
        for cand_idx, sub_data in enumerate(candidate_groups):
            # Transpose the candidate sub-2D list.
            # If sub_data is empty we skip candidate.
            if not sub_data:
                candidate_results.append((cand_idx, []))
                continue
            transposed = []
            # zip(*sub_data) returns tuples corresponding to each column.
            for col in zip(*sub_data):
                new_row = [round(float(val), 1) for val in col]

                # Filter all 0.0 items in new_row
                new_row = [val for val in new_row if val != 0.0]
                
                transposed.append(new_row)
            
            # Optionally, you can limit the number of rows after transposition. Here, no truncation is performed and all rows are calculated.
            # For each row in the transposed matrix, build the probability distribution.
            prob_data = []  # Each element: (row index, {value: probability})
            for row_idx, row in enumerate(transposed):
                count = Counter(row)
                total = sum(count.values())
                prob_dict = {num: freq/total for num, freq in count.items()}
                prob_data.append((row_idx, prob_dict))
            
            candidate_results.append((cand_idx, prob_data))
        
        results[group_num] = candidate_results

    return results


def sample_candidate(candidate_result, target_rows):
    """
    Given a candidate result tuple of the form:
        (candidate_index, [ (row index, probability dict), ... ])
    and a list of target row indices (target_rows) to sample from,
    for each target row, sample a value using its probability distribution.
    Return a dictionary mapping each target row index to its sampled value.
    """
    cand_idx, row_dists = candidate_result
    # Create a mapping from row index to its probability dictionary.
    row_dict = {row_idx: prob_dict for row_idx, prob_dict in row_dists}
    samples = {}
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
    return samples



if __name__ == "__main__":

    file_path = "Datasets/QWS/normalized_chunks_10_AbstractServices_Eech_has_250_ConcreteServices.txt"  # Replace as needed.
    candidate_number = 10
    distributions = get_QWS_distribution(file_path, candidate_number)
    
    # Print the probability distributions for each group.
    for group_id, row_dists in distributions.items():
        print(f"Group {group_id}:")
        for row_idx, prob_dict in row_dists:
            print(f"  Row {row_idx}: {prob_dict}")
    
    # For example, to sample from group1, row0:
    if 1 in distributions:
        # For example, sample from group 1, candidate 0:
        group1_candidate0 = next(item for item in distributions[1] if item[0] == 0)
        sampled = sample_candidate(group1_candidate0, [0, 0])
        print("Sampled values from Group 1, Candidate 0:", sampled)

    file_path = "Datasets/QWS/normalized_chunks_10_AbstractServices_Eech_has_250_ConcreteServices.txt"  # use with the actual file path
    candidate_number = 10
    distributions = get_QWS_distribution(file_path, candidate_number)
    for group_id, sub_dists in distributions.items():
        print(f"group {group_id}:")
        for sub_idx, prob_dict in sub_dists:
            print(f"  sub-group {sub_idx}: {prob_dict}")

    # Example: Sampling subgroup 0 of group 1
    group1 = distributions[1]
    # Find the probability dictionary for subgroup index 0
    sub_idx, prob_dict = next(item for item in group1 if item[0] == 0)
    sample = sample_candidate(prob_dict)
    print("sample result: ", sample)