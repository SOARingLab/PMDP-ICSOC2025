import ast

def generate_group_strings(file_path, candidate_number):
    """
    Reads a txt file with every two lines forming one group:
      - The first line is the group id information (starting from 1).
      - The second line is the group's data in the format [[...],[...],...,[...]].
    For each group, generate a string formatted as:
    "C={cX:{[0],[1],...,[candidate_number-1]}||U={uX#positive:{(list1,unknown),(list2,unknown),..., (listN,unknown)}}"
    
    The "C" part always uses candidate_number candidates,
    and the "U" part contains as many tuples as items in the 2D list from the file.
    """
    results = []
    with open(file_path, encoding="utf-8") as f:
        # Filter out empty lines and remove surrounding whitespace.
        lines = [line.strip() for line in f if line.strip()]
        
    group_count = len(lines) // 2
    for i in range(group_count):
        group_num = i + 1
        
        # First line: group id information (can be used if needed)
        group_id_line = lines[2 * i]
        
        # Second line: data (a string representing a 2D list)
        data_line = lines[2 * i + 1]
        try:
            # Safely parse the data string into a Python list
            data_list = ast.literal_eval(data_line)
        except Exception as ex:
            print(f"Failed to parse group {group_num}: {ex}")
            continue
        
        # Build the C part with candidate_number placeholders:
        # For example, if candidate_number==10, then C={c1:{[0],[1],...,[9]}}
        candidate_items = ",".join([f"[{j}]" for j in range(candidate_number)])
        c_part = f"C={{c{group_num}:{{{candidate_items}}}}}"
        
        # Build the U part using each one-dimensional list from data_list.
        # For each item, we format as "({list},unknown)"
        u_items = []
        for item in data_list:
            # Convert item (one-dimensional list) to string.
            # You can adjust formatting if required.
            item_str = str(item)
            u_items.append(f"({item_str},unknown)")
        u_items_str = ",".join(u_items)
        u_part = f"U={{u{group_num}#positive:{{{u_items_str}}}}}"
        
        # Concatenate C part and U part by "||"
        group_string = f"{c_part}||{u_part}"
        results.append(group_string)
    
    return results


if __name__ == "__main__":
    file_path = "Datasets/QWS2/normalized_chunks_10_AbstractServices_Eech_has_250_ConcreteServices.txt"  # Replace with your actual file path
    candidate_number = 10
    group_strings = generate_group_strings(file_path, candidate_number)
    for s in group_strings:
        print("\n")
        print(s)