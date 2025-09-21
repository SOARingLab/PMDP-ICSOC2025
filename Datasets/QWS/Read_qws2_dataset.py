import csv
import os
import numpy as np
def read_qws_dataset(file_path):
    """
    Reads the QWS dataset and extracts the first 9 QoS attributes from each record.

    Args:
    file_path (str): The path to the dataset file.

    Returns:
    list of list: A list of records, each containing the first 9 QoS attributes.
    """
    qos_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        
        # Skip the header lines until we reach the data records
        for row in reader:
            if row and row[0].startswith('302.75'):  # Assuming the first data record starts with '302.75'
                # Extract the first 5 QoS attributes of the first record

                qos_record = row[:8]
              
                #qos_record = row[:2] + row[4:6] + row[7:8] 
                #qos_record = [round(float(qos_record[0]) / 1000, 1)] + [int(round(float(value))) for value in qos_record[1:-1]] + [round(float(qos_record[-1]) / 1000, 1)] 
                qos_data.append(qos_record)
    
                break
        
        # Read the data records
        for row in reader:
            if row:
                ##########################################################################
                ## Format: QWS parameters are separated by commas (first nine)		##
                ## Format: (1) Response Time						##
                ## Format: (2) Availability						##
                ## Format: (3) Throughput						##
                ## Format: (4) Successability						##
                ## Format: (5) Reliability						##
                ## Format: (6) Compliance						##
                ## Format: (7) Best Practices						##
                ## Format: (8) Latency							##

                qos_record = row[:8]
               
                
                #qos_record = [round(float(qos_record[0]) / 1000, 1)] + [int(round(float(value))) for value in qos_record[1:-1]] + [round(float(qos_record[-1]) / 1000, 1)] 
                qos_data.append(qos_record)
    
    return qos_data

def split_qos_data(qos_data, n):
    """
    Splits the qos_data into n chunks, handling remainders appropriately.

    Args:
    qos_data (list of list): The list of QoS data records.
    n (int): The number of chunks to divide the data into.

    Returns:
    list of list of list: A list containing n chunks of the original data.
    """
    chunk_size = len(qos_data) // n
    remainder = len(qos_data) % n

    chunks = [qos_data[i:i + chunk_size] for i in range(0, len(qos_data) - remainder, chunk_size)]
    
    if remainder > chunk_size / 2:
        chunks.append(qos_data[-remainder:])
    else:
        chunks[-1].extend(qos_data[-remainder:])
        pass
    
    return chunks

def calculate_domains(chunks):
    """
    Calculates the domains for each chunk in each dimension.

    Args:
    chunks (list of list of list): The list of chunks, where each chunk is a list of records.

    Returns:
    list of list of list: A list containing the domains for each chunk in each dimension.
    """
    domains = []
    for chunk in chunks:
        chunk_domains = []
        num_columns = len(chunk[0])
        for i in range(num_columns):
            column_values = [row[i] for row in chunk]
            unique_values = list(set(column_values))
            chunk_domains.append(unique_values)
        domains.append(chunk_domains)
    return domains


def calculate_domain_sizes(domains):
    """
    Calculates the sizes of each domain in each chunk.

    Args:
    domains (list of list of list): The list of domains for each chunk.

    Returns:
    list of list of int: A list containing the sizes of each domain in each chunk.
    """
    domain_sizes = []
    for chunk in domains:
        chunk_sizes = [len(domain) for domain in chunk]
        domain_sizes.append(chunk_sizes)
    return domain_sizes


def normalize_chunks(chunks, better_is_high):
    """
    Normalizes the data in chunks across each column dimension,
    considering whether higher values are better for each column.

    Args:
        chunks (list of list of list): The list of chunks, where each chunk is a list of records.
        better_is_high (list of bool): A list indicating if higher values are better for each column.

    Returns:
        list of list of list: The normalized chunks.
    """
    # Flatten all records from all chunks into a single list
    all_records = [record for chunk in chunks for record in chunk]
    
    # Transpose to get columns
    columns = list(zip(*all_records))
    
    # Normalize each column
    normalized_columns = []
    for i, column in enumerate(columns):
        min_val = min(column)
        max_val = max(column)
        if better_is_high[i]:
            normalized_column = [
                round((value - min_val) / (max_val - min_val), 3) if max_val != min_val else 0 
                for value in column
            ]
        else:
            normalized_column = [
                round((max_val - value) / (max_val - min_val), 3) if max_val != min_val else 0 
                for value in column
            ]
        normalized_columns.append(normalized_column)
    
    # Transpose back to rows
    normalized_records = list(zip(*normalized_columns))
    
    # Split back into chunks of the original sizes
    normalized_chunks = []
    index = 0
    for chunk in chunks:
        chunk_size = len(chunk)
        normalized_chunks.append([list(record) for record in normalized_records[index:index + chunk_size]])
        index += chunk_size
    
    return normalized_chunks

def remove_duplicates(chunk):
    """
    Removes duplicate records from a chunk while preserving order.

    Args:
        chunk (list of list): A chunk containing multiple records.

    Returns:
        list of list: The chunk with duplicate records removed.
    """
    seen = set()
    unique_chunk = []
    for record in chunk:
        record_tuple = tuple(record)  # Convert to tuple to make it hashable
        if record_tuple not in seen:
            seen.add(record_tuple)
            unique_chunk.append(record)
    return unique_chunk
    
def save_normalized_chunks(normalized_chunks, file_path):
    """
    Saves the normalized chunks to a text file.

    Args:
        normalized_chunks (list of list of list): The normalized chunks.
        file_path (str): The path to the file where the normalized chunks will be saved.
    """
    with open(file_path, 'w') as file:
        for i, chunk in enumerate(normalized_chunks):
                    # Remove duplicate records in the chunk
                    unique_chunk = remove_duplicates(chunk)
                    
                    # Write the unique chunk to the file
                    file.write(f'The Concrete Services for Abstract Service {i+1}: \n{unique_chunk}\n')
                    
    
## Adding another attribute to the dataset used in the normed-mdp paper
def sample_attribute(probabilities=None):
    """
    Sample a value with the specified probability, mapping to 1, 2, 3.

    Args:
    probabilities (list of float, optional): Probabilities of sampling 1, 2, 3. Should be a list of length 3 that sums to 1.
    If not specified, defaults to uniform distribution [1/3, 1/3, 1/3].

    Returns:
    int: Properties that can be 1, 2, or 3.
    """
    if probabilities is None:
        probabilities = [1/3, 1/3, 1/3]
    else:
        if len(probabilities) != 3:
            raise ValueError("probabilities must be a list of three elements.")
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("The sum of probabilities must be 1.")
    
    return np.random.choice([1, 2, 3], p=probabilities)

def add_attributes(normalized_chunks):
    """
    Add three attributes to each record, with values 1, 2, and 3.

    Args:
    normalized_chunks (list of list of list): The original list of matrix chunks.

    Returns:
    list of list of list: The list of matrix chunks after adding the attributes.
    """
    for chunk in normalized_chunks:
        for record in chunk:
            # Add three attributes to each record
            for _ in range(3):
                attribute = sample_attribute(probabilities = [0.4, 0.2, 0.4])
                record.append(attribute)
    return normalized_chunks


if  __name__ == "__main__":

    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    #
    file_path = os.path.join(current_dir, 'qws2.txt')

    qos_data = read_qws_dataset(file_path)

    n = 50
    chunks = split_qos_data(qos_data, n)

    # Convert each element in each record to float
    chunks = [[[float(value) for value in record] for record in chunk] for chunk in chunks]
    better_is_high = [False, True, True, True, True, True, True, False]
    if len(better_is_high) != len(chunks[0][0]):
        raise ValueError("The better_is_high list must have the same length as the number of columns to specify the direction of optimization for each column.")

    normalized_chunks = normalize_chunks(chunks, better_is_high)

    # Add three attributes
    normalized_chunks = add_attributes(normalized_chunks)

    
    # Example usage:
    file_path = f'normalized_chunks_{n}_AbstractServices_Each_has_{len(chunks[0])}_ConcreteServices.txt'
    save_normalized_chunks(normalized_chunks, file_path)


    for i, ormalized_chunk in enumerate(normalized_chunks):
        print(f"Chunk {i+1}:")
        print(ormalized_chunk)


    domains = calculate_domains(normalized_chunks)


    domain_sizes = calculate_domain_sizes(domains)
    pass
