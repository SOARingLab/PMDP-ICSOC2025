import pandas as pd
import random

from tqdm import tqdm



def get_travelAgency_distribution(file_path="Datasets/TravelAgencyNFPsDataset/travel_dataset.csv"):
    """
    Read the CSV dataset and use candidate_no as the key,
    for the probability distribution of the transport segment statistics (price, time) pairs,
    for the probability distribution of the hotel segment statistics (price, HC) pairs,
    return a dictionary.
    """
    df = pd.read_csv(file_path, low_memory=False)
    candidate_distribution = {}

    for candidate_no, group in tqdm(df.groupby("candidate_no"), desc="Read TravelAgency Dataset and Generate Distribution"):
        if group.empty:
            continue
        segment = group.iloc[0]["segment"]
        freq = {}
        if segment == "transport":
            for _, row in group.iterrows():
                key = (row["price"], row["time"])
                freq[key] = freq.get(key, 0) + 1
        elif segment == "hotel":
            for _, row in group.iterrows():
                key = (row["price"], row["HC"])
                freq[key] = freq.get(key, 0) + 1
        else:
            continue

        total = sum(freq.values())
        prob_dist = {k: v / total for k, v in freq.items()}
        candidate_distribution[candidate_no] = prob_dist

    return candidate_distribution

# Define a function to sample a (price, time) or (price, HC) pair from the probability distribution given a candidate supplier number
import numpy as np

def sample_from_distribution(candidate_no, candidate_distribution):
    """
    Given a candidate supplier number, sample a (price, time) or (price, HC) pair from its probability distribution.
    Use NumPy to construct the cumulative probability distribution and perform binary search.
    """
    prob_dist = candidate_distribution.get(candidate_no)
    if not prob_dist:
        raise ValueError(f"No distribution found for candidate {candidate_no}")
    
    # # Convert dictionary keys and values to lists respectively
    keys = list(prob_dist.keys())
    weights = np.array(list(prob_dist.values()))
    
    # Calculate the cumulative probability distribution
    cum_weights = np.cumsum(weights)
    # Generate a random number [0, 1)
    r = np.random.rand()
    # Use np.searchsorted to find the index where r falls in the cumulative probabilities
    index = np.searchsorted(cum_weights, r)
    return keys[index]


if __name__ == "__main__":

    # Read CSV data set, low_memory=False can avoid some warnings
    df = pd.read_csv("enhanced_travel_dataset.csv", low_memory=False)

    # Select all numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Replace 99999 with NaN, then calculate the max value for each column (ignoring NaN)
    max_values = df[numeric_cols].replace(99999, np.nan).max(skipna=True)

    print("Numeric columns (excluding 99999) max values:")
    print(max_values)


    distribution = get_travelAgency_distribution()
    # Example: Suppose want to sample for candidate number "F_0_1_0", getting (price, time) pair,
    for _ in range(15):
        candidate_no = f"F_0_{_+1}_0"  # To modify according to the actual candidate_no
        sampled_value = sample_from_distribution(candidate_no, distribution)
        print(f"Sampled value for candidate {candidate_no}: {sampled_value}")


