import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# Parameters setting
# ---------------------------
date_range = np.arange(16)  # Date range 0,1,...,15
candidate_count_F = 20      # Number of flight suppliers under compliant traffic dates 0,1,...,19
candidate_count_T = 40      # Number of train suppliers under compliant traffic dates  0,1,...,39
candidate_count_H = 60     # Number of hotel suppliers under compliant hotel dates 0,1,...,59

records_per_candidate = 100  # Number of records per supplier

np.random.seed(42)

# Impossible values for candidates
IMPOSSIBLE_VALUE_for_NegativeBetterVariable = 99999
IMPOSSIBLE_VALUE_for_PositiveBetterVariable = 0

# ---------------------------
# Auxiliary functions
# ---------------------------
def round25(x):
    return int(round(x / 25) * 25)

def round10(x):
    return int(round(x / 10) * 10)

# ---------------------------
# Candidate generation functions (keep the original logic unchanged)
# ---------------------------
def generate_flight_candidate(D):

    price = np.random.normal(1000, 300)
    price = np.clip(round10(price), 300, 2000)

    # Flight time FT: generated from a normal distribution, with mean 120, std 30, adjusted by (price-550)*0.06, and with a 10% chance of adding noise (20~60)
    extra = np.random.uniform(10, 10) if np.random.rand() < 0.3 else 0
    time_val = np.random.normal(120, 30) - (price - 1000)*0.5 + extra
    time_val = np.clip(time_val, 60, 240)
    time_val = round10(time_val)


        # Dynamic pricing component
    days_advance = 15 - D  # Assume a total planning period of 15 days
    price += days_advance * 10  # The closer to departure, the higher the price

    # Weekend premium (D=5,6 are weekends)
    if (D % 7) >= 5:
        if np.random.rand() <= 0.8:
            price *= 1.3

    price = np.clip(round10(price), 300, 2000)

    return price, time_val

def generate_train_candidate(D):
    price = np.random.normal(500, 150)
    extra = np.random.uniform(30, 30) if np.random.rand() < 0.3 else 0
    time_val = np.random.normal(360, 60) - (price - 500)*0.5 + extra
    time_val = np.clip(round10(time_val), 120, 600)

    if (D % 7) >= 5:
        if np.random.rand() <= 0.8:
            price *= 1.3
    price = np.clip(round10(price), 100, 1500)

    return price, time_val

def generate_hotel_candidate(ci):
    hc = np.random.choice([3,4,5], p=[0.3,0.5,0.2])
    days_advance = 15 - ci
    if hc == 3:
        price = np.clip(np.random.normal(300, 50) + days_advance*-5, 150, 600)
    elif hc == 4:
        price = np.clip(np.random.normal(500, 100) + days_advance*-10, 300, 1000)
    else:
        price = np.clip(np.random.normal(800, 200) + days_advance*-15, 500, 2000)
    if (ci % 7) >= 5:
        price *= np.random.choice([1.0, 1.2], p=[0.3, 0.7])
    
    #### Return the single night price ####
    return round10(price), hc

# ---------------------------
# Data generation (new extended functions)
# ---------------------------
def generate_extended_flight(d, base_price, base_time):
    data = []
    price_std = np.clip(np.random.normal(100, 20), 10, 150)
    time_std = np.clip(np.random.normal(20, 10), 10, 40)

    # Generate candidate values: 5 to 15
    candidate_count = np.random.randint(5, 16)
    # Generate candidate prices and times based on the baseline values and random standard deviations (and clip)
    candidate_prices = np.clip(np.random.normal(base_price, price_std, candidate_count), 300, 2000)
    candidate_times = np.clip(np.random.normal(base_time, time_std, candidate_count), 60, 240)

    # Calculate the probability weights for the candidate values (based on normal distribution)
    if price_std > 0:
        price_weights = np.exp(-((candidate_prices - base_price)**2) / (2 * (price_std**2)))
    else:
        price_weights = np.ones(candidate_count)
    price_weights = price_weights / price_weights.sum()

    if time_std > 0:
        time_weights = np.exp(-((candidate_times - base_time)**2) / (2 * (time_std**2)))
    else:
        time_weights = np.ones(candidate_count)
    time_weights = time_weights / time_weights.sum()

    for _ in range(records_per_candidate):
        if np.random.rand() <= 0.95:
            # Sample from the candidate values
            new_price = np.random.choice(candidate_prices, p=price_weights)
            new_price = np.clip(round10(new_price), 300, 2000)
            new_time = np.random.choice(candidate_times, p=time_weights)
            new_time = np.clip(round10(new_time), 60, 240)
        else:
            new_price = base_price
            new_time = base_time
        data.append({"price": new_price, "time": new_time})

    return data

def generate_extended_train(d, base_price, base_time):
    data = []
    price_std = np.clip(np.random.normal(50, 10), 10, 50)
    time_std = np.clip(np.random.normal(40, 20), 10, 80)

    # Generate candidate values: 5 to 15
    candidate_count = np.random.randint(5, 16)
    # Generate candidate prices and times based on the baseline values and random standard deviations (and clip)
    candidate_prices = np.clip(np.random.normal(base_price, price_std, candidate_count), 100, 1500)
    candidate_times = np.clip(np.random.normal(base_time, time_std, candidate_count), 120, 600)

    # Calculate the probability weights for the candidate values (based on normal distribution)
    if price_std > 0:
        price_weights = np.exp(-((candidate_prices - base_price)**2) / (2 * (price_std**2)))
    else:
        price_weights = np.ones(candidate_count)
    price_weights = price_weights / price_weights.sum()
    
    if time_std > 0:
        time_weights = np.exp(-((candidate_times - base_time)**2) / (2 * (time_std**2)))
    else:
        time_weights = np.ones(candidate_count)
    time_weights = time_weights / time_weights.sum()
    
    for _ in range(records_per_candidate):
        if np.random.rand() <= 0.95:
            # Sample from the candidate values
            new_price = np.random.choice(candidate_prices, p=price_weights)
            new_price = np.clip(round10(new_price), 100, 1500)
            new_time = np.random.choice(candidate_times, p=time_weights)
            new_time = np.clip(round10(new_time), 120, 600)
        else:
            new_price = base_price
            new_time = base_time
        data.append({"price": new_price, "time": new_time})
    return data

def generate_extended_hotel(ci, co, base_price, base_hc):
    data = []
    nights = co - ci

    # Set standard deviations and nightly price ranges based on hotel star ratings
    if base_hc == 3:
        std = np.clip(np.random.normal(25, 25), 20, 40)
        lower_bound = 150
        upper_bound = 300
    elif base_hc == 4:
        std = np.clip(np.random.normal(50, 25), 20, 60)
        lower_bound = 250
        upper_bound = 500
    else:
        std = np.clip(np.random.normal(75, 25), 20, 100)
        lower_bound = 450
        upper_bound = 800

    # Generate candidate values: 5 to 15
    candidate_count = np.random.randint(5, 16)
    # Generate candidate prices (per night) and clip to the corresponding range
    candidate_prices = np.clip(np.random.normal(base_price, std, candidate_count), lower_bound, upper_bound)
    
    if std > 0:
        price_weights = np.exp(-((candidate_prices - base_price)**2) / (2 * (std**2)))
    else:
        price_weights = np.ones(candidate_count)
    price_weights = price_weights / price_weights.sum()

    for _ in range(records_per_candidate):
        if np.random.rand() <= 0.95:
            sampled_price = np.random.choice(candidate_prices, p=price_weights)
            new_total_price = sampled_price * nights
            new_total_price = np.clip(new_total_price, lower_bound * nights, upper_bound * nights)
            new_total_price = round10(new_total_price)
        else:
            total_price = base_price * nights
            new_total_price = np.clip(total_price, lower_bound * nights, upper_bound * nights)
            new_total_price = round10(new_total_price)
        data.append({"price": new_total_price, "HC": base_hc})
    return data

# ---------------------------
# Main data generation process
# ---------------------------
# Generate transport date pairs
transport_pairs = [{"D": d, "R": r, "valid": d < r} 
                  for d in date_range for r in date_range]

# Generate hotel date pairs
hotel_pairs = [{"CI": ci, "CO": co, "valid": ci < co}
              for ci in date_range for co in date_range]

# Generate transport data
F_transport_candidates = []
T_transport_candidates = []
for pair in tqdm(transport_pairs, desc="Generating transport candidates"):
    d, r, valid = pair["D"], pair["R"], pair["valid"]
    if valid:
        # Generate flight data
        for i in range(candidate_count_F):
            base_price, base_time = generate_flight_candidate(d)
            for ext in generate_extended_flight(d, base_price, base_time):
                F_transport_candidates.append({
                    "segment": "transport",
                    "D": d, "R": r,
                    "transport_type": "F",
                    "candidate_no": f"F_{d}_{r}_{i}",
                    "price": ext["price"],
                    "time": ext["time"],
                    "valid": True
                })
        # Generate train data
        for i in range(candidate_count_T):
            base_price, base_time = generate_train_candidate(d)
            for ext in generate_extended_train(d, base_price, base_time):
                T_transport_candidates.append({
                    "segment": "transport",
                    "D": d, "R": r,
                    "transport_type": "T",
                    "candidate_no": f"T_{d}_{r}_{i}",
                    "price": ext["price"],
                    "time": ext["time"],
                    "valid": True
                })
    else:
        # Non-compliant data retains original logic
        F_transport_candidates.append({
            "segment": "transport",
            "D": d, "R": r,
            "transport_type": "F",
            "candidate_no": -1,
            "price": IMPOSSIBLE_VALUE_for_NegativeBetterVariable,
            "time": IMPOSSIBLE_VALUE_for_NegativeBetterVariable,
            "valid": False
        })
        T_transport_candidates.append({
            "segment": "transport",
            "D": d, "R": r,
            "transport_type": "T",
            "candidate_no": -1,
            "price": IMPOSSIBLE_VALUE_for_NegativeBetterVariable,
            "time": IMPOSSIBLE_VALUE_for_NegativeBetterVariable,
            "valid": False
        })

# Generate hotel data
hotel_candidates = []
for pair in tqdm(hotel_pairs, desc="Generating hotel candidates"):
    ci, co, valid = pair["CI"], pair["CO"], pair["valid"]
    if valid:
        for i in range(candidate_count_H):
            base_price_per_night, base_hc = generate_hotel_candidate(ci)
            nights = co - ci
            #base_price = base_price_per_night * nights
            # Apply original discount logic
            if np.random.rand() < 0.2 and nights > 3:
                base_price *= 0.9 if np.random.rand() < 0.5 else 0.85
            for ext in generate_extended_hotel(ci, co, base_price_per_night, base_hc):
                hotel_candidates.append({
                    "segment": "hotel",
                    "CI": ci, "CO": co,
                    "candidate_no": f"H_{ci}_{co}_{i}",
                    "price": ext["price"],
                    "HC": ext["HC"],
                    "valid": True
                })
    else:
        hotel_candidates.append({
            "segment": "hotel",
            "CI": ci, "CO": co,
            "candidate_no": -1,
            "price": IMPOSSIBLE_VALUE_for_NegativeBetterVariable,
            "HC": IMPOSSIBLE_VALUE_for_PositiveBetterVariable,
            "valid": False
        })


        

# ---------------------------
# Data integration and output
# ---------------------------
df_F = pd.DataFrame(F_transport_candidates)
df_T = pd.DataFrame(T_transport_candidates)
df_H = pd.DataFrame(hotel_candidates)
df_all = pd.concat([df_F, df_T, df_H], ignore_index=True)

# Add data quality checks
assert (df_F[df_F['valid']].D < df_F[df_F['valid']].R).all(), "Invalid flight date pairs"
assert (df_H[df_H['valid']].CI < df_H[df_H['valid']].CO).all(), "Invalid hotel date pairs"


print(f"Traffic data: {len(df_F)+len(df_T)} items")
print(f"Hotel data: {len(df_H)} items")
print(f"Total data volume: {len(df_all)} items")
#df_all.to_csv("travel_dataset.csv", index=False)


# Filter out data where valid is True
valid_flight_data = df_F[df_F['valid'] == True]
valid_train_data = df_T[df_T['valid'] == True]
valid_hotel_data = df_H[df_H['valid'] == True]

Flight_Train = pd.concat([valid_flight_data, valid_train_data], axis=0, ignore_index=True)

# Add price-time scatter plot
sns.jointplot(x="price", y="time", hue="transport_type", data=Flight_Train)
plt.show()


# Add time series fluctuation chart
#hotel_prices.groupby("CI").mean()["price"].plot(title="Average Hotel Price by Check-in Date")

# Price over time distribution matrix

valid_pairs = valid_flight_data[valid_flight_data['D'] < valid_flight_data['R']]
sns.heatmap(pd.pivot_table(valid_pairs, index="D", columns="R", values="price"),  cmap='viridis')
plt.title('Flight Price Over Time')
plt.show()


sns.heatmap(pd.pivot_table(valid_flight_data, index="D", columns="R", values="price"), cmap='viridis')
plt.title('Flight Price Over Time')
plt.show()

sns.heatmap(pd.pivot_table(valid_train_data, index="D", columns="R", values="price"), cmap='viridis')
plt.title('Train Price Over Time')
plt.show()

# Time over time distribution matrix
sns.heatmap(pd.pivot_table(valid_flight_data, index="D", columns="R", values="time"), cmap='viridis')
plt.title('Flight Time Over Time')
plt.show()

sns.heatmap(pd.pivot_table(valid_train_data, index="D", columns="R", values="time"), cmap='viridis')
plt.title('Train Time Over Time')
plt.show()


# Hotel price distribution matrix
sns.heatmap(pd.pivot_table(valid_hotel_data, index="CI", columns="CO", values="price"), cmap='viridis')
plt.title('Hotel Price Distribution Matrix')
plt.show()

# Hotel star distribution matrix
sns.heatmap(pd.pivot_table(valid_hotel_data, index="CI", columns="CO", values="HC"), cmap='viridis')
plt.title('Hotel Star Rating Distribution Matrix')
plt.show()