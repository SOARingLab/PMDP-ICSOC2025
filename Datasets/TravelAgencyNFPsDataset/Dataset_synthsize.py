
# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# Parameter settings
# ---------------------------
# Date range from 0 to 15
date_range = np.arange(16)  
# Number of flight suppliers under compliant traffic dates
candidate_count_F = 20      
# Number of train suppliers under compliant traffic dates
candidate_count_T = 40      
# Number of hotel suppliers under compliant hotel dates
candidate_count_H = 60     
# Number of records per supplier
records_per_candidate = 100  

# Set the random seed for reproducibility
np.random.seed(42)

# Impossible values for different types of variables
# Impossible value for variables where a negative value is better
IMPOSSIBLE_VALUE_for_NegativeBetterVariable = 9999
# Impossible value for variables where a positive value is better
IMPOSSIBLE_VALUE_for_PositiveBetterVariable = 0

# ---------------------------
# Auxiliary functions
# ---------------------------
# Round a number to the nearest multiple of 25
def round20(x):
    return int(round(x / 25) * 25)

# Round a number to the nearest multiple of 10
def round10(x):
    return int(round(x / 10) * 10)

# ---------------------------
# Candidate generation functions (keep the original logic unchanged)
# ---------------------------
# Generate a flight candidate
def generate_flight_candidate(D):
    # Generate a price from a normal distribution
    price = np.random.normal(1000, 300)
    # Clip the price to the range [300, 2000] and round to the nearest multiple of 25
    price = np.clip(round20(price), 300, 2000)

    # Flight time FT: generated from a normal distribution, adjusted by price, and with a 10% chance of adding noise
    extra = np.random.uniform(10, 10) if np.random.rand() < 0.3 else 0
    time_val = np.random.normal(120, 30) - (price - 1000)*0.5 + extra
    # Clip the time to the range [60, 180] and round to the nearest multiple of 10
    time_val = np.clip(time_val, 60, 180)
    time_val = round10(time_val)

    # Dynamic pricing component
    # Calculate the number of days in advance
    days_advance = 15 - D  
    # Apply a discount for each day of delay in booking
    price += days_advance * -30  

    # Weekend premium (D = 5, 6 are weekends)
    if (D % 7) >= 5:
        if np.random.rand() <= 0.8:
            price *= 1.2

    # Clip the price again to the range [300, 2000] and round to the nearest multiple of 25
    price = np.clip(round20(price), 300, 2000)

    return price, time_val

# Generate a train candidate
def generate_train_candidate(D):
    # Generate a price from a normal distribution
    price = np.random.normal(500, 150)
    extra = np.random.uniform(30, 30) if np.random.rand() < 0.3 else 0
    time_val = np.random.normal(360, 60) - (price - 500)*0.5 + extra
    # Clip the time to the range [120, 600] and round to the nearest multiple of 10
    time_val = np.clip(round10(time_val), 120, 600)

    # Weekend premium
    if (D % 7) >= 5:
        if np.random.rand() <= 0.8:
            price *= 1.2
    # Clip the price to the range [100, 1500] and round to the nearest multiple of 25
    price = np.clip(round20(price), 100, 1500)

    return price, time_val

# Generate a hotel candidate
def generate_hotel_candidate(ci):
    # Randomly select the hotel class (3, 4, or 5 stars)
    hc = np.random.choice([3,4,5], p=[0.3,0.5,0.2])
    # Calculate the number of days in advance
    days_advance = 15 - ci
    if hc == 3:
        # Generate a price for a 3-star hotel, adjusted by the number of days in advance
        price = np.clip(np.random.normal(300, 50) + days_advance*-5, 150, 600)
    elif hc == 4:
        # Generate a price for a 4-star hotel, adjusted by the number of days in advance
        price = np.clip(np.random.normal(500, 100) + days_advance*-10, 300, 1000)
    else:
        # Generate a price for a 5-star hotel, adjusted by the number of days in advance
        price = np.clip(np.random.normal(800, 200) + days_advance*-15, 500, 2000)
    # Weekend premium
    if (ci % 7) >= 5:
        price *= np.random.choice([1.0, 1.2], p=[0.3, 0.7])
    
    # Return the single night price
    return round20(price), hc

# ---------------------------
# Data generation (new extended functions)
# ---------------------------
# Generate extended flight data
def generate_extended_flight(d, base_price, base_time):
    data = []
    # Generate a standard deviation for price
    price_std = np.clip(np.random.normal(50, 50), 0, 50)
    # Generate a standard deviation for time
    time_std = np.clip(np.random.normal(10, 10), 0, 20)

    for _ in range(records_per_candidate):
        if np.random.rand() <= 0.95:
            # Generate a new price from a normal distribution
            new_price = np.random.normal(base_price, price_std)
            # Clip the price to the range [300, 2000] and round to the nearest multiple of 25
            new_price = np.clip(round20(new_price), 300, 2000)
            # Generate a new time from a normal distribution
            new_time = np.random.normal(base_time, time_std)
            # Clip the time to the range [60, 150] and round to the nearest multiple of 10
            new_time = np.clip(round10(new_time), 60, 150)
        else:
            new_price = base_price
            new_time = base_time
        data.append({"price": new_price, "time": new_time})
    return data

# Generate extended train data
def generate_extended_train(d, base_price, base_time):
    data = []
    # Generate a standard deviation for price
    price_std = np.clip(np.random.normal(50, 50), 0, 50)
    # Generate a standard deviation for time
    time_std = np.clip(np.random.normal(10, 10), 0, 20)
    
    for _ in range(records_per_candidate):
        if np.random.rand() <= 0.95:
            # Generate a new price from a normal distribution
            new_price = np.random.normal(base_price, price_std)
            # Clip the price to the range [100, 1500] and round to the nearest multiple of 25
            new_price = np.clip(round20(new_price), 100, 1500)
            # Generate a new time from a normal distribution
            new_time = np.random.normal(base_time, time_std)
            # Clip the time to the range [120, 600] and round to the nearest multiple of 10
            new_time = np.clip(round10(new_time), 120, 600)
        else:
            new_price = base_price
            new_time = base_time

        data.append({"price": new_price, "time": new_time})
    return data

# Generate extended hotel data
def generate_extended_hotel(ci, co, base_price, base_hc):
    data = []
    # Calculate the number of nights
    nights = co - ci

    # Generate standard deviations for different hotel classes
    std_3 = np.clip(np.random.normal(25, 25), 25, 50)
    std_4 = np.clip(np.random.normal(25, 50), 25, 50)
    std_5 = np.clip(np.random.normal(25, 75), 50, 100)

    for _ in range(records_per_candidate):
        if np.random.rand() <= 0.95:
            # Generate a new price based on the hotel class
            if base_hc == 3:
                std = std_3
            elif base_hc == 4:
                std = std_4
            else:
                std = std_5
            new_price = np.random.normal(base_price, std)

            # Calculate the total price for the stay
            new_total_price = new_price * nights

            # Clip the total price based on the hotel class
            if base_hc == 3:
                new_total_price = np.clip(new_total_price, 150*nights, 300*nights)
            elif base_hc == 4:
                new_total_price = np.clip(new_total_price, 250*nights, 500*nights)
            else:
                new_total_price = np.clip(new_total_price, 450*nights, 800*nights)
            # Round the total price to the nearest multiple of 25
            new_total_price = round20(new_total_price)

        else:
            # Calculate the total price for the stay
            total_price = base_price * nights
            # Clip the total price based on the hotel class
            if base_hc == 3:
                new_total_price = np.clip(total_price, 150*nights, 300*nights)
            elif base_hc == 4:
                new_total_price = np.clip(total_price, 250*nights, 500*nights)
            else:
                new_total_price = np.clip(total_price, 450*nights, 800*nights)
            
            # Round the total price to the nearest multiple of 25
            new_total_price = round20(new_total_price)

        data.append({"price": new_total_price, "HC": base_hc})
    return data

# ---------------------------
# Main data generation process
# ---------------------------
# Generate transportation date plans
transport_pairs = [{"D": d, "R": r, "valid": d < r} 
                  for d in date_range for r in date_range]

# Generate hotel date plans
hotel_pairs = [{"CI": ci, "CO": co, "valid": ci < co}
              for ci in date_range for co in date_range]

# Generate transportation data
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
        # Non-compliant data, keep the original logic
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
            # Calculate the number of nights
            nights = co - ci
            #base_price = base_price_per_night * nights
            # Apply the original discount logic
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
# Convert flight transportation candidates to a DataFrame
df_F = pd.DataFrame(F_transport_candidates)
# Convert train transportation candidates to a DataFrame
df_T = pd.DataFrame(T_transport_candidates)
# Convert hotel candidates to a DataFrame
df_H = pd.DataFrame(hotel_candidates)
# Concatenate all DataFrames
df_all = pd.concat([df_F, df_T, df_H], ignore_index=True)

# Add data quality checks
assert (df_F[df_F['valid']].D < df_F[df_F['valid']].R).all(), "Invalid flight date pairs"
assert (df_H[df_H['valid']].CI < df_H[df_H['valid']].CO).all(), "Invalid hotel date pairs"

# Print the number of transportation, hotel, and total data records
print(f"Transportation data: {len(df_F)+len(df_T)} records")
print(f"Hotel data: {len(df_H)} records")
print(f"Total data volume: {len(df_all)} records")
# Save the combined data to a CSV file
df_all.to_csv("enhanced_travel_dataset.csv", index=False)

# Filter out valid flight data
valid_flight_data = df_F[df_F['valid'] == True]
# Filter out valid train data
valid_train_data = df_T[df_T['valid'] == True]
# Filter out valid hotel data
valid_hotel_data = df_H[df_H['valid'] == True]

# Concatenate valid flight and train data
Flight_Train = pd.concat([valid_flight_data, valid_train_data], axis=0, ignore_index=True)

# Add a price-time scatter plot
sns.jointplot(x="price", y="time", hue="transport_type", data=Flight_Train)
plt.show()

# Add a time series fluctuation plot
#hotel_prices.groupby("CI").mean()["price"].plot(title="Average Hotel Price by Check-in Date")


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

# 时间随时间分布矩阵
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

# Hotel star rating distribution matrix
sns.heatmap(pd.pivot_table(valid_hotel_data, index="CI", columns="CO", values="HC"), cmap='viridis')
plt.title('Hotel Star Rating Distribution Matrix')
plt.show()