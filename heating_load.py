# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# ELECTRICITY USAGE and TEMPERATURE
# CALCULATE HEATING LOAD
#
# This script expects electricity usage reports from Fingrid Datahub.
# This script expects temperature reports from Finnish Meteorological Institute.
# Other data sources can of course be used, but you need to modify data normalisation,
# and loading functions to support differing file schemas,
#
# Expected input file names:
# /raw/<year>_temperature_hourly.csv
# /raw/<year>_electricity_hourly.csv

# %%
# Data sources

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import pprint
pp = pprint.PrettyPrinter(depth=4)

RAW_FOLDER = "raw/"
NORMALISED_FOLDER = "normalised/"
ADDRESS = "<YOUR ADDRESS HERE>"
INTERNAL_TEMPERATURE = 22.0 # celcius, target temperature
NO_HEAT_TEMP_MIN = 18
NO_HEAT_TEMP_MAX = 22

os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(NORMALISED_FOLDER, exist_ok=True)


# %%
# NORMALISE ELECTRICITY DATA

def normalise_electricity(file):
    df = pd.read_csv(file, sep=';', usecols=['Alkuaika', 'Määrä'], parse_dates=['Alkuaika'])
    df.rename(columns={'Alkuaika': 'timestamp', 'Määrä': 'electricity_kWh'}, inplace=True)

    df['electricity_kWh'] = df['electricity_kWh'].str.replace(',', '.').astype(float)  

    # Ensure timestamps are UTC
    if df['timestamp'].dt.tz is None:  # If timestamps are naive
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:  # If already timezone-aware
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    # Define column to use as index
    df.set_index('timestamp', inplace=True)

    # Resample the time series to hourly
    df = df.resample('h').sum()

    # Create full hourly range for the year
    start, end = df.index.min().replace(month=1, day=1, hour=0), df.index.max().replace(month=12, day=31, hour=23)
    full_index = pd.date_range(start=start, end=end, freq='h', tz="UTC")

    # Reindex to fill missing timestamps
    df = df.reindex(full_index)

    # Create "is_valid" column (True for existing, False for missing data)
    df["is_valid"] = ~df["electricity_kWh"].isna()

    # Fill missing electricity values with 0.0 (Avoid using inplace=True)
    df["electricity_kWh"] = df["electricity_kWh"].fillna(0.0)

    return df

# ----- PROCESSING -----

electricity_files = glob(f"{RAW_FOLDER}/*electricity_hourly.csv")
for file in electricity_files:
    year = file.split('_')[0][-4:]  # Extract year from filename
    df = normalise_electricity(file)
    output_path = f"{NORMALISED_FOLDER}/{year}_electricity_hourly.csv"
    df.to_csv(output_path, index_label="timestamp")
    print(f"Normalised {file} -> {output_path}")


# %%
# NORMALISE TEMPERATURE DATA

def normalise_temperature(file):
    df = pd.read_csv(file, usecols=['Vuosi', 'Kuukausi', 'Päivä', 'Aika [UTC]', 'Lämpötilan keskiarvo [°C]'])
    df['timestamp'] = pd.to_datetime(df[['Vuosi', 'Kuukausi', 'Päivä']].astype(str).agg('-'.join, axis=1) + ' ' + df['Aika [UTC]'])
    df.rename(columns={'Lämpötilan keskiarvo [°C]': 'temperature_C'}, inplace=True)
    
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')  # Ensure timestamps are UTC

    # Define column to use as index
    df.set_index('timestamp', inplace=True)

    # Set type
    df = df[['temperature_C']].astype(float)

    # Resample the time series to hourly
    df = df.resample('h').mean()
    
    # Create full hourly range for the year
    start, end = df.index.min().replace(month=1, day=1, hour=0), df.index.max().replace(month=12, day=31, hour=23)
    full_index = pd.date_range(start=start, end=end, freq='h', tz="UTC")

    # Reindex to fill missing timestamps
    df = df.reindex(full_index)

    # Create "is_valid" column (True for existing, False for missing data)
    df["is_valid"] = ~df["temperature_C"].isna()

    # Fill missing temperature values with NaN (or any other placeholder if needed)
    df["temperature_C"] = df["temperature_C"].ffill()  # Use forward fill to propagate previous values

    return df

# ----- PROCESSING -----

temperature_files = glob(f"{RAW_FOLDER}/*temperature_hourly.csv")

for file in temperature_files:
    year = file.split('_')[0][-4:]  # Extract year from filename
    df = normalise_temperature(file)
    output_path = f"{NORMALISED_FOLDER}/{year}_temperature_hourly.csv"
    df.to_csv(output_path, index_label="timestamp")
    print(f"Normalised {file} -> {output_path}")

# %%
# LOAD NORMALISED DATA

## Used electricity in kWh
total_electricity_hourly = dict()
for file in glob(f"{NORMALISED_FOLDER}/*electricity_hourly.csv"):
    year = file.split('_')[0][-4:]
    total_electricity_hourly[year] = pd.read_csv(file, parse_dates=['timestamp'])
    total_electricity_hourly[year].set_index("timestamp", inplace=True)

## Outside temperature in celcius
total_temperature_hourly = dict()
for file in glob(f"{NORMALISED_FOLDER}/*temperature_hourly.csv"):
    year = file.split('_')[0][-4:]
    total_temperature_hourly[year] = pd.read_csv(file, parse_dates=['timestamp'])
    total_temperature_hourly[year].set_index("timestamp", inplace=True)


# %%
# ESTIMATE HOUSE BASE LOAD

## Base load estimation is needed to separate heating / cooling electricity
## usage from other electricity usage. This is estimated for each year
## separately. Removing this estimated base load from the electricity
## data will then give out best-quess value for actual heating power required.
##
## Estimation is done by looking at long time spans with temperature between
## room temperature degrees celcius. At those temperatures, the need for heating or
## cooling should be minimal. 25th percentile values over multiple years are
## used to get better estimate.

def estimate_base_load(total_electricity_hourly, total_temperature_hourly):
    base_load_estimates = {}

    for year in sorted(total_electricity_hourly.keys()):
        # Get electricity and temperature data for the year
        electricity_data = total_electricity_hourly[year]
        temperature_data = total_temperature_hourly[year]

        # Merge datasets
        combined_data = electricity_data[['electricity_kWh']].join(temperature_data[['temperature_C']], how='inner')
        combined_data.dropna(inplace=True)

        # Initialize list for storing (date, base_load, avg_temp)
        daily_base_loads = []

        # Group data by day
        for day, day_data in combined_data.groupby(combined_data.index.date):
            within_range = (day_data['temperature_C'] >= NO_HEAT_TEMP_MIN) & (day_data['temperature_C'] <= NO_HEAT_TEMP_MAX)
        
            # Find the longest continuous span
            longest_span = []
            current_span = []
        
            for idx, valid in enumerate(within_range):
                if valid:
                    current_span.append(day_data.index[idx])
                else:
                    if len(current_span) > len(longest_span):
                        longest_span = current_span
                    current_span = []
        
            # Check last span
            if len(current_span) > len(longest_span):
                longest_span = current_span
        
            # Compute base load for the longest span
            if longest_span:
                span_data = day_data.loc[longest_span]
                total_usage = span_data['electricity_kWh'].sum()
                avg_temp = span_data['temperature_C'].mean()
                hours = len(span_data)
        
                # Short spans are not representative
                if hours > 4:
                    scaled_base_load = (total_usage / hours) * 24  # Scale to full day
                    
                    # Check for NaN values in base load or avg temp
                    if np.isnan(scaled_base_load) or np.isnan(avg_temp):
                        continue  # Skip this entry if NaN
                    
                    # Append to the daily base load list
                    daily_base_loads.append((pd.Timestamp(day), scaled_base_load, avg_temp))

        # Convert daily base loads to Pandas Series for easy processing
        if daily_base_loads:
            # Convert to DataFrame for proper handling
            daily_base_load_df = pd.DataFrame(
                daily_base_loads, columns=["date", "base_load", "avg_temp"]
            ).set_index("date")

            # Drop rows with NaN values in base_load or avg_temp before resampling
            daily_base_load_df = daily_base_load_df.dropna(subset=["base_load", "avg_temp"])
        
            # Aggregate to monthly mean values
            monthly_base_loads = daily_base_load_df.resample('ME').quantile(0.25)
        
            # Pretty print monthly values
            #print(f"\n{year} Monthly Base Loads (kWh/day) & Avg Temperature (°C):")
            #for month, row in monthly_base_loads.iterrows():
            #    if not np.isnan(row["base_load"]):
            #        print(f"  {month.strftime('%Y-%m')}: {row['base_load']:.2f} kWh, {row['avg_temp']:.1f}°C")
        
            # Store yearly aggregates
            base_load_estimates[year] = {
                "mean": daily_base_load_df["base_load"].mean(),
                "q1": np.percentile(daily_base_load_df["base_load"], 25),
                "avg_temp": daily_base_load_df["avg_temp"].mean()  # Store avg temp for the year
            }
        else:
            base_load_estimates[year] = {"mean": None, "q1": None, "avg_temp": None}  # No valid data


    return base_load_estimates

# ----- PROCESSING -----

base_load_results = estimate_base_load(total_electricity_hourly, total_temperature_hourly)
# pp.pprint(base_load_results)

q1_values = {year: data["q1"] for year, data in base_load_results.items() if data["q1"] is not None}
DAILY_BASE_LOAD = np.mean([q1_values[year]])
print(f"Calculated base load: {DAILY_BASE_LOAD:.1f} kWh")

# ----- PLOTTING -----

# Extract data for plotting
years = sorted(base_load_results.keys())
q1_values = [base_load_results[year]['q1'] for year in years]

# Plot the results
plt.figure(figsize=(10, 5))
width = 0.4  # Bar width

# Plot Q1 (25th percentile)
plt.bar([int(y) + 0.2 for y in years], q1_values, width=width, label="Q1 (25th Percentile)", color="orange")

# Labels and Titles
plt.xlabel("Year")
plt.ylabel("Base Load (kWh/day)")
plt.title("Estimated Daily Base Load")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()

# %%
# CALCULATE ELECTRICITY USED FOR HEATING
# CALCULATE DAILY TIMESERIES

## Used electricity for heating in kWh, best estimate
heating_electricity_hourly = dict()
for year, hourly_energy in total_electricity_hourly.items():
    hourly_base_load = DAILY_BASE_LOAD / 24
    hourly_heating = hourly_energy['electricity_kWh'] - hourly_base_load
    hourly_heating = hourly_heating.clip(lower=0)
    heating_electricity_hourly[year] = pd.DataFrame({
        'electricity_kWh': hourly_heating
    })

## Delta temperature i.e. difference between outside and inside temperature
delta_temperature_hourly = dict()
for year, temps in total_temperature_hourly.items():
    delta_temp = INTERNAL_TEMPERATURE - temps["temperature_C"]
    delta_temperature_hourly[year] = pd.DataFrame({
        'temperature_C': delta_temp
    })
    delta_temperature_hourly[year].index = temps.index
print("\nTime series: delta_temperature_hourly")
print(delta_temperature_hourly["2020"])

## Heating efficiency i.e. heating electricity normalised by delta temperature
## kWh / K
heating_efficiency_hourly = dict()
for year in heating_electricity_hourly.keys():
    delta_T = INTERNAL_TEMPERATURE - total_temperature_hourly[year]['temperature_C']
    heating_efficiency = heating_electricity_hourly[year]['electricity_kWh'] / delt6a_T
    heating_efficiency_hourly[year] = pd.DataFrame({
        'efficiency_kWh/K': heating_efficiency
    })
    heating_efficiency_hourly[year].index = delta_temperature_hourly[year].index
print("\nTime series: heating_efficiency_hourly")
print(heating_efficiency_hourly["2020"])

# Resample electricity usage (sum per day)
heating_electricity_daily = dict()
for year, df in heating_electricity_hourly.items():
    heating_electricity = df.resample('D').sum()
    heating_electricity_daily[year] = pd.DataFrame({
        'electricity_kWh': heating_electricity['electricity_kWh']
    })

print("\nTime series: heating_electricity_daily")
print(heating_electricity_daily["2020"])

# Resample electricity usage (sum per day)
total_electricity_daily = dict()
for year, df in total_electricity_hourly.items():
    total_electricity = df.resample('D').sum()
    total_electricity_daily[year] = pd.DataFrame({
        'electricity_kWh': total_electricity['electricity_kWh']
    })
print("\nTime series: total_electricity_daily")
print(total_electricity_daily["2020"])

# Resample temperature (average per day)
total_temperature_daily = dict()
for year, df in total_temperature_hourly.items():
    total_temperature = df.resample('D').mean()
    total_temperature_daily[year] = pd.DataFrame({
        'temperature_C': total_temperature['temperature_C']
    })
print("\nTime series: total_temperature_daily")
print(total_temperature_daily["2020"])

# Recalculate heating efficiency with daily data
heating_efficiency_daily = dict()
for year in heating_electricity_daily.keys():
    delta_T = INTERNAL_TEMPERATURE - total_temperature_daily[year]['temperature_C']
    heating_efficiency = heating_electricity_daily[year]['electricity_kWh'] / delta_T  
    heating_efficiency_daily[year] = pd.DataFrame({
        'efficiency_kWh/K': heating_efficiency
    })
print("\nTime series: heating_efficiency_daily")
print(heating_efficiency_daily["2020"])

# %%
# PLOT HEATING EFFICIENCY

for year in sorted(heating_electricity_daily.keys()):
    timestamps = heating_electricity_daily[year].index  # Daily timestamps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- FIGURE 1: Electricity Usage & Temperature ---
    ax1.bar(timestamps, heating_electricity_daily[year]['electricity_kWh'], color="red", label="Heating Electricity")
    ax1.bar(timestamps, (total_electricity_daily[year]['electricity_kWh'] - heating_electricity_daily[year]['electricity_kWh']),
            color="blue", label="Base Load", alpha=0.7)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(timestamps, total_temperature_daily[year]['temperature_C'], color="black", linestyle="dashed", label="Temperature")
    
    # Set static y-axis limits
    ax1.set_ylim(0, 120)  # Electricity: 0 to 120 kWh
    ax1_twin.set_ylim(-25, 25)  # Temperature: -25 to 25°C

    ax1.set_ylabel("Electricity (kWh)")
    ax1_twin.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.set_title(f"Electricity Usage & Temperature - {year}")

    # --- FIGURE 2: Heating Efficiency ---
    ax2.bar(timestamps, heating_efficiency_daily[year]['efficiency_kWh/K'], color="green", width=0.8)

    # Set static y-axis limit for efficiency
    ax2.set_ylim(0, 10)  # Efficiency: 0 to 20 kWh/°C

    ax2.set_ylabel("Heating Efficiency (kWh/°C)")
    ax2.set_xlabel("Time")
    ax2.set_title(f"Heating Efficiency - {year}")

    plt.tight_layout()
    plt.show()


# %%
