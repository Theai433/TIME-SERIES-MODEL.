## TIME-SERIES-MODEL.
## INTRODUCTION.
Time series modeling is a powerful and widely used technique in statistics, data science, and machine learning. It involves analyzing time-based data to understand patterns, trends, and relationships within the data. The main objective of time series modeling is to make accurate predictions and forecasts based on historical observations. This comprehensive guide to time series modeling will cover the fundamental concepts, various techniques, applications, and best practices to help you understand and implement time series modeling in real-world situations.

## QUESTION.
Using the Craigslist Vehicles Dataset available on Kaggle (https://www.kaggle.com/datasets/mbaabuharun/craigslist-vehicles), we'd like you to create a Time-Series model.

## WHAT IS A TIME SERIES MODEL?
A time series model is a set of data points ordered in time, where time is the independent variable. These models are used to analyze and forecast the future. Time series data can be univariate (consisting of a single variable) or multivariate (consisting of multiple variables).
This includes stationary series, random walks, the Rho Coefficient, Dickey Fuller Test of Stationarity.

## Components of Time Series Data.
There are four primary components of time series data:

a. Trend: The long-term movement or direction of the data.

b. Seasonality: Regular fluctuations that repeat over a fixed period, such as daily or yearly.
 
c. Cyclic Patterns: Irregular fluctuations that do not follow a fixed pattern.

d. Random Noise: Unpredictable variations in the data that cannot be attributed to any specific pattern or trend.

## SOLUTION.
## Here's how I would do it:

## STEP 1: 
I'LL tart by addressing missing values in the dataset.I'll handle this by filling in missing values with "the median" for numerical columns and "the model" for categorical columns.
I'll then ensure that the data types of the columns are appropriate. Specifically, I'll make sure to convert the 'posting_date' column to a datetime data type.
I'll then Utilize the 'posting_date' column to create a datetime index for the dataset. This will facilitate the analysis of temporal patterns.With clean data,I'll then explore it using various visualizations and statistical analysis techniques. This step is crucial for understanding temporal patterns, identifying seasonal trends, and analyzing demand-supply dynamics by region and vehicle type.
I'll then use this dataset to build a time-series model to facilitate the creation of a time-series chart that represents the number of available vehicles over time, filtered by specific criteria such as region, vehicle type, etc. This will aid in understanding regional demand-supply dynamics, seasonal trends, and other relevant insights.

## STEP 2: 
I will start by importing the necessary python libraries

``` python
import pandas as pd
import matplotlib.pyplot as plt

# data_path = "data/craigslist_vehicles.csv"
data = pd.read_csv(data_path)

data = {
    'id': [362773, 362712, 362722, 362771, 362710],
    'url': [
        'https://abilene.craigslist.org/ctd/d/abilene-2002-bmw-x5/7307679724.html',
        'https://abilene.craigslist.org/ctd/d/abilene-2002-bmw-x5/7311833696.html',
        'https://abilene.craigslist.org/ctd/d/abilene-2006-toyota-camry-le/7311441996.html',
        'https://abilene.craigslist.org/ctd/d/abilene-2008-ford-expedition/7307680715.html',
        'https://abilene.craigslist.org/ctd/d/abilene-2008-ford-expedition/7311834578.html'
    ],
    'region': ['abilene', 'abilene', 'abilene', 'abilene', 'abilene'],
    'region_url': [
        'https://abilene.craigslist.org',
        'https://abilene.craigslist.org',
        'https://abilene.craigslist.org',
        'https://abilene.craigslist.org',
        'https://abilene.craigslist.org'
    ],
    'price': [4500, 4500, 4900, 6500, 6500],
    'year': [2002.0, 2002.0, 2006.0, 2008.0, 2008.0],
    'manufacturer': ['bmw', 'bmw', 'toyota', 'ford', 'ford'],
    'model': ['x5', 'x5', 'camry', 'expedition', 'expedition'],
    'condition': ['excellent', None, 'excellent', 'good', 'excellent']
# Add other columns as needed
    'posting_date': [
        '2021-04-16 00:00:00+00:00',
        '2021-04-24 00:00:00+00:00',
        '2021-04-23 00:00:00+00:00',
        '2021-04-16 00:00:00+00:00',
        '2021-04-24 00:00:00+00:00'
     ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# convert 'posting_date' to datetime data type
data['posting_date'] = pd.to_datetime(data['posting_date'],  utc=True)

# Check for missing values
missing_values = df.isnull().sum()

# Fill missing values in numerical columns with mean
numerical_columns = df.select_dtypes(include=['number']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Fill missing values in categorical columns with mode
categorical_columns = df.select_dtypes(exclude=['number']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Forward fill for time-ordered data
# df = df.fillna(method='fill')

# Interpolation for time-ordered data (linear interpolation)
# df = df.interpolate()

# If you decide to remove rows with missing values
# df = df.dropna()

df['posting_date'] = pd.to_datetime(df['posting_date'])

# Set 'posting_date' as the index
df.set_index('posting_date', inplace=True)

# Display the updated DataFrame
print(df)
```
## STEP 3:I'll then group and aggregate the data.
I used the groupby method to group the data by 'posting_date,' 'region,' and 'type.'
I then used the agg method to calculate various summary statistics for the grouped data. You can adjust the aggregation methods according to your analysis requirements. In this dataset, I calculated the average price, found the median year, and chose the first values for other columns.
Finally, I used the reset_index() method to convert the grouped results back into a DataFrame.

``` python
# Group by 'posting_date', 'region', and 'type' and calculate summary statistics
aggregated_data = df.groupby(['posting_date', 'region', 'type']).agg({
    'price': 'mean',  # Calculate the average price
    'year': 'median',  # Find the median year
    'manufacturer': 'first',  # Choose the first manufacturer
    'model': 'first',  # Choose the first model
    'condition': 'first',  # Choose the first condition
    'county': 'first',  # Choose the first county
    'state': 'first',  # Choose the first state
    'lat': 'mean',  # Calculate the average latitude
    'long': 'mean',  # Calculate the average longitude
}).reset_index()

# Show the aggregated data
print(aggregated_data)
```
## STEP 4: Analyze the aggregated data.
Since I now had aggregated the data, I analyzed temporal patterns, seasonal trends, and demand-supply dynamics. Some potential analyses I performed are:

Temporal Patterns: plotted the average price over time to identify any trends or patterns.
Seasonal Trends: Used time series decomposition techniques (e.g., seasonal decomposition of time series or seasonal subseries plots) to identify seasonal patterns.
Demand-Supply Dynamics: Analyzed how the price and the number of listings for different vehicle types change over time.
Here's an example of how to visualize the average price over time:

``` python
# Plot the average price over time
plt.figure(figsize=(12, 6))
for vehicle_type in aggregated_data['type'].unique():
    data_type = aggregated_data[aggregated_data['type'] == vehicle_type]
    plt.plot(data_type['posting_date'], data_type['price'], label=vehicle_type)

plt.xlabel('Posting Date')
plt.ylabel('Average Price')
plt.title('Average Price Over Time for Different Vehicle Types')
plt.legend()
plt.grid(True)
plt.show()

# Group data by day and count the number of listings
daily_counts = df.groupby(df.index.date).size()
```
## Step 5: Customize the layout for the time-frequency graph.
``` python
fig_freq.update_layout(
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-')
plt.title('Number of Listings Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Listings')
plt.grid(True)
plt.xticks(rotation=45)
)
```
## STEP 6: Show the time-frequency graph
``` python
fig_freq.show()
```
## STEP 7: Perform seasonal decomposition
``` python
#Perform seasonal decomposition 
result = seasonal_decompose(df['price'], model='additive', period=365)  # Adjust 'period' as needed

# Create a new DataFrame to store decomposition components
decomposition_df = pd.DataFrame({
    'Trend': result.trend,
    'Seasonal': result.seasonal,
    'Residual': result.resid,
})

# Reset the index to include 'posting_date'
decomposition_df['posting_date'] = df.index

# Reset the index for plotting
result_df = pd.DataFrame({
    'Trend': result.trend,
    'Seasonal': result.seasonal,
    'Residual': result.resid,
})

# Plot the seasonal decomposition
plt.figure(figsize=(12, 6))
plt.subplot(411)
plt.plot(result_df['Trend'], label='Trend')
plt.legend()

plt.subplot(412)
plt.plot(result_df['Seasonal'], label='Seasonal')
plt.legend()

plt.subplot(413)
plt.plot(result_df['Residual'], label='Residual')
plt.legend()

plt.tight_layout()
```
## STEP 7: Customize the layout
``` python
# Customize the layout
plt.suptitle('Seasonal Decomposition of Price', fontsize=16)
plt.subplots_adjust(top=0.9)

# show the plot
fig_decompose.show()
```
## CONCLUSION.
Time series modeling is a versatile and powerful technique for analyzing and forecasting time-based data. By understanding the fundamental concepts, techniques, applications, and best practices, you can effectively leverage time series modeling to make data-driven decisions and drive value in your organization. As you embark on your time series modeling journey, remember to stay updated with the latest advancements and trends in the field to ensure that your models remain accurate, relevant, and impactful.
