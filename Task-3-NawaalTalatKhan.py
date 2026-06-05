# -*- coding: utf-8 -*-
"""Project3_SQL Data Analysis

Original file is located at
    https://colab.research.google.com/drive/1lxyh0jx550-apIM1xsp9WkCehRamKb7-
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv(r"/content/sample_data/Teen_Mental_Health_Dataset.csv")

df.head()

df = df.drop_duplicates()
null = df.isnull().sum()
df = df.drop(columns=["platform_usage", "gender"])
print("\nDataFrame head after dropping duplicates and columns:")
print(df.head())

!pip install scikit-learn
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['social_interaction_level'] = le.fit_transform(df['social_interaction_level'])

df.head()

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/content/sample_data/Teen_Mental_Health_Dataset.csv")
df = df.drop_duplicates()
df = df.drop(columns=["platform_usage", "gender"])

le = LabelEncoder()
df['social_interaction_level'] = le.fit_transform(df['social_interaction_level'])

df_stats = df.describe()
display(df_stats)

"""## Understanding Trends and Outliers

To find patterns (trends) and unusual data points (outliers) in our data, we used a few visual tools:

*   **Histograms:** These charts show how often different values appear for each numerical feature, helping us see the overall spread and common ranges.

*   **Box Plots:** These plots highlight the middle range of the data and clearly mark individual data points that are far away from the rest. These isolated points are our potential outliers.

*   **Correlation Heatmap:** This colored grid shows how strongly different features move together. A bright color means they change in a similar way (positive trend), a dark color means they change in opposite ways (negative trend), and a neutral color means they don't seem related. This helps us spot relationships, for example, if more screen time is linked to higher stress.
"""

import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = ['age', 'daily_social_media_hours', 'sleep_hours',
                  'screen_time_before_sleep', 'academic_performance', 'physical_activity',
                  'stress_level', 'anxiety_level', 'addiction_level', 'depression_label']

plt.figure(figsize=(18, 15))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 15))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

"""**From my analysis, here's a quick summary:**

**Distributions:** Most features, like social media use, often lean towards lower values but have some individuals with very high usage.

**Outliers:** Unusual, extreme values are present in areas like daily social media hours and screen time before sleep, indicating some teens have much higher usage than others.

**Correlations:** There are clear links between factors. For example, higher stress, anxiety, and addiction levels tend to move together. Also, less sleep might be connected to higher stress.

## Query Data with SQL

I used SQL commands directly on my pandas DataFrame. To do this, we'll use a library called `pandasql`. First, I installed it.
"""

!pip install pandasql

from pandasql import sqldf

"""To make running SQL queries even easier, I've created a small helper function called `pysqldf`. This function lets us write our SQL code and run it directly against our DataFrame, which is named `df`."""

def pysqldf(q):
    """Run a SQL query on the 'df' DataFrame."""
    return sqldf(q, {'df': df})

"""### 1. Basic SELECT Query: Picking Columns and Rows

Think of `SELECT` as choosing which columns you want to see. `LIMIT` helps you see just a few rows at the beginning. Here, we're asking to see the `age`, `daily_social_media_hours`, `sleep_hours`, and `stress_level` for the first 5 entries.
"""

q_select = """
SELECT age, daily_social_media_hours, sleep_hours, stress_level
FROM df
LIMIT 5;
"""
display(pysqldf(q_select))

"""### 2. Using WHERE Clause: Filtering Your Data

The `WHERE` clause is like a filter. It helps you pick only the rows that meet certain conditions. For example, we can find teenagers who are older than 16 (`age > 16`) AND have a `stress_level` of 5 or more (`stress_level >= 5`). We'll only show the first 10 of these.
"""

q_where = """
SELECT age, daily_social_media_hours, stress_level, anxiety_level
FROM df
WHERE age > 16 AND stress_level >= 5
LIMIT 10;
"""
display(pysqldf(q_where))

"""### 3. Using ORDER BY Clause: Arranging Your Results

`ORDER BY` helps you sort your results. You can sort by any column, either from smallest to largest (ascending, or `ASC`) or largest to smallest (descending, or `DESC`). Here, we're looking for the top 10 teenagers with the *highest* `daily_social_media_hours`.
"""

q_order_by = """
SELECT age, daily_social_media_hours, sleep_hours
FROM df
ORDER BY daily_social_media_hours DESC
LIMIT 10;
"""
display(pysqldf(q_order_by))

"""### 4. Using GROUP BY and Aggregations (COUNT, SUM, AVG): Summarizing Data

`GROUP BY` lets you group rows that have the same value in a specific column (like `age`). Then, you can use *aggregation functions* to get summaries for each group:
-   `COUNT(*)`: Tells you how many items are in each group.
-   `AVG()`: Calculates the average value for a column in each group.

Here, we're grouping all teenagers by their `age` and then calculating the total number of teens, their average social media hours, average sleep, and average stress level for each age group.
"""

q_group_by_agg = """
SELECT
    age,
    COUNT(*) AS num_teenagers,
    AVG(daily_social_media_hours) AS avg_social_media_hours,
    AVG(sleep_hours) AS avg_sleep_hours,
    AVG(stress_level) AS avg_stress_level
FROM df
GROUP BY age
ORDER BY age;
"""
display(pysqldf(q_group_by_agg))
