import matplotlib.pyplot as plt
import numpy as np

# Data for visualizations

# 1. AI Impact on GDP Growth (Bar Chart)
sectors = ['Agriculture', 'Healthcare', 'Finance', 'Manufacturing', 'Retail' , 'Education']
contribution_to_gdp = [2.5, 3.2, 4.0, 3.5, 2.8, 3.0]  # Hypothetical values in percentage

# 2. Sector-wise AI Adoption (Pie Chart)
sectors_ai_adoption = ['Agriculture', 'Healthcare', 'Finance', 'Governance', 'Education']
adoption_rates = [20, 25, 20, 15, 20]  # Hypothetical adoption rates in percentage

# 3. Economic Benefits of AI (Line Graph)
years = np.arange(2020, 2031)
productivity_gains = [1.0, 1.2, 1.4, 1.7, 2.0, 2.4, 2.8, 3.3, 3.8, 4.4, 5.0]  # Hypothetical values in trillion USD
cost_reduction = [0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.3, 2.7, 3.2]  # Hypothetical values in trillion USD
market_expansion = [0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.5, 1.9, 2.4, 3.0, 3.7]  # Hypothetical values in trillion USD

# Plotting the visualizations

# 1. Bar Chart: AI Impact on GDP Growth
plt.figure(figsize=(8, 6))
plt.bar(sectors, contribution_to_gdp, color='skyblue')
plt.title('AI Impact on GDP Growth Across Sectors')
plt.xlabel('Sectors')
plt.ylabel('Contribution to GDP Growth (%)')
plt.show()

# 2. Pie Chart: Sector-wise AI Adoption
plt.figure(figsize=(8, 6))
plt.pie(adoption_rates, labels=sectors_ai_adoption, autopct='%1.1f%%', colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'violet'])
plt.title('Sector-wise AI Adoption in India')
plt.show()

# 3. Line Graph: Economic Benefits of AI
plt.figure(figsize=(10, 6))
plt.plot(years, productivity_gains, label='Productivity Gains', marker='o')
plt.plot(years, cost_reduction, label='Cost Reduction', marker='o')
plt.plot(years, market_expansion, label='Market Expansion', marker='o')
plt.title('Projected Economic Benefits of AI in India (2020-2030)')
plt.xlabel('Year')
plt.ylabel('Economic Benefit (Trillion USD)')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np

# Load a map of India using GeoPandas
india_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
india_map = india_map[india_map.name == "India"]

# Simulate some digital divide data for different regions in India
np.random.seed(42)
states = ["Punjab", "Maharashtra", "West Bengal", "Karnataka", "Tamil Nadu", "Rajasthan", "Gujarat", "Uttar Pradesh",
          "Bihar", "Kerala"]
urban_internet_penetration = np.random.uniform(60, 90, len(states))
rural_internet_penetration = np.random.uniform(10, 40, len(states))

# Create a DataFrame to hold this data
digital_divide_data = pd.DataFrame({
    'state': states,
    'urban_internet_penetration': urban_internet_penetration,
    'rural_internet_penetration': rural_internet_penetration
})

# Simulate positions for the states on the map for demonstration (not geographically accurate)
positions = {
    'Punjab': [75.5, 30.9],
    'Maharashtra': [73.8, 19.2],
    'West Bengal': [87.9, 22.8],
    'Karnataka': [76.5, 15.3],
    'Tamil Nadu': [78.7, 11.1],
    'Rajasthan': [73.9, 26.9],
    'Gujarat': [72.6, 22.2],
    'Uttar Pradesh': [80.9, 26.8],
    'Bihar': [85.3, 25.1],
    'Kerala': [76.5, 10.8]
}

# Plotting the digital divide map
fig, ax = plt.subplots(figsize=(10, 15))
india_map.plot(ax=ax, color='lightgrey')

# Plot urban and rural internet penetration as points with different colors
for state, pos in positions.items():
    urban_pen = digital_divide_data[digital_divide_data.state == state]['urban_internet_penetration'].values[0]
    rural_pen = digital_divide_data[digital_divide_data.state == state]['rural_internet_penetration'].values[0]

    ax.scatter(*pos, s=urban_pen * 10, color='blue', alpha=0.5, label='Urban' if state == 'Punjab' else "")
    ax.scatter(*pos, s=rural_pen * 10, color='red', alpha=0.5, label='Rural' if state == 'Punjab' else "")

# Annotating the states
for state, pos in positions.items():
    ax.text(pos[0], pos[1] + 1.5, state, fontsize=10, ha='center')

# Adding labels and legend
plt.title("Digital Divide Across Urban and Rural Areas in India", fontsize=18)
plt.legend(scatterpoints=1, loc='lower left', title="Penetration Level")
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data for AI-driven Startups Growth (2013-2023)
years = list(range(2013, 2024))
startups = [50, 60, 75, 90, 120, 150, 180, 220, 300, 400, 520]  # Sample growth data

# Create a DataFrame
data = pd.DataFrame({
    'Year': years,
    'Number of AI-driven Startups': startups
})

# Plotting the Bar Graph for AI-driven Startups Growth
plt.figure(figsize=(10, 6))
plt.bar(data['Year'], data['Number of AI-driven Startups'], color='skyblue')
plt.title('Growth of AI-driven Startups in India (2013-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Startups', fontsize=14)
plt.xticks(data['Year'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Data
crops = ['Rice', 'Wheat', 'Maize', 'Pulses', 'Vegetables']
yield_increase = [30, 25, 35, 28, 40]

# Create bar chart
plt.bar(crops, yield_increase, color='green')
plt.xlabel('Crops')
plt.ylabel('Yield Increase (%)')
plt.title('Increase in Crop Yields Due to AI-Powered Precision Farming')
plt.show()

import matplotlib.pyplot as plt

# Data
categories = ['Diagnostics', 'Patient Management', 'Telemedicine', 'Drug Discovery']
percentages = [40, 30, 20, 10]

# Create pie chart
plt.pie(percentages, labels=categories, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of AI Applications in Healthcare')
plt.show()

import matplotlib.pyplot as plt

# Data
years = range(2015, 2026)
inclusion_growth = [5, 10, 18, 25, 35, 45, 55, 65, 75, 85, 95]

# Create line graph
plt.plot(years, inclusion_growth, marker='o', linestyle='-', color='blue')
plt.xlabel('Year')
plt.ylabel('Financial Inclusion Growth (%)')
plt.title('Growth of AI-Driven Financial Inclusion')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data
challenges = ['Digital Divide', 'Data Privacy', 'Cybersecurity']
scores = [10, 20, 30]

# Complete the circle by appending the first score and angle at the end
scores += scores[:1]
angles = np.linspace(0, 2 * np.pi, len(challenges), endpoint=False).tolist()
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Plot the data
ax.fill(angles, scores, color='blue', alpha=0.25)  # Filled area
ax.plot(angles, scores, marker='o', linestyle='-', color='blue')

# Add labels and title
ax.set_xticks(angles[:-1])
ax.set_xticklabels(challenges)
plt.title('Severity of Challenges in AI Adoption in India', pad=20)

# Show the plot
plt.show()


# Data Visualization 4: Projected GDP Contribution of AI across Key Sectors

# Years and GDP contribution data
years = ['2020', '2021', '2022', '2023', '2024', '2025']
agriculture = [10, 15, 20, 25, 30, 35]  # GDP contribution in billion USD
healthcare = [8, 12, 16, 20, 25, 30]
finance = [12, 18, 24, 30, 38, 45]

x = np.arange(len(years))  # label locations
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, agriculture, width, label='Agriculture')
rects2 = ax.bar(x, healthcare, width, label='Healthcare')
rects3 = ax.bar(x + width, finance, width, label='Finance')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('GDP Contribution (Billion USD)', fontsize=12)
ax.set_title('Projected GDP Contribution of AI across Key Sectors (2020-2025)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()

# Save the figure
#plt.savefig('gdp_contribution_ai.png')
plt.show()





