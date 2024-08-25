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

"""import matplotlib.pyplot as plt

# Data
sectors = ['Agriculture', 'Healthcare', 'Finance']
gdp_contribution = [[10, 15, 20, 25, 30, 35],
                    [8, 12, 16, 20, 25, 30],
                    [12, 18, 24, 30, 38, 45]]

# Create stacked bar chart
plt.figure(figsize=(10, 6))
plt.bar(sectors, gdp_contribution[0], label='Agriculture')
plt.bar(sectors, gdp_contribution[1], label='Healthcare', bottom=gdp_contribution[0])
plt.bar(sectors, gdp_contribution[2], label='Finance', bottom=[x + y for x, y in zip(gdp_contribution[0], gdp_contribution[1])])
plt.xlabel('Sector')
plt.ylabel('GDP Contribution (Billion USD)')
plt.title('Projected GDP Contribution of AI Across Key Sectors')
plt.legend()
plt.show()"""

import matplotlib.pyplot as plt
import numpy as np

# Data: Years and GDP contribution for each sector
years = ['2020', '2021', '2022', '2023', '2024', '2025']
agriculture = [10, 15, 20, 25, 30, 35]  # Length: 6
healthcare = [8, 12, 16, 20, 25, 30]    # Length: 6
finance = [12, 18, 24, 30, 38, 45]      # Length: 6

x = np.arange(len(years))  # Positions for the groups
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(12, 7))

# Create bars for each sector with proper x-offsets
rects1 = ax.bar(x - width, agriculture, width, label='Agriculture')
rects2 = ax.bar(x, healthcare, width, label='Healthcare')
rects3 = ax.bar(x + width, finance, width, label='Finance')

# Add labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Year')
ax.set_ylabel('GDP Contribution (Billion USD)')
ax.set_title('Projected GDP Contribution of AI Across Key Sectors (2020-2025)')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()

# Attach a text label above each bar in rects
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()

# Show plot
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data
challenges = ['Data Privacy', 'Cybersecurity', 'Digital Divide']
scores = [8, 7, 9]

# Create the radar chart
import numpy as np

# Assuming challenges is a list or array-like object
angles = np.linspace(0, 2 * np.pi, len(challenges))
plt.polar(angles, scores, marker='o', linestyle='-', color='blue')

# Add labels and title
plt.xticks(angles, challenges)
plt.title('Severity of Challenges in AI Adoption in India')

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
plt.savefig('gdp_contribution_ai.png')
plt.show()

