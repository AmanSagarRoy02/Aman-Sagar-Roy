import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

file_path = r"C:\Users\amans\OneDrive\Documents\Netflix Dataset.xlsx"
netflix_data = pd.read_excel(file_path, sheet_name='Sheet1')
netflix_data['date_added'] = pd.to_datetime(netflix_data['date_added'])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
fig.suptitle('Netflix Content Dashboard', fontsize=16)
content_type_counts = netflix_data['type'].value_counts()
axes[0, 0].pie(content_type_counts, labels=content_type_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
axes[0, 0].set_title('Distribution of Content Types')
additions_over_time = netflix_data['date_added'].dt.year.value_counts().sort_index()
axes[0, 1].plot(additions_over_time.index, additions_over_time.values, marker='o', linestyle='-', color='b')
axes[0, 1].set_title('Content Additions Over Time')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Number of Shows Added')
genre_counts = Counter(", ".join(netflix_data['listed_in']).split(", "))
top_genres = dict(genre_counts.most_common(10))
axes[1, 0].barh(list(top_genres.keys()), list(top_genres.values()), color='green')
axes[1, 0].set_title('Top Genres')
axes[1, 0].set_xlabel('Number of Shows')
country_counts = netflix_data['country'].value_counts().head(10)
axes[1, 1].barh(country_counts.index, country_counts.values, color='purple')
axes[1, 1].set_title('Top 10 Countries by Number of Shows')
axes[1, 1].set_xlabel('Number of Shows')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
