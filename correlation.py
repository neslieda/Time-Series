import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

merged_data = pd.read_csv("merged_test_data.csv")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_data[['count', 'prcp', 'tavg', 'pres', 'wspd', 'wdir']])
scaled_df = pd.DataFrame(scaled_features, columns=['count', 'prcp', 'tavg', 'pres', 'wspd', 'wdir'])

correlation = scaled_df.corr()
print("Korelasyon Matrisinin Ölçeklenmiş Hali:\n", correlation)

plt.figure(figsize=(14, 10))

for i, col in enumerate(['prcp', 'tavg', 'pres', 'wspd', 'wdir']):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=merged_data['count'], y=merged_data[col])
    plt.title(f'count vs {col}')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Korelasyon Matrisinin Isı Haritası')
plt.show()
