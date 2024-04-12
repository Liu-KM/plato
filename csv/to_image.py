import pandas as pd
import matplotlib.pyplot as plt
import sys

file_name = sys.argv[1]
df = pd.read_csv(f"./{file_name}")

# Plotting again with the data now being read as if from CSV files
plt.figure(figsize=(10, 6))

# Plot for CSV 1
plt.plot(df['round'], df['accuracy'], marker='o', linestyle='-', color='blue')

# Adding titles and labels
plt.title(f'Accuracy_{file_name}')
plt.xlabel('Round')
plt.ylabel('Accuracy')


# Show plot
plt.grid(True)
plt.savefig(f'{file_name.split(".")[0]}.png', format='png', dpi=300) 