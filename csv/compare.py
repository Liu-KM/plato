import pandas as pd
import matplotlib.pyplot as plt
import sys

# Convert the CSV strings to DataFrames

if len(sys.argv)>=3:
    df1 = pd.read_csv(f"./{sys.argv[1]}")
    df2 = pd.read_csv(f"./{sys.argv[2]}")
else:   
    df1 = pd.read_csv("./20_100client_noniid_svd.csv")
    df2 = pd.read_csv("./4154566.csv")

# Plotting again with the data now being read as if from CSV files
plt.figure(figsize=(10, 6))

# Plot for CSV 1
plt.plot(df1['round'], df1['accuracy'], marker='o', linestyle='-', color='blue', label='Mine Accuracy')

# Plot for CSV 2
plt.plot(df2['round'], df2['accuracy'], marker='o', linestyle='--', color='red', label='Original Accuracy')

# Adding titles and labels
plt.title('Accuracy_20/100_Client_noniid')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()

# Show plot
plt.grid(True)
if len(sys.argv)>=4:
    plt.savefig(f'{sys.argv[3]}.png', format='png', dpi=300) 
else:
    plt.savefig('20_100client_svd.png', format='png', dpi=300) 
plt.show()