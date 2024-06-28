import pandas as pd

# Specify the path to your CSV file
csv_file = 'LSA64_60fps.csv'

# Read the CSV file into a dataframe
df = pd.read_csv(csv_file)
counter = 0
# Print the dataframe
for column in df.loc[0]:
    counter+=1
print(counter)