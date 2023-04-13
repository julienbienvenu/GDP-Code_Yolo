import pandas as pd

# Load the .xlsx file
xlsx_file = pd.read_excel('carrier.xlsx')

# Convert the data to a JSON object
json_data = xlsx_file.to_json(orient='records')

# Write the JSON data to a file
with open('carrier.json', 'w') as json_file:
    json_file.write(json_data)