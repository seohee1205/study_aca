import pandas as pd
import os

# Set directories where CSV files are located
csv_dirs = ['d:/study_data/_data/aif/초미세먼지/META',  'd:/study_data/_data/aif/초미세먼지/TEST_AWS',
            'd:/study_data/_data/aif/초미세먼지/TEST_INPUT', 'd:/study_data/_data/aif/초미세먼지/TRAIN',
            'd:/study_data/_data/aif/초미세먼지/TRAIN_AWS']

# Loop through all CSV files in directories
for csv_dir in csv_dirs:
    # Create empty list to hold dataframes
    df_list = []
    
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            # Read CSV file into dataframe
            df = pd.read_csv(os.path.join(csv_dir, file))
            
            # Do any necessary data cleaning or manipulation here
            # ...
            
            # Append dataframe to list
            df_list.append(df)
    
    # Concatenate all dataframes in list into one dataframe
    combined_df = pd.concat(df_list)
    
    # Save combined dataframe to CSV file
    if csv_dir == 'd:/study_data/_data/aif/초미세먼지/META':
        combined_df.to_csv('d:/study_data/_data/aif/초미세먼지/META_all.csv', index=False)
    elif csv_dir == 'd:/study_data/_data/aif/초미세먼지/TEST_AWS':
        combined_df.to_csv('d:/study_data/_data/aif/초미세먼지/TEST_AWS.csv', index=False)
    elif csv_dir == 'd:/study_data/_data/aif/초미세먼지/TEST_INPUT':
        combined_df.to_csv('d:/study_data/_data/aif/초미세먼지/TEST_INPUT_all.csv', index=False)
    elif csv_dir == 'd:/study_data/_data/aif/초미세먼지/TRAIN':
        combined_df.to_csv('d:/study_data/_data/aif/초미세먼지/TRAIN_all.csv', index=False)
    elif csv_dir == 'd:/study_data/_data/aif/초미세먼지/TRAIN_AWS':
        combined_df.to_csv('d:/study_data/_data/aif/초미세먼지/TRAIN_AWS_all.csv', index=False)            

#############################################################




