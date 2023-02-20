import pandas as pd

#integrate several dataset into one dataset
#input: several datasets
#output: one dataset
datasets_names = ["aggression_parsed_dataset.csv","attack_parsed_dataset.csv","toxicity_parsed_dataset.csv","kaggle_parsed_dataset.csv","twitter_parsed_dataset.csv","youtube_parsed_dataset.csv"]
def merge_dataset(df1,df2):
    for index, row in df2.iterrows():
        if(row['Text'] not in df1['Text'].values):
            df1 = df1.append(row)
        else:
            if(row['class_label'] == 1):
                df1.loc[df1['Text'] == row['Text'], 'class_label'] = row['class_label']
    return df1
def format_dataset(df):
    df = df.dropna()
    #rename oh_label to class_label
    df = df.rename(columns={'oh_label':'class_label'})
    df = df.reset_index(drop=True)
    return df[["Text","class_label"]]


def test():
    #count number of 1 in each dataset
    for i in range(len(datasets_names)):
        df = pd.read_csv("dataset/"+datasets_names[i])
        df = format_dataset(df)
        df = df.head(100)
        #check if each row having 1 is in the merged dataset
        for index, row in df.iterrows():
            if(row['class_label'] == 1):
                if(row['Text'] not in df['Text'].values):
                    print("Error: ",row['Text']," not in merged dataset")
                    return
        print("Tested dataset: ",datasets_names[i])

def main():
    df = pd.read_csv("dataset/"+datasets_names[0])
    df = format_dataset(df)
    for i in range(1,len(datasets_names)):
        df2 = pd.read_csv("dataset/"+datasets_names[i])
        df2 = format_dataset(df2)
        print("Merging dataset: ",datasets_names[i])
        merge_dataset(df,df2)
        print("Merged"+ datasets_names[i]+ "dataset size: ",df.shape)
    df.to_csv("merged_dataset.csv",index=False)
    print("Merged dataset size: ",df.shape)

if __name__ == "__main__":
    main()