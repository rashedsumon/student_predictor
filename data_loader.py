import os
import glob
import pandas as pd
import kagglehub

def load_student_data():
    """
    Downloads the dataset via kagglehub and loads the exams.csv file.
    Returns a pandas DataFrame.
    """
    print("Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("rkiattisak/student-performance-in-mathematics")
    print(f"Path to dataset files: {path}")
    
    # Look for the exams.csv file inside the downloaded directory
    csv_path = os.path.join(path, "exams.csv")
    
    # Fallback check if path structure varies slightly
    if not os.path.exists(csv_path):
        search_path = glob.glob(os.path.join(path, "**", "exams.csv"), recursive=True)
        if search_path:
            csv_path = search_path[0]
        else:
            raise FileNotFoundError("Could not find exams.csv in the downloaded files.")
            
    df = pd.read_csv(csv_path)
    return df

if __name__ == "__main__":
    # Test execution
    data = load_student_data()
    print("Successfully loaded data! Sample rows:")
    print(data.head(2))