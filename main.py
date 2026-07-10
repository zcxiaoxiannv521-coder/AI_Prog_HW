from src.m1_data_processing import data_processing
from src.m2_visualization import visualization
from src.m3_modeling import modeling
from src.m4_qa_system import qa_system

def main():

    df = data_processing()
    visualization(df)
    modeling(df)
    qa_system(df)

if __name__ == "__main__":
    main()