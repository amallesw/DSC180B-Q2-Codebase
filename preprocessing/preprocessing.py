import os
import pandas as pd
import numpy as np
import h5py
import argparse
import re

def preprocess_pixels(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Converts the 'pixels' column in the DataFrame from strings to NumPy arrays
    and normalizes the pixel values to be between 0 and 1. The shape of the array
    is determined by the dataset_name.
    
    Parameters:
    - data: DataFrame containing a 'pixels' column with pixel data in string format.
    - dataset_name: The name of the dataset, which determines the reshaping.
    
    Returns:
    - DataFrame with the 'pixels' column containing normalized NumPy arrays.
    """
    def process_pixel(pixel):
        pixel_array = np.array(pixel.split(), dtype=np.uint8)
        if dataset_name == "dartmouth":
            pixel_array = pixel_array.reshape((3, 48, 48))
        elif dataset_name == "fer":
            pixel_array = pixel_array.reshape((48, 48))
        pixel_array = pixel_array / 255.0
        return pixel_array

    new_data = data.copy()
    new_data['pixels'] = new_data['pixels'].apply(process_pixel)
    return new_data

def save_preprocessed_data(images, labels, file_path):
    """
    Saves the preprocessed images and labels into an HDF5 file.
    
    Parameters:
    - images: NumPy array of images.
    - labels: NumPy array of labels.
    - file_path: The file path to save the HDF5 file.
    """
    with h5py.File(file_path, "w") as h5f:
        h5f.create_dataset("images", data=images)
        h5f.create_dataset("labels", data=labels)
        
    print(f"Preprocessing complete. Data saved to: {file_path}")
    
def reconstruct_csvs(dataset_name: str, data_path:str) -> None:
    """
    Reconstructs FER and Dartmouth csvs based on chunked csvs
    
    Parameters:
    - dataset_name: (str) "fer" or "dartmouth"
    - data_path: (str) filepath of full dataframe
    """
    chunks = []
    input_folder = f"data/{dataset_name}"
    
    filenames = [filename for filename in os.listdir(input_folder) if filename.endswith('.csv') and "chunk" in filename]
    sorted_filenames = sorted(filenames, key=lambda x: int(re.search(r"chunk_(\d+).csv", x).group(1)))

    for filename in sorted_filenames:
        chunk = pd.read_csv(f'{input_folder}/{filename}')
        chunks.append(chunk)

    original_df = pd.concat(chunks, ignore_index=True)
    original_df.to_csv(data_path, index=False)
    print(f"Saved reconstructed DataFrame to {data_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset images.")
    parser.add_argument("--dataset_name", type=str, choices=["fer", "dartmouth"],
                        required=True, help="The name of the dataset to preprocess.")
    args = parser.parse_args()

    data_path = f"data/{args.dataset_name}/{args.dataset_name}48x48.csv"
    hdf5_file_path = f"data/{args.dataset_name}/{args.dataset_name}_preprocessed.h5"
    
    # Construct full DataFrame from chunks
    reconstruct_csvs(args.dataset_name, data_path)

    # Load the data
    df = pd.read_csv(data_path)
    new_df = preprocess_pixels(df, args.dataset_name)

    # Prepare the data for saving
    image_data = np.stack(new_df["pixels"].values)
    label_data = np.array(new_df["emotion"])

    # Save the preprocessed data
    save_preprocessed_data(image_data, label_data, hdf5_file_path)

if __name__ == "__main__":
    main()
