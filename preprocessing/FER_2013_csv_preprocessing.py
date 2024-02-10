import os
import pandas as pd 
import numpy as np
import h5py

def preprocess_pixels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'pixels' column in the DataFrame from strings to 48x48 NumPy arrays
    and normalizes the pixel values to be between 0 and 1.
    
    Parameters:
    - data: pd.DataFrame containing a 'pixels' column with pixel data in string format.
    
    Returns:
    - A new DataFrame with the 'pixels' column containing normalized NumPy arrays.
    """
    
    def process_pixel(pixel):
        image_array = np.fromstring(pixel, sep=' ', dtype=np.float32).reshape(48, 48)
        normalized_image_array = image_array / 255.0
        return normalized_image_array
    
    new_data = data.copy()
    new_data['pixels'] = new_data['pixels'].apply(process_pixel)
    return new_data

def main():
    df = pd.read_csv("../data/fer2013.csv")
    new_df = preprocess_pixels(df)

    hdf5_file = "../data/fer_2013_processed.h5"
    image_data = np.stack(new_df["pixels"].values)
    label_data = np.array(new_df["emotion"])

    with h5py.File(hdf5_file, "w") as h5f:
        h5f.create_dataset("images", data=image_data)
        h5f.create_dataset("labels", data=label_data)
        
    print(f"Processed data saved in {hdf5_file}")

if __name__ == "__main__":
    main()