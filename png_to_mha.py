import SimpleITK as sitk
import os

def convert_png_to_mha(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all PNG files in the input directory
    png_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    for png_file in png_files:
        # Full path to the input PNG file
        input_path = os.path.join(input_dir, png_file)

        # Read the PNG image
        image = sitk.ReadImage(input_path)

        # Full path to the output MHA file
        output_file_name = os.path.splitext(png_file)[0] + '.mha'
        output_path = os.path.join(output_dir, output_file_name)

        # Save the image as an MHA file
        sitk.WriteImage(image, output_path)

        print(f"Converted {input_path} to {output_path}")

# Specify the input and output directories
input_directory = '/Users/sirbucks/Desktop/cosas-algorithm-submition/data_2/image'
output_directory = '/Users/sirbucks/Desktop/cosas-algorithm-submition/output_mha'

# Call the conversion function
convert_png_to_mha(input_directory, output_directory)
