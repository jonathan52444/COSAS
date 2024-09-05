
######
# This is just an example of the code 
# that may be used to resize the image to the original dimensions
######

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import torch

def read_mha_with_header(path):
    # Read the .mha file
    itk_image = sitk.ReadImage(path)
    
    # Get the image array
    image_array = sitk.GetArrayFromImage(itk_image)
    
    # Get the original size (note: SimpleITK uses x,y,z order)
    original_size = itk_image.GetSize()
    
    return image_array, original_size

def process_and_resize_output(net, image, original_size, patch_size):
    h, w, _ = image.shape
    outputs = net(image, True, patch_size[0])
    output_masks = outputs['masks']
    out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
    prediction = out.cpu().detach().numpy()
    
    # Resize to original dimensions
    if prediction.shape != (original_size[1], original_size[0]):  # Note the order swap
        prediction = zoom(prediction, (original_size[1] / prediction.shape[0], 
                                       original_size[0] / prediction.shape[1]), order=0)
    
    return prediction

def write_mha(path, array):
    image = sitk.GetImageFromArray(array)
    sitk.WriteImage(image, path, useCompression=False)

# Main processing loop
for filename in os.listdir(input_root):
    if filename.endswith('.mha'):
        output_path = f'{output_root}/{filename}'
        try:
            input_path = os.path.join(input_root, filename)
            image, original_size = read_mha_with_header(input_path)
            
            prediction = process_and_resize_output(net, image, original_size, patch_size)
            
            write_mha(output_path, prediction.astype('uint8'))
            print(f"Processed and saved: {output_path}")
        except Exception as error:
            print(f"Error processing {filename}: {str(error)}")