import cv2

import numpy as np

from PIL import Image



def process_to_size(input_path, target_size):

    """

    Process image to be exactly target_size x target_size.

    If image dimensions are already correct, keeps them.

    Otherwise scales and crops to achieve exact target dimensions.

    

    Args:

        input_path: Path to input image

        output_path: Path to save processed image

        target_size: Desired width/height of square output image

    """

    # Open image

    img = Image.open(input_path).convert('RGB')

    width, height = img.size

    

    # If already target size, just save and return

    if width == target_size and height == target_size:

        return img

        

    # Calculate scaling ratio to make smallest dimension match target

    ratio = target_size / min(width, height)

    

    # Only scale if we're significantly off from target size

    if True:

        scaled_width = int(width * ratio)

        scaled_height = int(height * ratio)

        

        # Scale the image

        if ratio > 1:  # Need to upscale

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            initial_upscale = cv2.resize(img_cv, None, fx=2, fy=2, 

                                       interpolation=cv2.INTER_CUBIC)

            denoised = cv2.bilateralFilter(initial_upscale, 9, 75, 75)

            scaled_image = cv2.resize(denoised, (scaled_width, scaled_height), 

                                    interpolation=cv2.INTER_LANCZOS4)

            scaled_img = Image.fromarray(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))

        else:  # Need to downscale

            scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

    else:

        scaled_img = img

        scaled_width, scaled_height = width, height

    

    # Only crop if necessary to achieve target size

    if scaled_width != target_size or scaled_height != target_size:

        left = (scaled_width - target_size) // 2

        top = (scaled_height - target_size) // 2

        result = scaled_img.crop((left, top, left + target_size, top + target_size))

    else:

        result = scaled_img

    

    # Verify final dimensions

    final_width, final_height = result.size

    if final_width != target_size or final_height != target_size:

        # Force exact dimensions if somehow still off

        result = result.resize((target_size, target_size), Image.Resampling.LANCZOS)

    return result
