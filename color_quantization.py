import cv2
import numpy as np

def color_quantization(image, k=8):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Convert to float32
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    
    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]
    
    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image

# Read input image
input_image = cv2.imread("C:/Users/prolo/OneDrive/Desktop/4-1/LAB/Image Processing Lab/tiger.jpg")

if input_image is not None:
    # Perform color quantization
    quantized_image = color_quantization(input_image, k=8)

    # Display the original and quantized images
    cv2.imshow("Original Image", input_image)
    cv2.imshow("Quantized Image", quantized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the input image.")

