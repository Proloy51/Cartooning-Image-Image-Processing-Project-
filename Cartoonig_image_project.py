import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import Tk, Button, ttk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

def gaussian_kernel(sigma_x,kernel_size):
    
    kernel = np.zeros((kernel_size,kernel_size))
    
    h = 1/(2.0*np.pi*sigma_x*sigma_x)
    
    n= kernel_size//2
            
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            p =  ((i**2)+(j**2))/(2*(sigma_x**2))
            kernel[i+n,j+n] = h*np.exp(-p)

    return kernel 


def convolutionGray(img, kernel):
    n = kernel.shape[0] // 2
    img_bordered = cv2.copyMakeBorder(img, top=n, bottom=n, left=n, right=n, borderType=cv2.BORDER_CONSTANT)

    out = np.zeros((img_bordered.shape[0], img_bordered.shape[1], 1))

    for x in range(n, img_bordered.shape[0] - n):
        for y in range(n, img_bordered.shape[1] - n):
            sum = 0
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    if 0 <= x - i < img_bordered.shape[0] and 0 <= y - j < img_bordered.shape[1]:
                        sum += img_bordered[x - i, y - j] * kernel[i + n, j + n]
            out[x, y] = sum

    return out



def convolution(img, kernel):
    n = kernel.shape[0] // 2
    img_bordered = cv2.copyMakeBorder(img, top=n, bottom=n, left=n, right=n, borderType=cv2.BORDER_CONSTANT)
    out = np.zeros_like(img)

    for x in range(n, img_bordered.shape[0] - n):
        for y in range(n, img_bordered.shape[1] - n):
            sum = np.sum(img_bordered[x - n:x + n + 1, y - n:y + n + 1] * kernel)
            out[x - n, y - n] = sum

    return out



def x_derivatives(sigma,kernel_size):
    kernel = np.zeros((kernel_size,kernel_size))
    
    n = kernel_size//2
    
    h = 1/(2.0*math.pi*(sigma**2))
            
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            p = ((i**2)+(j**2))/(2*(sigma**2))
            kernel[i+n,j+n] = (-i/(sigma**2))*h*np.exp(-p)
    
    #print(kernel)
    return kernel


def y_derivatives(sigma,kernel_size):
    kernel = np.zeros((kernel_size,kernel_size))
    n = kernel_size//2
    
    h = 1/(2.0*math.pi*(sigma**2))
            
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            p = ((i**2)+(j**2))/(2*(sigma**2))
            kernel[i+n,j+n] = (-j/(sigma**2))*h*np.exp(-p)
    
    #print(kernel)
    return kernel

def non_max_suppression(img, angle):
    m, n = img.shape[0],img.shape[1]
    out = np.zeros((m, n))
    
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            try:
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q, r = img[i, j + 1], img[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q, r = img[i - 1, j + 1], img[i + 1, j - 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q, r = img[i - 1, j], img[i + 1, j]
                else:
                    q, r = img[i - 1, j - 1], img[i + 1, j + 1]
                if img[i, j] >= q and img[i, j] >= r:
                    out[i, j] = img[i, j]
                else:
                    out[i, j] = 0
            except IndexError as e:
                pass
    return out


def find_threshold(img):
    
    oldThreshold = np.mean(img)
    
    newThreshold = threshold_generator(img,oldThreshold)
    
    while(abs(newThreshold-oldThreshold) > 0.1 ** 6):
        oldThreshold = newThreshold
        newThreshold = threshold_generator(img,oldThreshold)
        
    return newThreshold
    
    
def threshold_generator(img,threshold):
    m,n = img.shape[0],img.shape[1]
    
    sum1 = 0
    sum2 = 0
    n1 = 0
    n2 = 0
    
    for x in range(m):
        for y in range(n):
            if img[x,y]>threshold:
                sum1+=img[x,y]
                n1+=1
            else:
                sum2+=img[x,y]
                n2+=1
    
    highthreshold = sum1/n1
    lowthreshold = sum2/n2
    
    return (highthreshold+lowthreshold)/2

    
def doubleThresholding(img):
    
    threshold = find_threshold(img)
    
    if threshold is None:
        return np.zeros_like(img)
    
    weak = np.uint8(75)
    strong = np.uint8(255)
    
    out = np.zeros(img.shape)
    
    highThreshold = threshold * .5
    lowThreshold = highThreshold * .5

     
    strong_indices = np.where(img >= highThreshold)
    zeros_indices = np.where(img <= lowThreshold)
    weak_indices = np.where((img >= lowThreshold) & (img <= highThreshold))
    
    if len(strong_indices) == 2:
        for i, j in zip(strong_indices[0], strong_indices[1]):
            out[i, j] = strong
    elif len(strong_indices) == 1:
        for idx in strong_indices[0]:
            out[idx] = strong
    
    if len(zeros_indices) == 2:
        for i, j in zip(zeros_indices[0], zeros_indices[1]):
            out[i, j] = 0
    elif len(zeros_indices) == 1:
        for idx in zeros_indices[0]:
            out[idx] = 0
    
    if len(weak_indices) == 2:
        for i, j in zip(weak_indices[0], weak_indices[1]):
            out[i, j] = weak
    elif len(weak_indices) == 1:
        for idx in weak_indices[0]:
            out[idx] = weak
    
    return out


def double_thresholding(img, high_threshold_ratio, low_threshold_ratio):
    high_threshold = np.max(img) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    
    strong_edge = img >= high_threshold
    weak_edge = (img >= low_threshold) & (img < high_threshold)
    
    return strong_edge, weak_edge


def hysteresis(img):
    
    out = img.copy()
    
    weak = 75
    strong =255

    m,n = img.shape[0],img.shape[1]
    
    for i in range(1,m-1):
        for j in range(1,n-1):
            if(out[i,j]==weak):
                 out[i,j] = strong if (out[i-1,j-1]==strong or out[i-1,j]==strong or out[i-1,j+1]==strong or out[i,j-1]==strong or out[i,j+1]==strong or out[i+1,j-1]==strong or out[i+1,j]==strong or out[i+1,j+1]==strong) else 0       
    
    return out


def hysteresis_thresholding(strong_edge, weak_edge):
    m, n = strong_edge.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if weak_edge[i, j]:
                if strong_edge[i - 1:i + 2, j - 1:j + 2].any():
                    strong_edge[i, j] = True
                else:
                    weak_edge[i, j] = False
    return strong_edge


    
def canny_edge_detector(img,sigma,th_high,th_low,kernel_size):
    blurred_img = convolution(img,gaussian_kernel(sigma,kernel_size))
    
    I_x = convolution(blurred_img,x_derivatives(sigma,kernel_size)) 
    I_y = convolution(blurred_img,y_derivatives(sigma,kernel_size))
    
    I_mag = np.sqrt(I_x**2 + I_y**2)
    
    angles = np.arctan2(I_y.copy(),I_x.copy())
    
    nms = non_max_suppression(I_mag,angles)
    
    dbl_thresholded = doubleThresholding(nms)
    
    final_output = hysteresis(dbl_thresholded)
    
    cv2.normalize(blurred_img,blurred_img,0,255,cv2.NORM_MINMAX)
    blurred_img = np.round(blurred_img).astype(np.uint8)
    
    cv2.normalize(I_x,I_x,0,255,cv2.NORM_MINMAX)
    I_x = np.round(I_x).astype(np.uint8)
    
    cv2.normalize(I_y,I_y,0,255,cv2.NORM_MINMAX)
    I_y = np.round(I_y).astype(np.uint8)
    
  #  cv2.normalize(I_mag,I_mag,0,255,cv2.NORM_MINMAX)
    I_mag = np.round(I_mag).astype(np.uint8)
    
    cv2.normalize(nms,nms,0,255,cv2.NORM_MINMAX)
    nms = np.round(nms).astype(np.uint8)
    
    cv2.normalize(dbl_thresholded,dbl_thresholded,0,255,cv2.NORM_MINMAX)
    dbl_thresholded = np.round(dbl_thresholded).astype(np.uint8)
    
    cv2.normalize(final_output,final_output,0,255,cv2.NORM_MINMAX)
    final_output = np.round(final_output).astype(np.uint8)
    
    return final_output
    


def median(img, kernel_size):
    n = kernel_size // 2
    gray_img_bordered = cv2.copyMakeBorder(src=img, top=n, bottom=n, left=n, right=n, borderType=cv2.BORDER_CONSTANT)
    output_median = np.zeros_like(img)
    for i in range(n, img.shape[0] - n):
        for j in range(n, img.shape[1] - n):
            region = gray_img_bordered[i:i + kernel_size, j:j + kernel_size]
            output_median[i, j] = np.median(region)

    return output_median



def negative(img):
    negative_image = np.zeros_like(img)
    m = np.max(img)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            negative_image[i,j] = m-img[i,j]
    return negative_image



def thresholding(img, threshold):
    threshold_image = np.zeros_like(img)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            intensity = img[i, j]
            if intensity > threshold:
                threshold_image[i, j] = 255
            else:
                threshold_image[i, j] = 0
    return threshold_image


def color_quantization(image, k=8):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
     
    return segmented_image 


def halftone_shading(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    halftone = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)
    return halftone



def open_file_and_process(choice):
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    
    if file_path:
        image = cv2.imread(file_path)
        
        if choice == 1:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            median_image = median(gray_image, kernel_size=5)
            edged_image = cv2.Canny(median_image, 75, 150)
            thr = thresholding(edged_image, threshold=50)
            pencil_sketched = negative(thr)
            
            
            '''plt.figure(figsize=(10, 5))

            # Plot original image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            # Plot blurred image
            plt.subplot(1, 2, 2)
            plt.imshow(pencil_sketched)
            plt.title('Pencil Sketched Image')
            plt.axis('off')
                        
            
            plt.tight_layout()
            plt.show()'''
            
            cv2.imshow("Original Image", image)
            cv2.imshow("Pencil Sketched Image", pencil_sketched)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif choice == 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            sigma = 1
            kernel_size = 5

            median_image = median(gray_image, 5)

            high_threshold_ratio = 0.7
            low_threshold_ratio = 0.4

            edged_image = cv2.Canny(median_image, 75, 150)
            edged_image_using_amar_function = canny_edge_detector(median_image, sigma, high_threshold_ratio, low_threshold_ratio, kernel_size)

            thr = thresholding(edged_image, threshold=50)
            thr = thr.astype(np.uint8) 
            thr = negative(thr)

            smooth_image = cv2.bilateralFilter(image, d=19, sigmaColor=75, sigmaSpace=75)
            smooth_image = cv2.resize(smooth_image, (thr.shape[1], thr.shape[0]))

            cartoon = cv2.bitwise_and(smooth_image, smooth_image, mask=thr)

            cv2.imshow("Original Image", image)
            cv2.imshow("Edge Detection", cartoon)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif choice == 3:
            cartoon_image = color_quantization(image)

            cv2.imshow("Original Image", image)
            cv2.imshow("Color Quantization", cartoon_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif choice == 4:
            cartoon_image = halftone_shading(image)

            cv2.imshow("Original Image", image)
            cv2.imshow("Comic Book Style", cartoon_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif choice == 5:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            median_image = median(gray_image, kernel_size=5)
            edged_image = cv2.Canny(median_image, 75, 150)
            thr = thresholding(edged_image, threshold=50)
            
            pencil_sketched = negative(thr)
            
            negative_img = negative(thr)
            smooth_image = cv2.bilateralFilter(image, d=19, sigmaColor=75, sigmaSpace=75)
            cartoon_by_edge_detection = cv2.bitwise_and(smooth_image, smooth_image, mask=negative_img)
            
            cartoon_image = color_quantization(image)
            
            cartoon_image_comic = halftone_shading(image)
            
            
            
            cv2.imshow("Original Image", image)
            cv2.imshow("Pencil Sketched Image", pencil_sketched)
            cv2.imshow("Edge Detection", cartoon_by_edge_detection)
            cv2.imshow("Color Quantization", cartoon_image)
            cv2.imshow("Comic Book Style", cartoon_image_comic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
    
def main():
    root = Tk()
    root.title("Image Processing GUI")
    root.geometry("500x500")
    root.configure(bg='white')

    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 12), padding=10)
    style.map('TButton', foreground=[('pressed', 'red'), ('active', 'blue')],
          background=[('pressed', '!disabled', 'yellow'), ('active', 'white')])
    
    button_frame = tk.Frame(root, bg='white')
    button_frame.pack(pady=20, padx=20, expand=True)

    button1 = ttk.Button(button_frame, text="1. To Pencil Sketch an image", command=lambda: open_file_and_process(1), style='TButton')
    button2 = ttk.Button(button_frame, text="2. To cartoon an image by Outline and Edge Detection", command=lambda: open_file_and_process(2), style='TButton')
    button3 = ttk.Button(button_frame, text="3. To cartoon an image by Color Quantization", command=lambda: open_file_and_process(3), style='TButton')
    button4 = ttk.Button(button_frame, text="4. To cartoon an image by Comic Book Style", command=lambda: open_file_and_process(4), style='TButton')
    button5 = ttk.Button(button_frame, text="5. To see all the outcomes together", command=lambda: open_file_and_process(5), style='TButton')
    button_exit = ttk.Button(button_frame, text="0. Exit", command=root.destroy, style='TButton')

    button1.pack(pady=10)
    button2.pack(pady=10)
    button3.pack(pady=10)
    button4.pack(pady=10)
    button5.pack(pady=10)
    button_exit.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
