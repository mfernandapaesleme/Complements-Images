import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.filters.rank import entropy, enhance_contrast_percentile
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, disk, square
from skimage.filters import frangi, threshold_otsu, gaussian
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte, img_as_float
import math
from skimage import data, filters
from matplotlib import pyplot as plt
import os
import glob
import cv2
from image_viewer import ImageViewer


def my_segmentation(img, img_mask, seuil):
    img_out = (img_mask & (img < seuil))
    return multiscale_vessel_segmentation(img_out, img_mask)

def multiscale_vessel_segmentation(img, img_mask, sigma_min=1, sigma_max=5, num_sigma=5):
    """
    Parameters:
    -----------
    img : ndarray
        Image d'entrée en niveaux de gris
    img_mask : ndarray
        Masque binaire indiquant la région d'intérêt
    sigma_min : float
        Échelle minimale pour le filtre Frangi
    sigma_max : float
        Échelle maximale pour le filtre Frangi
    num_sigma : int
        Nombre d'échelles à considérer
        
    Returns:
    --------
    vessel_segmentation : ndarray
        Segmentation binaire des vaisseaux
    """
    # Prétraitement : amélioration du contraste et illumination
    img_contrast = enhance_contrast_percentile(img_as_ubyte(img), disk(5), p0=0.8, p1=99.2)
    print (f"Image contrast shape: {img_contrast.shape}, dtype: {img_contrast.dtype}")
    
    # Multiscale vessel segmentation using Frangi filter
    selem = [disk(1), disk(2), disk(3), disk(5), disk(7)]  # Para vasos de diferentes espessuras
    results_data = []

    for s in selem:
        result = white_tophat(img_contrast, s)  # Realiza a abertura branca
        results_data.append(result)  # Armazena o resultado
        print(f"Top-hat with disk({s}) - min: {np.min(result):.4f}, max: {np.max(result):.4f}")

    combined = np.maximum.reduce(results_data)
    print(f"Combined response - min: {np.min(combined):.4f}, max: {np.max(combined):.4f}")

    threshold = threshold_otsu(combined)
    binary_vessels = combined > threshold
    print(f"Otsu threshold: {threshold:.4f}")

    # Remove pequenos objetos (ruído)
    cleaned = opening(binary_vessels, disk(1))
    
    # Conecta pequenas quebras
    connected = closing(cleaned, disk(2))
    
    # 7. Aplicar máscara para manter apenas região de interesse
    final_result = connected & img_mask.astype(bool)

    print(f"Final result - shape: {final_result.shape}, dtype: {final_result.dtype}")
    print(f"Pixels detectados: {np.sum(final_result)}")

    return final_result.astype(np.uint8)

def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT) # On reduit le support de l'évaluation...
    img_out_skel = skeletonize(img_out) # ...aux pixels des squelettes
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs
    ACCU = TP / (TP + FP) if (TP + FP) > 0 else 0 # Precision
    RECALL = TP / (TP + FN) if (TP + FN) > 0 else 0 # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

# Get all image files (OSC, OSN, ODC, ODN)
image_extensions = ['*_OSC.jpg', '*_OSN.jpg', '*_ODC.jpg', '*_ODN.jpg']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(f'./images_IOSTAR/{ext}'))
image_files.sort()  # Sort to ensure consistent order

# Initialize lists to store results
all_accuracy = []
all_recall = []
image_names = []
all_results = []  # Store all image data for interactive display

print(f"Found {len(image_files)} images to process")
print("-" * 50)

for img_file in image_files:
    # Extract image identifier from filename
    base_name = os.path.basename(img_file)
    # Extract number from filename (e.g., "02" from "star02_OSC.jpg")
    if base_name.startswith('star'):
        img_id = base_name.split('_')[0].replace('star', '').zfill(2)
    else:
        print(f"Warning: Unexpected filename format: {base_name}")
        continue
    
    # Construct GT filename
    gt_file = f'./images_IOSTAR/GT_{img_id}.png'
    
    if not os.path.exists(gt_file):
        print(f"Warning: Ground truth file {gt_file} not found, skipping {base_name}")
        continue
    
    try:
        print(f"Processing {base_name}...")
        
        # Load original image
        img = np.asarray(Image.open(img_file)).astype(np.uint8)
        nrows, ncols = img.shape
        row, col = np.ogrid[:nrows, :ncols]
        
        # Create mask for inscribed disk
        img_mask = (np.ones(img.shape)).astype(np.uint8)
        invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
        img_mask[invalid_pixels] = 0
        
        # Perform segmentation
        img_out = my_segmentation(img, img_mask, 80)
        
        # Load ground truth
        img_GT = np.asarray(Image.open(gt_file)).astype(np.uint8)
        
        # Evaluate
        ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
        
        # Store results
        all_accuracy.append(ACCU)
        all_recall.append(RECALL)
        image_names.append(base_name)
        
        print(f"  Accuracy = {ACCU:.4f}, Recall = {RECALL:.4f}")
        
        # Store all data for interactive display
        results_data = {
            'img': img,
            'img_out': img_out,
            'img_out_skel': img_out_skel,
            'img_GT': img_GT,
            'GT_skel': GT_skel,
            'accuracy': ACCU,
            'recall': RECALL,
            'name': base_name
        }
        all_results.append(results_data)
        
    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")
        continue

# Print summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total images processed: {len(all_accuracy)}")
print(f"Average Accuracy: {np.mean(all_accuracy):.4f} ± {np.std(all_accuracy):.4f}")
print(f"Average Recall: {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
print(f"Min Accuracy: {np.min(all_accuracy):.4f}")
print(f"Max Accuracy: {np.max(all_accuracy):.4f}")
print(f"Min Recall: {np.min(all_recall):.4f}")
print(f"Max Recall: {np.max(all_recall):.4f}")

# Detailed results table
print("\nDETAILED RESULTS:")
print("-" * 40)
print(f"{'Image':<15} {'Accuracy':<10} {'Recall':<10}")
print("-" * 40)
for name, acc, rec in zip(image_names, all_accuracy, all_recall):
    print(f"{name:<15} {acc:<10.4f} {rec:<10.4f}")

# Create interactive viewer
if all_results:
    print("\nLaunching interactive image viewer...")
    print("Use 'Previous' and 'Next' buttons to navigate through images")
    print("Use 'Summary' button to see overall statistics")
    viewer = ImageViewer(all_results)
    viewer.show()
else:
    print("No images were successfully processed.")