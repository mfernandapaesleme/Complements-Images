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


import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, disk, square
from skimage.filters import frangi, threshold_otsu, gaussian, threshold_local
from skimage.filters.rank import enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte, img_as_float
from skimage import filters, morphology
import cv2

def frangi_segmentation(img, img_mask, adaptive_threshold=False):
    """
    Segmentação melhorada de vasos sanguíneos
    """
    # 1. Preprocessamento mais agressivo
    img_float = img_as_float(img)

    img_float = erosion(img_float, disk(1))  # Erosão para remover pequenos ruídos
    
    # Equalização adaptativa do histograma (CLAHE)
    img_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = img_clahe.apply(img_as_ubyte(img_float))
    
    # 2. Filtro Frangi para detecção de vasos (IMPLEMENTAÇÃO CORRETA)
    # Múltiplas escalas para capturar vasos de diferentes larguras
    sigmas = np.arange(0.5, 5, 0.5)  # Escalas mais finas
    frangi_response = frangi(img_enhanced, sigmas=sigmas, black_ridges=True)
    
    print(f"Frangi response - min: {np.min(frangi_response):.4f}, max: {np.max(frangi_response):.4f}")
    
    # 3. Threshold adaptativo em vez de fixo
    if adaptive_threshold:
        # Threshold local adaptativo
        threshold_value = threshold_local(frangi_response, block_size=35, offset=0.01)
        binary_vessels = frangi_response > threshold_value
    else:
        # Otsu threshold no resultado do Frangi
        threshold_value = threshold_otsu(frangi_response)
        binary_vessels = frangi_response > threshold_value
        print(f"Otsu threshold: {threshold_value:.4f}")
    
    # 4. Pós-processamento morfológico
    # Remover ruído pequeno
    cleaned = morphology.remove_small_objects(binary_vessels, min_size=60)
    
    # Conectar estruturas próximas
    connected = closing(cleaned, disk(1))
    
    # Suavização final
    final_result = opening(connected, disk(1))
    
    # Aplicar máscara
    final_result = final_result & img_mask.astype(bool)
    
    print(f"Pixels detectados: {np.sum(final_result)}")
    
    return final_result.astype(np.uint8)

def hybrid_segmentation(img, img_mask):
    """
    Abordagem híbrida combinando múltiplas técnicas
    """
    img_float = img_as_float(img)
    
    # 1. Frangi filter
    sigmas = np.arange(0.5, 4, 0.5)
    frangi_result = frangi(img_float, sigmas=sigmas, black_ridges=True)
    
    # 2. Top-hat transform multi-escala
    selem_sizes = [1, 2, 3, 5]
    tophat_results = []
    for size in selem_sizes:
        tophat = white_tophat(img_as_ubyte(img_float), disk(size))
        tophat_results.append(tophat)
    
    # Combinar resultados do top-hat
    combined_tophat = np.maximum.reduce(tophat_results)
    combined_tophat = img_as_float(combined_tophat)
    
    # 3. Combinar Frangi e Top-hat
    # Normalizar ambos para [0,1]
    frangi_norm = (frangi_result - np.min(frangi_result)) / (np.max(frangi_result) - np.min(frangi_result))
    tophat_norm = (combined_tophat - np.min(combined_tophat)) / (np.max(combined_tophat) - np.min(combined_tophat))
    
    # Combinação ponderada
    combined = 0.7 * frangi_norm + 0.3 * tophat_norm
    
    # 4. Threshold adaptativo
    threshold_value = threshold_otsu(combined)
    binary_vessels = combined > (threshold_value * 0.5)  # Threshold mais baixo para capturar mais vasos
    
    # 5. Pós-processamento
    cleaned = morphology.remove_small_objects(binary_vessels, min_size=5)
    final_result = closing(cleaned, disk(2))
    
    # Aplicar máscara
    final_result = final_result & img_mask.astype(bool)
    
    return final_result.astype(np.uint8)

def simple_vessel_enhancement(img, img_mask):
    """
    Abordagem mais simples e direta
    """
    # 1. Frangi filter básico
    img_float = img_as_float(img)
    sigmas = np.arange(1.0, 2.5, 0.5)
    frangi_response = frangi(img_float, sigmas=sigmas, black_ridges=True)
    
    # 2. Threshold bem conservador - apenas os melhores candidatos
    threshold_93 = np.percentile(frangi_response, 93)  # Top 7%
    binary_vessels = frangi_response > threshold_93

    # 3. Limpeza mínima
    cleaned = morphology.remove_small_objects(binary_vessels, min_size=20)
    final_result = cleaned & img_mask.astype(bool)
    
    print(f"Pixels detectados (método simples): {np.sum(final_result)}")
    
    return final_result.astype(np.uint8)

# Função principal atualizada
def my_segmentation(img, img_mask, method='hybrid'):
    """
    Função principal de segmentação melhorada
    """
    if method == 'frangi':
        return frangi_segmentation(img, img_mask)
    elif method == 'hybrid':
        return hybrid_segmentation(img, img_mask)
    elif method == 'simple':
        return simple_vessel_enhancement(img, img_mask)
    else:
        # Método original melhorado
        return frangi_segmentation(img, img_mask, adaptive_threshold=False)

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
        img_out = my_segmentation(img, img_mask, method='simple')

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