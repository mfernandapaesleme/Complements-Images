import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters, feature
from matplotlib import pyplot as plt
from skimage.filters import frangi, hessian, meijering, sato

def multiscale_vessel_segmentation(img, img_mask, sigma_min=1, sigma_max=5, num_sigma=5):
    """
    Segmentation multi-échelle des vaisseaux sanguins utilisant le filtre Frangi.
    
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
    # Prétraitement : amélioration du contraste
    img_eq = enhance_contrast_percentile(img_as_ubyte(img), disk(5), p0=2, p1=98)
    
    # Application du filtre Frangi à différentes échelles
    sigmas = np.linspace(sigma_min, sigma_max, num_sigma)
    vessel_response = frangi(img_eq, sigmas=sigmas, black_ridges=True)
    
    # Normalisation entre 0 et 1
    vessel_response = (vessel_response - vessel_response.min()) / (vessel_response.max() - vessel_response.min())
    
    # Seuillage adaptatif
    threshold = filters.threshold_otsu(vessel_response[img_mask])
    vessel_binary = vessel_response > threshold
    
    # Post-traitement: nettoyage des petits objets et fermeture pour connecter les vaisseaux
    cleaned = ndi.binary_opening(vessel_binary & img_mask, structure=disk(1))
    connected = ndi.binary_closing(cleaned, structure=disk(2))
    
    # Suppression des petits objets isolés (qui ne sont probablement pas des vaisseaux)
    min_vessel_size = 20
    labeled_array, num_features = ndi.label(connected)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Ignorer l'arrière-plan
    too_small = component_sizes < min_vessel_size
    too_small_mask = too_small[labeled_array]
    connected[too_small_mask] = False
    
    return connected

def alternative_vessel_segmentation(img, img_mask):
    """
    Méthode alternative combinant détection de contours et morphologie mathématique
    """
    # Prétraitement
    img_eq = enhance_contrast_percentile(img_as_ubyte(img), disk(3), p0=2, p1=98)
    
    # Filtre de détection de contours multidirectionnel
    edges = filters.sobel(img_eq)
    edges_norm = (edges - edges.min()) / (edges.max() - edges.min())
    
    # Filtre d'entropie pour améliorer la détection des zones de texture différente
    entropy_img = entropy(img_as_ubyte(img), disk(5))
    entropy_norm = (entropy_img - entropy_img.min()) / (entropy_img.max() - entropy_img.min())
    
    # Combinaison des indices
    combined = (1 - edges_norm) * entropy_norm
    
    # Seuillage adaptatif
    threshold = filters.threshold_otsu(combined[img_mask])
    vessel_binary = (combined > threshold) & img_mask
    
    # Post-traitement
    vessel_binary = closing(vessel_binary, disk(2))
    vessel_binary = opening(vessel_binary, disk(1))
    
    return vessel_binary

def my_segmentation(img, img_mask, method='multiscale', **kwargs):
    """
    Fonction de segmentation principale
    
    Parameters:
    -----------
    img : ndarray
        Image d'entrée en niveaux de gris
    img_mask : ndarray
        Masque binaire indiquant la région d'intérêt
    method : str
        Méthode de segmentation ('multiscale', 'alternative' ou 'simple')
    kwargs : dict
        Paramètres supplémentaires pour les méthodes spécifiques
        
    Returns:
    --------
    vessel_segmentation : ndarray
        Segmentation binaire des vaisseaux
    """
    # Inversion de l'image (les vaisseaux sont foncés dans l'original)
    img_inv = 255 - img
    
    if method == 'multiscale':
        return multiscale_vessel_segmentation(img_inv, img_mask, **kwargs)
    elif method == 'alternative':
        return alternative_vessel_segmentation(img_inv, img_mask)
    elif method == 'simple':
        # Méthode de seuillage simple (implémentation originale)
        threshold = kwargs.get('threshold', 180)
        return (img_mask & (img_inv > threshold))
    else:
        raise ValueError(f"Méthode '{method}' non reconnue")

def evaluate(img_out, img_GT):
    """
    Évaluation de la qualité de la segmentation par rapport à la vérité terrain.
    
    Parameters:
    -----------
    img_out : ndarray
        Segmentation produite par l'algorithme
    img_GT : ndarray
        Vérité terrain (Ground Truth)
        
    Returns:
    --------
    ACCU : float
        Précision (vrais positifs / (vrais positifs + faux positifs))
    RECALL : float
        Rappel (vrais positifs / (vrais positifs + faux négatifs))
    img_out_skel : ndarray
        Squelette de la segmentation
    GT_skel : ndarray
        Squelette de la vérité terrain
    """
    GT_skel = thin(img_GT, max_iter=15)  # On suppose que la demie épaisseur maximum
    img_out_skel = thin(img_out, max_iter=15)  # d'un vaisseau est de 15 pixels...
    
    TP = np.sum(img_GT & img_out)  # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT)  # Faux positifs (relaxés)
    FN = np.sum(GT_skel & ~img_out)  # Faux négatifs (relaxés)
    
    ACCU = TP / (TP + FP) if (TP + FP) > 0 else 0  # Précision
    RECALL = TP / (TP + FN) if (TP + FN) > 0 else 0  # Rappel
    
    # F1-score (mesure combinée de précision et rappel)
    F1 = 2 * (ACCU * RECALL) / (ACCU + RECALL) if (ACCU + RECALL) > 0 else 0
    
    print(f"Précision = {ACCU:.4f}, Rappel = {RECALL:.4f}, F1-score = {F1:.4f}")
    
    return ACCU, RECALL, img_out_skel, GT_skel

# Exemple d'utilisation
if __name__ == "__main__":
    # Ouvrir l'image originale en niveau de gris
    img = np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
    print(f"Dimensions de l'image: {img.shape}")
    
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    
    # On ne considère que les pixels dans le disque inscrit
    img_mask = np.ones(img.shape).astype(np.bool_)
    invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
    img_mask[invalid_pixels] = 0
    
    # Appliquer les différentes méthodes de segmentation
    img_out_simple = my_segmentation(img, img_mask, method='simple', threshold=180)
    img_out_multiscale = my_segmentation(img, img_mask, method='multiscale')
    img_out_alternative = my_segmentation(img, img_mask, method='alternative')
    
    # Ouvrir l'image Vérité Terrain en booléen
    img_GT = np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.bool_)
    
    # Évaluation des différentes méthodes
    print("Méthode simple:")
    ACCU_simple, RECALL_simple, img_out_skel_simple, GT_skel = evaluate(img_out_simple, img_GT)
    
    print("\nMéthode multi-échelle:")
    ACCU_multi, RECALL_multi, img_out_skel_multi, _ = evaluate(img_out_multiscale, img_GT)
    
    print("\nMéthode alternative:")
    ACCU_alt, RECALL_alt, img_out_skel_alt, _ = evaluate(img_out_alternative, img_GT)
    
    # Affichage des résultats
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Image Originale')
    
    plt.subplot(232)
    plt.imshow(img_out_multiscale)
    plt.title('Segmentation multi-échelle')
    
    plt.subplot(233)
    plt.imshow(img_out_skel_multi)
    plt.title('Squelette multi-échelle')
    
    plt.subplot(234)
    plt.imshow(img_GT)
    plt.title('Vérité Terrain')
    
    plt.subplot(235)
    plt.imshow(img_out_alternative)
    plt.title('Segmentation alternative')
    
    plt.subplot(236)
    plt.imshow(GT_skel)
    plt.title('Vérité Terrain Squelette')
    
    plt.tight_layout()
    plt.show()