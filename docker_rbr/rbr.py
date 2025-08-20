# Python Standard Libraries
from pathlib import Path
from enum import Enum

# Third Party Libraries
import numpy as np
import cv2
import torchstain
from stardist.models import StarDist2D
from shapely.geometry import Polygon


class Confidence(Enum):
    HIGH = 2
    MEDIUM = 1
    LOW = 0


class Shape(Enum):
    ROUND = 0
    OVAL = 1
    ELONGATED = 2
    IRREGULAR = 3


CELL_DETECT_MODEL = StarDist2D.from_pretrained('2D_versatile_he')
TARGET_PATH = Path('/opt/app/resources') / 'target.png' # Stain normalization target


def stain_norm(image: np.ndarray) -> np.ndarray:
    """
    Normalize the stain of an image using Macenko normalization which performs 
    better during StarDist detection.

    Parameters
    ----------
    image : np.ndarray
        Input image in RGB format.

    Returns
    -------
    np.ndarray
        Stain-normalized image.
    """
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    target_img = cv2.cvtColor(cv2.imread(str(TARGET_PATH)), cv2.COLOR_BGR2RGB)
    normalizer.fit(target_img)
    norm, _, _ = normalizer.normalize(image)
    return norm


def detect_cells(
    image: np.ndarray, distance_threshold: int = 40
) -> tuple[np.ndarray, dict, list]:
    """
    Detect cells in the image using the StarDist model. Further checks if one or
    more are close to the image center.

    Parameters
    ----------
    img: np.ndarray
        The image to process.
    distance_threshold : int
        The distance threshold for considering a cell "close" to the center.

    Returns
    -------
    tuple
        A tuple containing:
        - labels: The predicted labels for the image.
        - shapes: The detected shapes in the image.
        - close_indices: Indices of shapes that are close to the image center.
        - conf: The confidence in the analysis.
    """
    if image.max() > 1.0:
        image = image / 255.0
    conf = Confidence.HIGH

    # try confident detection
    labels, shapes = CELL_DETECT_MODEL.predict_instances(image, prob_thresh=0.6, nms_thresh=0.3)

    # fallback to less confident detection if no cells are found
    if labels.max() == 0:
        labels, shapes = CELL_DETECT_MODEL.predict_instances(image, prob_thresh=0.51, nms_thresh=0.3)
        conf = Confidence.LOW

    # Image center
    h, w = labels.shape
    center = np.array([w / 2, h / 2])

    # get centoids close to the center
    centroids = np.array(shapes['points'])
    distances = np.linalg.norm(centroids - center, axis=1)
    close_indices = np.where(distances < distance_threshold)[0]

    # If more than one shape is close to the center check if they are also close
    # to each other
    if len(close_indices) >= 2:
        center_idx = close_indices[np.argmin(distances[close_indices])]
        center_coord = centroids[center_idx]

        refined_close_indices = []
        for idx in close_indices:
            dist_to_center_shape = np.linalg.norm(centroids[idx] - center_coord)
            if dist_to_center_shape <= distance_threshold:
                refined_close_indices.append(idx)

        close_indices = np.array(refined_close_indices)

    return labels, shapes, close_indices, conf


def get_cell_morphology(
    coord: tuple, centroid: tuple, image: np.ndarray, radius: int = 10
) -> Shape:
    """
    Classify the morphology of a cell.

    Parameters
    ----------
     coord: tuple
        Coordinates of the cell in the format (y, x).
    centroid: tuple
        Centroid of the cell in the format (y, x).
    img: np.ndarray
        The image containing the cell.
    radius: int, optional
        Radius around the centroid to consider for ring shape detection.

    Returns
    -------
    Shape
        The classified shape of the cell.
    """
    y, x = coord
    poly_pts = np.stack([x, y], axis=1).astype(np.int32)
    poly = Polygon(poly_pts)

    # Geometric features
    circularity = 4 * np.pi * poly.area / (poly.length ** 2) if poly.length > 0 else 0
    smoothness = poly.length / poly.convex_hull.length if poly.convex_hull.length > 0 else 0
    aspect_ratio = (
        (poly.bounds[2] - poly.bounds[0]) / (poly.bounds[3] - poly.bounds[1])
        if (poly.bounds[3] - poly.bounds[1]) > 0 else 0
    )

    # Ringshape detection
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 1)

    cx, cy = int(centroid[0]), int(centroid[1])
    h, w = image.shape[:2]
    x0, x1 = max(0, cx - radius), min(w, cx + radius)
    y0, y1 = max(0, cy - radius), min(h, cy + radius)

    patch = image[y0:y1, x0:x1]
    patch_mask = mask[y0:y1, x0:x1]

    ring_ness = 0
    if patch.shape[0] > 0 and patch.shape[1] > 0:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        H, W = gray.shape
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - W // 2) ** 2 + (Y - H // 2) ** 2)

        center_mask = (dist < radius * 0.3) & (patch_mask == 1)
        ring_mask = (radius * 0.6 < dist) & (dist < radius * 0.9) & (patch_mask == 1)

        if np.any(center_mask) and np.any(ring_mask):
            ring_ness = np.mean(gray[ring_mask]) - np.mean(gray[center_mask])

    result = None
    if ring_ness > 12:
        result = Shape.ROUND
    elif circularity > 0.8 and smoothness > 0.95 and (1.4 > aspect_ratio > 0.70):
        result = Shape.OVAL
    elif circularity > 0.6 and smoothness > 0.8 and (1.5 > aspect_ratio > 0.66):
        result = Shape.ELONGATED
    else:
        result = Shape.IRREGULAR

    return result


def get_orientation(shape: tuple) -> float:
    """
    Get the orientation angle of a shape represented by its coordinates.

    Parameters
    ----------
    shape : tuple
        Coordinates of the shape in the format (y, x).

    Returns
    -------
    tuple(angle, confidence)
        - angle: float
            The orientation angle of the shape in degrees.
        - confidence: Confidence
            The confidence of the angle based on the cells morphology.

    """
    conf = Confidence.HIGH
    
    y, x = shape
    pts = np.stack([x, y], axis=1)

    # Try fitEllipse (needs >=5 points)
    if len(pts) >= 5:
        _, (major, minor), ang = cv2.fitEllipse(pts)  # w,h correspond to axes lengths
        # Ensure angle corresponds to the major axis
        if minor > major:
            major, minor = minor, major
            angle = ang + 90
        
        angle = angle % 180
        
        ratio = float(minor) / float(major)
        if ratio < 0.5:
            conf = Confidence.LOW
        elif ratio < 0.8:
            conf = Confidence.MEDIUM
        else:
            conf = Confidence.HIGH
    else:
        # Center the points
        pts_centered = pts - np.mean(pts, axis=0)

        # PCA: get principal axis
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        # Take angle of first eigenvector
        main_axis = eigvecs[:, np.argmax(eigvals)]
        angle_rad = np.arctan2(main_axis[1], main_axis[0])
        angle = np.degrees(angle_rad) % 180

        conf = Confidence.MEDIUM

    return angle, conf


def check_orientation(coord1: tuple, coord2: tuple):
    """Determine if two shapes are parallel in orientation.

    Computes axis angles and shape descriptors for both shapes, compares their
    orientation, and adapts tolerance if shapes are near-circular, small, or
    poorly elongated. Optionally visualizes results.

    Parameters
    ----------
    shape_1 : array-like
        First shape (points).
    shape_2 : array-like
        Second shape (points).
    tolerance : float, optional (default=15)
        Base angular tolerance in degrees.
    debug : bool, optional (default=True)
        If True, prints debug info and visualizes orientation.
    img : str, optional
        Path to the image file for visualization.
    base_path : Path, optional (default="./normalized")
        Base directory for relative image paths.

    Returns
    -------
    is_parallel : bool
        True if orientations are parallel within tolerance.
    diff : float
        Angular difference between the two shapes in degrees.
    q : dict
        Quality flags:
            - 'low_ecc': at least one shape has low eccentricity.
            - 'low_ar': at least one shape has low aspect ratio.
            - 'small' : at least one shape has very small area.
    """
    angle_1, conf_1 = get_orientation(coord1)
    angle_2, conf_2 = get_orientation(coord2)

    if conf_1.value > conf_2.value:
        conf_1 = conf_2

    angle = abs(angle_1 - angle_2) % 180
    if angle > 90:
        angle = 180 - angle

    return angle, conf_1

def refine_aucmedi_prediction(img: np.ndarray, pred_nmf: float, pred_amf: float) -> tuple[float, float]:
    """
    Refine AMF/NMF predictions using conventional computer-vision heuristics.

    This function post-processes raw predictions from an AUCMEDI model
    (probabilities for AMF vs. NMF) by applying interpretable geometric
    rules based on cell morphology.

    The refinement follows a case-based strategy depending on how many
    nuclei (cells) are detected in the image center:

    Cases
    -----
    0 cells:
        - If no nuclei are found increase AMF probability.
    1 cell:
        - Classify the morphology of the detected cell.
        - Adjust AMF/NMF accordingly to the detected morphology.
    2 cells:
        - Compare orientation of both cells.
        - If shapes are parallel increase NMF probability, else AMF.
    3+ cells:
        - Leave scores unchanged.

    Parameters
    ----------
    img : np.ndarray
        Raw image without any preprocessing applied
    pred_nmf : float
        Initial probability score for NMF.
    pred_amf : float
        Initial probability score for AMF.

    Returns
    -------
    tuple (refined_pred_nmf, refined_pred_amf)
        - refined_pred_nmf : float
            Updated probability for NMF.
        - refined_pred_amf : float
            Updated probability for AMF.
    """
    img = stain_norm(img)
    _, shape_dict, cells_ind, confidence = detect_cells(img)

    match len(cells_ind):
        case 0:
            # No cells detected, increase AMF probability
            pred_amf += 0.01 if confidence == Confidence.HIGH else 0.025
            pred_nmf -= 0.01 if confidence == Confidence.HIGH else 0.025
        case 1:
            shape = get_cell_morphology(
                coord=shape_dict['coord'][cells_ind[0]],
                centroid=shape_dict['points'][cells_ind[0]],
                image=img
            )
            match shape:
                case Shape.ROUND:
                    pred_nmf += 0.2 if confidence == Confidence.HIGH else 0.1
                    pred_amf -= 0.2 if confidence == Confidence.HIGH else 0.1
                case Shape.OVAL:
                    pred_nmf += 0.2 if confidence == Confidence.HIGH else -0.1
                    pred_amf -= 0.2 if confidence == Confidence.HIGH else -0.1
                case Shape.ELONGATED:
                    pred_nmf += 0.2 if confidence == Confidence.HIGH else -0.1
                    pred_amf -= 0.2 if confidence == Confidence.HIGH else -0.1
                case Shape.IRREGULAR:
                    pass
        case 2:
            # Skip if either contour is tiny
            angle, angle_conf = check_orientation(
                shape_dict['coord'][cells_ind[0]],
                shape_dict['coord'][cells_ind[1]]
            )
            if angle_conf.value < confidence.value:
                confidence = angle_conf

            # parallel = NMF
            if angle <= 10.0:
                match confidence:
                    case Confidence.HIGH:
                        pred_nmf += 0.6
                        pred_amf -= 0.6
                    case Confidence.MEDIUM:
                        pred_nmf += 0.3
                        pred_amf -= 0.3
                    case Confidence.LOW:
                        pred_nmf += 0.1
                        pred_amf -= 0.1
            # small = still hints toward NMF
            elif angle <= 20.0:
                match confidence:
                    case Confidence.HIGH:
                        pred_nmf += 0.4
                        pred_amf -= 0.4
                    case Confidence.MEDIUM:
                        pred_nmf += 0.3
                        pred_amf -= 0.3
                    case Confidence.LOW:
                        pred_nmf += 0.2
                        pred_amf -= 0.2
            # non-parallel = AMF
            else:
                pass
             
    pred_amf = float(np.clip(pred_amf, 0.0, 1.0))
    pred_nmf = float(np.clip(pred_nmf, 0.0, 1.0))

    return pred_nmf, pred_amf