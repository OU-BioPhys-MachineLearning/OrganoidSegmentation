import numpy as np
from PIL import Image
import torch
from transformers import SamModel, SamProcessor, pipeline
import cv2
from pathlib import Path
from skimage.measure import label

import os

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from typing import List, Tuple, Optional, Any, Union

from pathlib import Path
import skimage.measure
import pandas as pd


def AnalyzeAndExport(images: np.ndarray, path: str) -> None:
    """
    Analyzes the segmented images and exports properties of
    segmented regions to an Excel file. The function takes in a
    3D numpy array of segmented images, where each slice along the
    first dimension corresponds to a different image. It uses
    skimage.measure.regionprops to calculate various properties of
    the segmented regions in each image, such as area, eccentricity,
    perimeter, etc. Properties are stored in a dictionary of pandas
    DataFrames, where each DataFrame corresponds to a specific
    property and contains values for all regions across all images.
    Finally, the function writes each DataFrame to a separate sheet
    in an Excel file at the specified path. This allows for easy
    analysis and visualization of the properties of the segmented
    regions across the entire dataset. (This piece of code is copied
    from OrganoID, but since we are revising the masks, we need to
    re-run the analysis to get the properties of the revised masks.)
    
    Parameters:
    - images: A 3D numpy array of segmented images, where each
      slice along the first dimension corresponds to a different
      image. The segmented regions should be represented by integer
      labels, where each unique integer corresponds to a different
      segmented region. The background should be represented by 0s.
    - path: The path to the Excel file where the properties will be
      exported. The function will create a new Excel file at this
      path, or overwrite it if it already exists. Each sheet in the
      Excel file will correspond to a specific property of the
      segmented regions, and will contain the values for all regions
      across all images. The properties that are calculated and
      exported include area, axis lengths, centroid, eccentricity,
      equivalent diameter, Euler number, extent, Feret diameter,
      orientation, perimeter, and solidity.
    """



    with pd.ExcelWriter(path) as writer:
        propertyNames = ['area', 'axis_major_length', 'axis_minor_length',
                         'centroid', 'eccentricity', 
                         'equivalent_diameter_area', 'euler_number',
                         'extent', 'feret_diameter_max', 'orientation',
                         'perimeter', 'perimeter_crofton', 'solidity']

        size = (np.max(images)+1, images.shape[0])
        data = {propertyName: pd.DataFrame(np.ndarray(size, dtype=str)) for
                 propertyName in propertyNames}

        for t in range(images.shape[0]):
            regions = skimage.measure.regionprops(images[t])
            for propertyName in propertyNames:
                for region in regions:
                    value = getattr(region, propertyName)
                    label = region.label
                    data[propertyName].iloc[label, t] = str(value)

        for propertyName in propertyNames:
            data[propertyName].to_excel(writer, sheet_name=propertyName)

propertyNames = [
    'area',
    'axis_major_length',
    'axis_minor_length',
    'centroid',
    'eccentricity',
    'equivalent_diameter_area',
    'euler_number',
    'extent',
    'feret_diameter_max',
    'orientation',
    'perimeter',
    'perimeter_crofton',
    'solidity',
]


def analyze_masks(d:Path) -> None:
    """
    Analyzes the segmented masks in a directory and exports properties
    of segmented regions to an Excel file. The function takes in a
    Path object representing the directory containing the segmented
    masks, and processes each mask to extract properties of the
    segmented regions. The properties are calculated using
    skimage.measure.regionprops, and are stored in a dictionary of
    pandas DataFrames, where each DataFrame corresponds to a specific
    property and contains values for all regions across all masks.
    Finally, the function writes each DataFrame to a separate sheet in
    an Excel file located in the same directory. This allows for easy
    analysis and visualization of the properties of the segmented
    regions across all masks in the directory.

    Parameters:
    - d: A Path object representing the directory containing the
      segmented masks. The masks should be in TIFF format, and should
      have "binary" in their filename. The function will process each
      mask in the directory, extract properties of the segmented
      regions, and export them to an Excel file located in the same
      directory.
    """

    image_paths = list(d.glob("*binary.TIF"))
    if not image_paths:
        image_paths = list(d.glob("*.TIF"))
    elif not image_paths:
        image_paths = list(d.glob("*.png"))
    elif not image_paths:
        raise RuntimeError(f"No TIFF or PNG images found in {d}")
    image_paths.sort()

    print(f"{len(image_paths)} Images Found in {d}")

    images = np.zeros((len(image_paths), 1536, 1536), dtype=int)

    for i in range(len(image_paths)):
        images[i, :, :] = skimage.measure.label(
            np.array(Image.open(image_paths[i]).convert("L")).astype(int))

    print(str(d / "data.xlsx"))
    
    AnalyzeAndExport(images, str(d / "data.xlsx"))


def revise_mask(
    orgomask: np.ndarray, composite: np.ndarray
) -> np.ndarray:
    """
    Revises the original OrganoID mask based on the composite mask
    generated by SAM. The function takes in the original OrganoID
    mask and the composite mask, and labels the connected components
    in the original mask. It then iterates through each labeled
    component and checks if it overlaps with the composite mask. If a
    component overlaps with the composite mask, it is retained in the
    revised mask. The revised mask is created by taking the pixel-wise
    maximum of all accepted components, resulting in a new mask that
    retains only the regions of the original OrganoID mask that have
    agreement with the composite SAM mask. This revised mask can be
    used for further analysis or comparison with other segmentation
    methods.

    Parameters:
    - orgomask: A numpy array representing the original OrganoID
      mask, where the segmented regions are represented by 1s and the
      background is represented by 0s.
    - composite: A numpy array representing the composite mask
      generated by SAM, which can be used as a reference for revising
      the original OrganoID mask. The composite mask should have the
      same dimensions as the original mask, and should also be a
      binary array where the segmented regions are represented by 1s
      and the background is represented by 0s.

    Returns:
    - revised: A numpy array representing the revised OrganoID mask,
      where only the regions of the original mask that overlap with
      the composite SAM mask are retained. The revised mask is created
      by labeling the connected components in the original mask,
      checking for overlap with the composite mask, and taking the
      pixel-wise maximum of the accepted components to create the
      final revised mask.
    """

    labeled, num = label(orgomask, return_num=True)

    revised = np.zeros_like(orgomask)
    ones = np.ones_like(orgomask)

    for i in range(1, num + 1):
        mask = labeled == i * ones
        if (mask * composite).any():
            revised = np.fmax(revised, mask)
    


    return revised

def compare_masks(
    image_path: Union[str, Path],
    masks: List[np.ndarray],
    composite: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Compares multiple segmentation masks generated by different
    methods (e.g., OrganoID, GroundingDINO, SAM) for the same image,
    and selects the best mask based on agreement with a composite
    mask. The function takes in the list of masks and an optional
    composite mask, and calculates the agreement between each mask and
    the composite mask using a similarity metric. The mask with the
    highest agreement that exceeds a certain threshold is selected as
    the best mask. If no mask exceeds the threshold, the mask with the
    highest agreement is returned. The function can also display the
    masks and their agreement scores if desired.

    Parameters:
    - image_path: The file path of the original image corresponding to
      the masks being compared. This is used for reference and
      debugging purposes, and can be displayed alongside the masks if
      desired.
    - masks: A list of numpy arrays representing the segmentation
      masks generated by different methods for the same image. Each
      mask should be a binary array where the segmented region is
      represented by 1s and the background is represented by 0s.
    - composite: An optional numpy array representing a composite mask
      that can be used as a reference for comparison. This could be a
      mask generated by a combination of methods, or a mask that has
      been revised based on certain criteria. If provided, the
      agreement between each mask and the composite mask will be
      calculated to determine the best mask. If not provided, the
      function will simply return the first mask in the list.

    Returns:
    - best_mask: A numpy array representing the best segmentation mask
      selected based on agreement with the composite mask. This is the
      mask that has the highest agreement score that exceeds a certain
      threshold, or the mask with the highest agreement if no mask
      exceeds the threshold. The best mask can be used for further
      analysis or comparison with other methods.
    """
   
    n = len(masks)

    if composite is not None:
        mask = revise_mask(masks[0], composite)
    else:
        mask = masks[0]



    maxoverlap = 0.8
    maxindex = 0
    overlaps = [np.nan]
    for i in range(1, n):
        denominator = max(
            np.sqrt(np.sum(masks[i] ** 2) * np.sum(mask ** 2)), 1
        )
        overlap = np.sum(mask * masks[i]) / denominator
        overlaps.append(overlap)
        if overlap > maxoverlap:
            maxoverlap = overlap
            maxindex = i
    

    
    
    if maxindex == 0:
        filename = str(image_path).split("/")[-1]
        print(
            "No good mask found. "
            f"Organoid may not be present in {filename}"
        )
        return None
    else:
        return masks[maxindex]
    
def analyze_image(
    model: SamModel,
    processor: SamProcessor,
    device: int,
    raw_image: Image.Image,
    loc: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    boxes: Optional[np.ndarray] = None,
    multiple: bool = False,
    boxscores: Optional[List[torch.Tensor]] = None,
) -> np.ndarray:
    """
    Analyzes a raw image using the SAM model and processor, with
    optional input points (centroids) or bounding boxes. The function
    prepares the inputs for the SAM model based on the provided
    centroids or bounding boxes, and then runs the model to generate
    segmentation masks. If multiple centroids are provided, they will
    be used as input points for SAM. If bounding boxes are provided,
    they will be used as input boxes for SAM. The function then
    processes the outputs from the SAM model to extract the predicted
    masks and their corresponding scores, and selects the best mask
    based on the scores. The best mask is returned as a numpy array,
    which can be used for further analysis or comparison with other
    segmentation methods. The function can also handle cases where no
    input points or boxes are provided, and will raise an error if
    neither is given.

    Parameters:
    - model: The SAM model to be used for segmentation.
    - processor: The processor for preparing inputs for the SAM model.
    - device: The device on which to run the SAM model.
    - raw_image: The raw image to be analyzed.
    - loc: A tuple representing the location of the input point
      (centroid).
    - boxes: A list of bounding boxes.
    - multiple: A boolean indicating whether multiple input points are
      provided.
    - boxscores: A list of scores for the bounding boxes.

    Returns:
    - best_mask: A numpy array representing the best segmentation mask
      selected based on agreement with the composite mask. This is the
      mask that has the highest agreement score that exceeds a certain
      threshold, or the mask with the highest agreement if no mask
      exceeds the threshold. The best mask can be used for further
      analysis or comparison with other methods.
    """

    if multiple and loc:
        input_points = [loc]
    elif loc:
        input_points = [[loc]]

    if loc is not None:
        inputs = processor(
            raw_image, input_points=input_points, return_tensors="pt"
        ).to(device)
    elif boxes is not None:
        inputs = processor(
            raw_image, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(device)
    else:
        raise RuntimeError(
            "No input points or boxes provided for SAM segmentation."
        )
    
    # pop the pixel_values as they are not needed
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    # print(inputs)
    with torch.no_grad():
        outputs = model(**inputs, multi_label=False)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores.cpu()

    masks = np.array(masks[0])
    if len(scores.shape) == 3:
        scores = np.array(scores.cpu())[0, :, :]

    if boxscores is not None:
        # print("Box Scores:", boxscores)
        for i in range(scores.shape[0]):
            # print(boxscores[i])
            for j in range(scores.shape[1]):
                scores[i, j] += boxscores[0][i]



    

    bestmasks = np.zeros(
        (masks.shape[0], 1, masks.shape[-2], masks.shape[-1])
    )
    bestscores = np.max(scores, axis=1, keepdims=True)
    for i in range(masks.shape[0]):
        j = np.argmax(scores[i, :])
        print(f"Selecting Mask {j + 1} out of {masks.shape[1]}")
        bestmasks[i, 0, :, :] = masks[i, j, :, :]

    if bestmasks.shape[0] == 1:
        finalmask = bestmasks[0, 0, :, :]
    else:
        finalmask = merge_masks(bestmasks, bestscores, show=False)
        # print(finalmask)

    return finalmask


def merge_masks(
    masks: np.ndarray, scores: np.ndarray, show: bool = False
):
    """
    Merges multiple masks generated by SAM into a single mask based
    on their scores and overlaps. The function takes in the array of
    masks and their corresponding scores, and iteratively selects the
    best mask while checking for overlaps with other masks. If a mask
    has a high mean value (indicating it is likely a background mask)
    or overlaps significantly with a higher-scoring mask, it is
    rejected. The accepted masks are combined using a pixel-wise
    maximum operation to create the final merged mask. The function
    can also display the masks and their scores if the show parameter
    is set to True.

    Parameters:
    - masks: A numpy array of shape (N, 1, H, W) containing the N
      masks generated by SAM, where H and W are the height and width
      of the masks.
    - scores: A numpy array of shape (N, 1) containing the scores for
      each of the N masks, which indicate their quality or confidence.
    - show: A boolean flag indicating whether to display the masks
      and their scores during the merging process for debugging
      purposes. If True, the function will print messages about which
      masks are accepted or rejected based on their mean values and
      overlaps, and will show the masks visually. If False, the
      function will perform the merging without any visual output or
      debug messages.

    Returns:
    - finalmask: A numpy array of shape (H, W) containing the final
      merged mask after processing all the input masks and their
      scores.
    """

    finalmask = np.zeros_like(masks[0, 0, :, :])

    n = masks.shape[0]
    for i in range(n):
        mask1 = masks[i, 0, :, :]
        accept = True
        if np.mean(mask1) > 0.5:
            accept = False
            if show:
                print(f"Mask #{i} is a Background Mask!")
        else:
            for j in range(n):
                mask2 = masks[j, 0, :, :]
                if np.mean(mask2) > 0.5:
                    break
                overlap_exists = np.sum(mask1 * mask2) > 0
                if overlap_exists and scores[i] < scores[j]:
                    accept = False
                    if show:
                        print(f"Mask #{i} Overlaps Mask #{j}!")
                    break

        if accept:
            finalmask = np.fmax(finalmask, mask1)
            # print(f"Mask #{i} Accepted into Final GDMask!")

                
    return finalmask
                

def read_centroid(
    string: str, OrganoID: bool = True
) -> Tuple[float, float]:
    """
    Reads a centroid from a string representation, and applies
    scaling if the centroid was generated by OrganoID. The function
    takes in a string representation of a centroid, which is typically
    in the format "(x, y)", and parses it to extract the x and y
    coordinates. If the centroid was generated by OrganoID, it applies
    a scaling factor of 3 to the coordinates to account for the
    difference in resolution between the original images and the images
    used for analysis. The function returns the centroid as a tuple of
    (x, y) coordinates, which can be used as input points for SAM or
    for other analysis purposes.
    """

    loc = string.split(",")
    if OrganoID:
        loc = (float(loc[1][:-1]) * 3, float(loc[0][1:]) * 3)
    else:
        loc = (float(loc[1][:-1]), float(loc[0][1:]))
    return loc

def analyze_directory(directory: str, model_path: str, output_directory: str = "") -> None:
    """
    Analyzes the segmented images in a directory using the OrganoID
    analysis pipeline. The function takes in the path to the directory
    containing the segmented images, and runs the OrganoID analysis
    pipeline on the images to extract properties of the segmented
    regions and save them to an Excel file. The function uses the
    os.system command to execute the OrganoID analysis script with the
    appropriate arguments, including the path to the model, the input
    directory, and the output directory for the analysis results. This
    allows for easy integration of the OrganoID analysis pipeline into
    a larger workflow for processing and analyzing segmented images.
    """

    if not output_directory:
        output_directory = directory + "_oid"

    os.system(
        f"python OrganoID.py run {model_path} "
        f"{directory} {output_directory} "
        f"--no_separation --overlay --belief --analyze --binary "
        f"-A 200 -T 0.99"
    )
    



def get_generator(device: int) -> Any:
    """
    Initializes and returns a SAM mask generation pipeline on the
    specified device. The function takes in the device identifier
    (e.g., 0 for GPU, -1 for CPU) and creates a SAM mask generation
    pipeline using the transformers library. The pipeline is
    configured to use the specified device and to generate masks with
    a batch size of 1 and a data type of float32. This generator can
    be used for generating segmentation masks with SAM in various
    functions, such as the SAM256 function for composite segmentation.
    """

    print("Using Device ", device)
    return pipeline(
        "mask-generation",
        device=device,
        points_per_batch=1,
        torch_dtype=torch.float32,
    )

def SAM256(
    i: int,
    files: List[Union[str, Path]],
    masks: List[Union[str, Path]],
    generator: Optional[Any] = None,
    blur: int = 0,
) -> np.ndarray:
    """
    Generates a segmentation mask for the i-th image in the directory
    using SAM with a composite approach. The function takes in the list
    of image files and corresponding masks, and an optional
    pre-initialized SAM mask generation pipeline. It generates multiple
    masks using SAM and selects the best one based on agreement with
    the original OrganoID mask. If no mask exceeds a certain agreement
    threshold, it returns the mask with the highest agreement. The
    function can also apply blurring to the final mask if specified.

    Parameters:
    - i: The index of the image to process.
    - files: The list of image files in the directory.
    - masks: The list of corresponding OrganoID masks for the images.
    - generator: An optional pre-initialized SAM mask generation
      pipeline to use for generating masks. If None, a new pipeline
      will be created on the default device.
    - blur: The kernel size for blurring the final mask. This is done
      to merge very finely separated masks, since these generally
      correspond to a single, irregularly shaped organoid. If 0, no
      blurring will be applied.

    Returns:
    - A numpy array representing the best segmentation mask generated
      by SAM based on agreement with the original OrganoID mask. The
      mask is selected from the multiple masks generated by SAM, and
      can be further processed with blurring if specified. This mask
      can be used for comparison with other segmentation methods or
      for further analysis.
    """

    image_path = files[i]

    orgomask = Image.open(masks[i])
    orgomask = orgomask / max(np.max(orgomask), 1)
    outputs = generator(str(image_path.resolve()), points_per_batch=1)



    matchingmasks = []

    threshold = 0.6
    maxagreement = 0
    maxmask = np.zeros_like(orgomask)
    j=0
    for mask in outputs["masks"]:
        agreement = np.sum(mask * orgomask) / np.sum(mask ** 2)
        if agreement > maxagreement:
            maxagreement = agreement
            maxmask = mask
        if agreement > threshold:
            matchingmasks.append(mask / np.max(mask))

        j += 1
    if maxagreement < threshold:
        print("No mask exceeded {m} agreement.".format(m=maxagreement))
        return maxmask

    mask = np.zeros_like(outputs["masks"][0])

    for j in range(len(matchingmasks)):
        accepted = True
        for k in range(len(matchingmasks)):
            if j == k:
                break

            overlap_threshold = 0.1 * np.sum(
                matchingmasks[j] * matchingmasks[j]
            )
            overlap = np.sum(matchingmasks[j] * matchingmasks[k])
            overlap = overlap >= overlap_threshold

            better = np.sum(matchingmasks[j] * orgomask) > np.sum(
                matchingmasks[k] * orgomask
            )

            if overlap and not better:
                accepted = False
        
        if accepted:
            mask = np.fmax(mask, matchingmasks[j])

    if np.max(mask) == 0:
        mask = maxmask

    if blur:
        blurred = cv2.blur(mask, ksize=(blur, blur))
        return np.array(mask, bool) | (blurred > 0.5)
    else:
        return mask


def centroid_SAM(
    model: SamModel,
    processor: SamProcessor,
    device: int,
    raw_image: Image.Image,
    centroids: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Generates a segmentation mask for the given raw image using SAM
    with the centroids as input points. The function takes in the SAM
    model and processor, the device to run the model on, the raw image
    to segment, and the list of centroids to use as input points for
    SAM. It returns the best segmentation mask based on the centroids,
    which can be used for comparison with other segmentation methods.
    """

    return analyze_image(
        model=model,
        processor=processor,
        device=device,
        raw_image=raw_image,
        loc=centroids,
        multiple=True,
    )

def OrganoID_only(
    i: int,
    masks: List[Union[str, Path]],
    overlays: List[Union[str, Path]] = [],
) -> np.ndarray:
    """
    Loads the original OrganoID mask for the i-th image in the
    directory, and optionally displays the overlay if provided. The
    mask is returned as a numpy array. This function is used to
    compare the original OrganoID mask with the new masks generated by
    GroundingDINO and SAM, and to visualize the differences between
    them.
    """
    if overlays:
        overlay = Image.open(overlays[i])
        plt.imshow(overlay)
        plt.show()
    return np.array(Image.open(masks[i]))


def gdmask(
    gdmodel: Any,
    gdprocessor: Any,
    model: SamModel,
    processor: SamProcessor,
    device: int,
    i: int,
    files: List[Union[str, Path]],
    gdprompt: str = "A dark, solid cluster.",
) -> np.ndarray:
    """
    Generates a segmentation mask for the i-th image in the directory
    using GroundingDINO for object detection and SAM for segmentation.
    The function takes in the GroundingDINO model and processor, the SAM
    model and processor, the device to run the models on, the list of
    image files, and optional parameters for displaying the results and
    debugging. The function returns the best segmentation mask based on
    the GroundingDINO detections, which can be used for comparison with
    the original OrganoID mask and the centroid-based SAM mask.

    Parameters:
    - gdmodel: The GroundingDINO model to use for object detection.
    - gdprocessor: The GroundingDINO processor to use for preparing
      inputs.
    - model: The SAM model to use for segmentation.
    - processor: The SAM processor to use for preparing inputs.
    - device: The device to run the models on.
    - i: The index of the image to process.
    - files: The list of image files.
    - gdprompt: The prompt to use for GroundingDINO object detection.
    """


    image = Image.open(files[i]).convert("RGB")

    inputs = gdprocessor(
        images=image, text=gdprompt, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = gdmodel(**inputs)

    results = gdprocessor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    if results[0]["boxes"].numel() > 0:
        return analyze_image(
            model=model,
            processor=processor,
            device=device,
            raw_image=image,
            boxes=results[0]["boxes"].cpu(),
            boxscores=[results[0]["scores"].cpu()],
        )
    else:
        return np.zeros((image.size[0], image.size[1]))

def path(pathstring: str):
    """
    Escapes spaces and parentheses in a file path string for use in
    command line operations.
    """
    return (
        pathstring.replace(" ", "\\ ")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def segment_directory(
    directory: Path,
    sam_model: SamModel,
    processor: SamProcessor,
    gdmodel: Any,
    gdprocessor: Any,
    oid_model_path: str = "TrainableModel",
    generator: Optional[Any] = None,
    device: int = 0,
    mode: Optional[str] = None,
    blur256: int = 0,
    output_directory: str = "",
    gdprompt: str = "A dark, solid cluster.",
    analysis_dir: Optional[str] = None,
    revise_OrganoID: bool = False,
) -> None:

    """
    Segment images in a directory using GroundingDINO and/or SAM
    and save the new masks to a specified output directory.

    Parameters:
    - directory: Path to the directory containing the original images.
    - model: The SAM model to use for segmentation.
    - processor: The SAM processor to use for preparing inputs.
    - gdmodel: The GroundingDINO model to use for object detection.
    - gdprocessor: The GroundingDINO processor to use for preparing
      inputs.
    - oid_model_path: The path to the OrganoID model to use for
      analysis. This is needed to run the initial OrganoID analysis on
      the images, which generates the Excel file with centroids and the
      original masks and overlays. The analysis results are used for
      comparison with the new masks generated by GroundingDINO and SAM.
    - generator: An optional pre-initialized SAM mask generation
      pipeline to use for Composite segmentation. If None, a new
      pipeline will be created on the specified device.
    - device: The device to run the models on (e.g., 0 for GPU, -1 for
      CPU).
    - mode: The mode of re-segmentation to perform. Options include
      "GD" for GroundingDINO only, "SAM" for SAM only, "Composite"
      for composite segmentation, and "Centroid" for centroid-based
      SAM. If None, all methods will be run and compared.
    - blur256: The kernel size for blurring the Composite mask. If 0,
      no blurring will be applied.
    - output_directory: An optional path to save the new masks. If
      empty, a default directory will be created based on the input
      directory and mode.
    - gdprompt: The text prompt to use for GroundingDINO object
      detection.
    - analysis_dir: An optional path to the directory containing the
      analysis results (Excel file, masks, overlays) from the initial
      OrganoID run. If None, it will be inferred from the input
      directory.
    - revise_OrganoID: Whether to use the Composite mask to revise the
      original OrganoID mask before comparison. If True, the Composite
      mask will be used to filter the OrganoID mask, and only the
      overlapping regions will be considered for comparison with
      GroundingDINO and Centroid masks.
    """

    print(f"Mode: {mode}")

    # Initialize the SAM mask generation pipeline if not provided.
    # This allows for reusing the same pipeline across multiple
    # directories, which is more efficient than creating a new
    # pipeline for each directory.
    if generator is None:
        generator = pipeline(
            "mask-generation",
            device=0,
            points_per_batch=1,
            torch_dtype=torch.float32,
        )
    

    print("Resegmenting " + str(directory).split("/")[-1])

    # Determine the analysis directory based on the input directory
    # and mode. This directory should contain the Excel file with
    # centroids, as well as the masks and overlays from the initial
    # OrganoID run. If analysis_dir is provided, it will be used
    # directly.
    if analysis_dir is not None:
        analysis_directory = analysis_dir
    else:
        analysis_directory = Path(str(directory)[:-3] + "_analysis")
    print("Analysis Directory:", analysis_directory)

    if not analysis_directory.exists():
        analyze_directory(str(directory), oid_model_path, str(analysis_directory))


    if output_directory == "":
        if mode == "GD":
            output_directory = Path(str(directory)[:-3] + "_GD")
        elif mode == "SAM":
            output_directory = Path(str(directory)[:-3] + "_SAM")
        elif mode == "Composite":
            output_directory = Path(str(directory)[:-3] + "_Composite_SAM")
        elif mode == "Centroid":
            output_directory = Path(str(directory)[:-3] + "_CENTROID")
        else:
            output_directory = Path(
                str(directory)[:-3] + "_hybrid_sam_analysis"
            )

    # Create the output directory if it doesn't exist, and print a
    # message indicating where the new masks will be saved.
    print("Saving New Masks To", output_directory)
    output_directory.mkdir(exist_ok=True)

    # Get the list of image files and corresponding masks from the
    # analysis directory. The image files should be in TIFF format,
    # and the masks should have "binary" in their filename. The Excel
    # file containing the centroids should also be located in the
    # analysis directory.
    files = list(directory.glob("*.TIF"))
    files.sort()

    # Get the Excel file containing the centroids. There should be
    # exactly one Excel file in the analysis directory, and it should
    # contain a sheet named "centroid" with the centroid data for each
    # image. The centroids will be used for the centroid-based SAM
    # segmentation method.
    excelfile = next(analysis_directory.glob("*.xlsx"))
    data = pd.read_excel(
        excelfile, sheet_name="centroid", index_col=0
    )
    print("Excel File:", excelfile)

    # Determine if the analysis was done automatically by OrganoID or
    # manually. Since images are saved at a different resolution than
    # OrganoID uses intentionally and all measurements are counted in
    # pixels, the centroids will be at a different scale if the
    # analysis was done manually. If the Excel file is named
    # "data.xlsx", we will assume it was done manually and apply a
    # scaling factor of 3 to the centroids.
    isOrganoID = str(excelfile).split("/")[-1] != "data.xlsx"

    masks = list(analysis_directory.glob("*binary*"))
    masks.sort()
    # print(len(masks), "Masks Found")

    # Check that the number of image files matches the number of
    # masks. If not, raise an error indicating the mismatch. This is
    # important because we will be processing each image file and its
    # corresponding mask together, so they need to be aligned correctly.
    if len(files) != len(masks):
        num_files = len(files)
        num_masks = len(masks)
        files_dir = str(directory).split("/")[-1]
        masks_dir = str(analysis_directory).split("/")[-1]
        raise RuntimeError(
            f"{num_files} files in {files_dir} does not match "
            f"{num_masks} masks in {masks_dir}"
        )

    # Extract the centroids from the Excel file for each image.
    # The centroids are stored as strings in the Excel file, so we
    # will need to parse them and convert them to tuples of (x, y)
    # coordinates. We will also apply the appropriate scaling factor
    # based on whether the analysis was done automatically by OrganoID
    # or manually. Centroids contains only the centroid closest to the
    # center of the image, while allcentroids contains all centroids
    # found in the image. The centroids will be used for the
    # centroid-based SAM segmentation method, where we will use the
    # centroids as input points for SAM to generate masks.
    centroids = []
    allcentroids = []
    center_x, center_y = 758, 768
    for col in data.columns:
        centroid = (0, 0)
        allcentroids.append([])
        for i in data.index:
            if str(data.at[i, col])[0] == "(":
                newcentroid = read_centroid(data.at[i, col], isOrganoID)

                allcentroids[-1].append(newcentroid)
                dist_new = (newcentroid[0] - center_x) ** 2 + (
                    newcentroid[1] - center_y
                ) ** 2
                dist_curr = (centroid[0] - center_x) ** 2 + (
                    centroid[1] - center_y
                ) ** 2
                if dist_new <= dist_curr:
                    centroid = newcentroid
        centroids.append(centroid)

    # Segment each image in the directory using the specified
    # method(s) and save the new masks to the output directory. For
    # each image file, we will check if there is a corresponding
    # centroid (i.e., if an organoid was found in the image). If there
    # is a centroid, we will proceed with the segmentation based on the
    # specified mode. If there is no centroid and the mode is "SAM",
    # "SAM256", or "Centroid", we will save an empty mask and display
    # the original OrganoID mask for reference. After generating the
    # new mask for each image, we will save it to the output directory
    # with a filename that indicates the original image name and the
    # method used for segmentation.
    for i in range(len(files)):
        filename = str(files[i]).split("/")[-1]

        print(f"Processing {filename} with mode {mode}...")

        if centroids[i] != (0, 0):
            raw_image = Image.open(files[i]).convert("RGB")

            # Generate a mask based on the specified mode. If mode is
            # "GD", we will use GroundingDINO for object detection and
            # SAM for segmentation. If mode is "SAM", we will use SAM
            # with the original OrganoID mask as a reference to generate
            # a new mask. If mode is "Composite", we will use the SAM
            # Composite method. If mode is "Centroid", we will use the
            # centroid-based SAM method. If mode is None, we will run all
            # methods and compare the results.
            if mode == "GD":
                print("Calculating GroundingDINO Mask...")
                bestmask = gdmask(
                    gdmodel=gdmodel,
                    gdprocessor=gdprocessor,
                    model=sam_model,
                    processor=processor,
                    device=device,
                    i=i,
                    files=files,
                    gdprompt=gdprompt,
                )


            elif mode == "SAM":
                print("Generating Masks with OrganoID + SAM only.")
                print("Loading OrganoID Mask...")
                orgomask = OrganoID_only(i, masks=masks)
                print("Calculating Centroid Mask...")
                centroidmask = centroid_SAM(
                    model=sam_model,
                    processor=processor,
                    device=device,
                    raw_image=raw_image,
                    centroids=allcentroids[i],
                )

                print("Calculating SAM Composite Mask...")
                sammask = SAM256(
                    i,
                    files=files,
                    masks=masks,
                    generator=generator,
                    blur=blur256,
                )
                print("Selecting Best Mask...")
                bestmask = compare_masks(
                    files[i],
                    [
                        orgomask / np.max(orgomask),
                        centroidmask,
                        sammask,
                    ],
                )

            elif mode == "SAM256":
                print("Generating Masks with OrganoID + SAM Composite only.")
                print("Loading OrganoID Mask...")
                orgomask = OrganoID_only(i, masks=masks)
                print("Calculating SAM Composite Mask...")
                sammask = SAM256(
                    i,
                    files=files,
                    masks=masks,
                    generator=generator,
                    blur=blur256,
                )
                compare_masks(
                    files[i],
                    [orgomask / np.max(orgomask), sammask],
                )
                bestmask = sammask
                
            elif mode == "Centroid":
                print("Generating Masks with OrganoID Centroid + SAM only.")
                print("Loading OrganoID Mask...")
                orgomask = OrganoID_only(i, masks=masks)
                print("Calculating Centroid Mask...")
                centroidmask = centroid_SAM(
                    model=sam_model,
                    processor=processor,
                    device=device,
                    raw_image=raw_image,
                    centroids=allcentroids[i],
                )

                compare_masks(
                    files[i],
                    [
                        orgomask / max(np.max(orgomask), 1),
                        centroidmask,
                    ],
                )
                bestmask = centroidmask

            else:
                print("Loading OrganoID Mask...")
                orgomask = OrganoID_only(i, masks=masks)
                print("Calculating Centroid Mask...")
                centroidmask = centroid_SAM(
                    model=sam_model,
                    processor=processor,
                    device=device,
                    raw_image=raw_image,
                    centroids=allcentroids[i],
                )
                print("Calculating GroundingDINO Mask...")
                dinomask = gdmask(
                    gdmodel=gdmodel,
                    gdprocessor=gdprocessor,
                    model=sam_model,
                    processor=processor,
                    device=device,
                    i=i,
                    files=files,
                    gdprompt=gdprompt,
                )
                print("Calculating SAM Composite Mask...")
                sammask = SAM256(
                    i,
                    files=files,
                    masks=masks,
                    generator=generator,
                    blur=blur256,
                )

                print("Selecting Best Mask...")

                composite = sammask if revise_OrganoID else None

                bestmask = compare_masks(
                    files[i],
                    [orgomask, dinomask, centroidmask, sammask],
                    composite=composite,
                )

        elif mode in ["SAM", "SAM256", "Centroid"]:
            print(f"No Organoid found in {files[i]}")

        else:
            bestmask = np.zeros_like(OrganoID_only(i, masks=masks))

        if mode == "GD":
            basename = str(files[i]).split("/")[-1][:-4]
            name = Path(output_directory, basename + "_gd_mask.TIF")
        else:
            basename = str(files[i]).split("/")[-1][:-4]
            name = Path(output_directory, basename + "_sam_mask.TIF")
        if bestmask is not None:
            cv2.imwrite(str(name), bestmask.astype(int))
