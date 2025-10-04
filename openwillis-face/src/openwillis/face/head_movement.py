import json
import os
import logging
import cv2

import numpy as np
import pandas as pd
import feat 

#from .util import create_cropped_frame, calculate_padding

import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
##################################################
# Head movement extraction using py-feat
##################################################
def get_undetected_facepose(frame_index):
    """
    Return a placeholder face pose when no face is detected in the frame.

    Parameters
    ----------
    frame_index : int
        The frame index in the video.

    Returns
    -------
    np.ndarray
        A numpy array with the following values:
        [frame_index, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    """
    return np.array([frame_index, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])  

def get_facepose(frame_index, frame, detector):
    """
    Extract bounding box and facial landmark coordinates for a frame using py-feat's Detector.

    Parameters
    ----------
    frame_index : int
        The frame index in the video.
    frame : np.ndarray
        The frame image.
    detector : feat.Detector
        An instance of py-feat's Detector class.

    Returns
    -------
    np.ndarray
        A numpy array with the following values:
        [frame_index, bb_x1, bb_y1, bb_x2, bb_y2, face_confidence, pitch, roll, yaw]
        If no face is detected, returns array with NaN values except for frame_index.
    """
    try:
        faceposes = detector.detect_facepose(frame)
        faces = faceposes['faces'][0][0]  # (x1, y1, x2, y2, face_confidence)
        poses = faceposes['poses'][0][0]  # (pitch, roll, yaw)
        return np.hstack(([frame_index], faces, poses))
    except Exception as e:
        logger.info(f'Error processing frame {frame_index}: {str(e)}')
        return get_undetected_facepose(frame_index)

def crop_and_get_facepose(frame_index, frame, detector, bbox, padding_percent=0.1):
    """
    Extract bounding box and facial landmark coordinates for a frame using py-feat's Detector.

    Parameters
    ----------
    frame_index : int
        The frame index in the video.
    frame : np.ndarray
        The frame image.
    detector : feat.Detector
        An instance of py-feat's Detector class.
    bbox : dict
        A dictionary containing the bounding box coordinates of the face.                           
        Keys: ['bb_x1', 'bb_y1', 'w', 'h']

    Returns
    -------
    np.ndarray
        A numpy array with the following values:
        [frame_index, bb_x1, bb_y1, bb_x2, bb_y2, face_confidence, pitch, roll, yaw]
    """
    padded_bbox = calculate_padding(bbox, padding_percent)
    frame = create_cropped_frame(frame, padded_bbox)
    facepose = get_facepose(frame_index, frame, detector)
    bbox_face_pose_fmt = [bbox['bb_x'], bbox['bb_y'], bbox['bb_x']+bbox['bb_w'], bbox['bb_y']+bbox['bb_h'],1]
    facepose[1:6]=bbox_face_pose_fmt

    return facepose

def extract_landmarks_and_bboxes(video_path, frames_per_second=3, bbox_list=[], padding_percent=0.1):
    """
    Extract bounding boxes and facial landmark coordinates from a video using py-feat's Detector,
    sampling at a specified number of frames per second.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    bbox_list : list of dict, optional
        A list of bounding boxes for each frame.
    padding_percent : float, optional
        Amount of padding to add around the bounding box as a percentage of its size.
        Default is 0.1 (10%).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
            ['frame', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'face_confidence', 'landmarks']
        - 'frame': The frame index in the video.
        - 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2': The bounding box coordinates of the detected face.
        - 'face_confidence': A float indicating the detection confidence from py-feat.
        - 'landmarks': A list of (x, y) tuples for each facial landmark detected.
    """
# ok how should we split this up. One way is just to create two separate methods
# here - yeah that's right because we don't want to initialize the detector and for mediapipe
# we can do more efficient batched methods.
# except the mp also goes row by row so it likely doesn't make a  huge difference 
    detector = feat.Detector()
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    skip_interval = max(1, int(video_fps / frames_per_second))

    frame_index = 0
    faces_list = []
    bbox_list_len = len(bbox_list)

    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        if bbox_list_len > 0 and num_frames != bbox_list_len:
            raise ValueError('Number of frames in video and number of bounding boxes do not match')
        
        while True:
            try:
                
                ret, frame = cap.read()
                if not ret:
                    break  

                if frame_index % skip_interval == 0:
                    if len(bbox_list)!=0:
                        facepose = crop_and_get_facepose(
                            frame_index,
                            frame,
                            detector,
                            bbox_list[frame_index],
                            padding_percent=padding_percent
                        )
                    else:
                        facepose = get_facepose(frame_index, frame, detector)
                else:
                    facepose = get_undetected_facepose(frame_index) 
            
            except Exception as e:
                logger.info(f'error processing frame: {frame_index} in file: {video_path} & Error: {e}')
                facepose = get_undetected_facepose(frame_index) 
            
            frame_index += 1
            faces_list.append(facepose)

    except Exception as e:
        logger.info(f'error processing file: {video_path} & Error: {e}')
 

    cap.release()

    out_array = np.array(faces_list)
    
    out_df = pd.DataFrame(
        out_array,
        columns=[
        'frame',
        'bb_x1',
        'bb_y1',
        'bb_x2',
        'bb_y2',
        'face_confidence',
        'pitch',
        'roll',
        'yaw']
    )

    out_df['time'] = out_df['frame']/video_fps
    
    return out_df

def compute_face_centers(df):
    """Computes the center coordinates of bounding boxes.
        Parameters:
        ----------
            df (pd.DataFrame): DataFrame containing bounding box coordinates ('bb_x1', 'bb_x2', 'bb_y1', 'bb_y2').
        
        Returns:
        ----------
            pd.DataFrame: DataFrame with added 'face_center_x' and 'face_center_y' columns representing the center coordinates.
    """
    # @TODO: also add z that is nan
    df['face_center_x'] = df[['bb_x1', 'bb_x2']].mean(axis=1)
    df['face_center_y'] = df[['bb_y1', 'bb_y2']].mean(axis=1)

    return df

####################################################
# Mediapipe head movement extraction
####################################################
def init_facemesh():
    """
    ---------------------------------------------------------------------------------------------------

    This function initializes a Facemesh object from the Mediapipe library, with a minimum detection
    confidence of 0.5. It returns the Facemesh object.

    Parameters:
    ............
    None

    Returns:
    ............
    face_mesh : Mediapipe object
        Facemesh object with minimum detection confidence of 0.5

    ---------------------------------------------------------------------------------------------------
    """

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=2)
    return face_mesh

def filter_landmarks(col_name, keypoints):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes the column name and landmark keypoints detected by Facemesh as inputs, and
    returns a Pandas dataframe with the filtered landmarks in the specified column.

    Parameters:
    ............
    col_name : str
        Column name to filter landmarks into
    keypoints : dict
        Landmark keypoints detected by Facemesh

    Returns:
    ............
    df : pandas.DataFrame
        Dataframe with the filtered landmarks in the specified column

    ---------------------------------------------------------------------------------------------------
    """

    col_list = list(range(0, 468))
    cols = ['lmk' + str(s+1).zfill(3) + '_' + col_name for s in col_list]

    item = list(map(lambda d: d[col_name], keypoints['landmark']))
    df = pd.DataFrame([item], columns=cols)

    return df

def get_column():
    """
    ---------------------------------------------------------------------------------------------------

    This function returns an empty Pandas dataframe with columns corresponding to the 468 facial landmark
    coordinates, labeled with the landmark number and x/y/z coordinate.

    Parameters:
    ............
    None

    Returns:
    ............
    df : pandas.DataFrame
        Empty dataframe with columns for each facial landmark coordinate

    ---------------------------------------------------------------------------------------------------
    """

    col_list = list(range(0, 468))
    col_name = []

    value = [np.nan] * 468 * 3
    lmk_cord = ['x', 'y', 'z']

    for coord in lmk_cord:
        cols = ['lmk' + str(s + 1).zfill(3) + '_' + coord for s in col_list]
        col_name.extend(cols)

    df = pd.DataFrame([value], columns = col_name)
    return df

def create_keypoints_df(face_landmarks):
    '''
    Create a dictionary of keypoints from the landmarks.'''

    
    keypoints = protobuf_to_dict(face_landmarks)

    if len(keypoints)>0:
        df_x = filter_landmarks('x', keypoints)
        df_y = filter_landmarks('y', keypoints)
        df_z = filter_landmarks('z', keypoints)
        df_coord = pd.concat([df_x, df_y, df_z], axis=1)
    return df_coord

def _get_landmark_dataframes(result):
    """Return a list of per-face landmark dataframes from a FaceMesh result."""

    if result is None:
        return []

    landmarks = getattr(result, 'multi_face_landmarks', None)
    if not landmarks:
        return []

    return [create_keypoints_df(face_landmarks) for face_landmarks in landmarks]


def _select_face_by_bbox(df_list, bbox):
    """Pick the landmark dataframe whose face center is closest to the bbox center."""

    if not df_list or bbox is None:
        return None

    bbox_center_x = (bbox['bb_x'] + bbox['bb_x'] + bbox['bb_w']) / 2
    bbox_center_y = (bbox['bb_y'] + bbox['bb_y'] + bbox['bb_h']) / 2

    min_distance = float('inf')
    selected_df = None

    for df in df_list:
        if 'lmk001_x' not in df.columns or 'lmk001_y' not in df.columns:
            continue

        face_center_x = df['lmk001_x'].values[0]
        face_center_y = df['lmk001_y'].values[0]

        distance = ((face_center_x - bbox_center_x) ** 2 + (face_center_y - bbox_center_y) ** 2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            selected_df = df

    return selected_df


def filter_coord(result, bbox=None):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes the output from a Facemesh object and returns a Pandas dataframe with the filtered
    3D coordinates of each facial landmark detected.

    Parameters:
    ............
    result : Mediapipe object
        Output from a Facemesh object
    bbox : dict, optional
        Bounding box used to disambiguate between multiple detected faces.

    Returns:
    ............
    df_coord : pandas.DataFrame
        Dataframe with the filtered 3D coordinates of each facial landmark detected

    ---------------------------------------------------------------------------------------------------
    """
    df_coord = get_column()
    df_coord_list = _get_landmark_dataframes(result)

    if not df_coord_list:
        return df_coord

    if bbox is None:
        return df_coord_list[0]

    selected_df = _select_face_by_bbox(df_coord_list, bbox)
    return selected_df if selected_df is not None else df_coord

def process_and_format_face_mesh(img, face_mesh, df_common, bbox=None):
    """
    Process the given image using the face_mesh model and format the resulting face landmarks.

    Args:
        img (numpy.ndarray): The input image.
        face_mesh: The face_mesh model.
        df_common (pandas.DataFrame): The common dataframe.

    Returns:
        pandas.DataFrame: The formatted dataframe containing the face landmarks.
    """
    result = face_mesh.process(img)
    df_coord = filter_coord(result, bbox=bbox)
    df_landmark = pd.concat([df_common, df_coord], axis=1)
    return df_landmark




def run_facemesh(path, frames_per_second=3, bbox_list=[]):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to an image file as input, runs Facemesh on the image, and returns a list
    of dataframes containing the landmark coordinates for each frame of the video.

    Parameters:
    ............
    path : str
        Path to image file
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    bbox_list : list, optional
        List of bounding boxes for each frame in the video

    Returns:
    ............
    df_list : list
        List of dataframes containing landmark coordinates for each frame of the video

    ---------------------------------------------------------------------------------------------------
    """

    df_list = []
    frame = 0

    try:
        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        len_bbox_list = len(bbox_list)
        
        # Calculate skip interval to get desired frames per second
        if frames_per_second >= video_fps:
            skip_interval = 0
        else:
            skip_interval = max(0, int(video_fps / frames_per_second))
        
        face_mesh = init_facemesh()
        bbox_list_passed = len_bbox_list > 0

        if bbox_list_passed & (num_frames != len_bbox_list):
            raise ValueError('Number of frames in video and number of bounding boxes do not match')

        n_frames_skipped = skip_interval

        while True:
            try:
                ret_type, img = cap.read()
                
                if ret_type is not True:
                    break

                if n_frames_skipped < skip_interval:
                    n_frames_skipped += 1
                    df_landmark = get_undected_markers(frame, video_fps)

                elif n_frames_skipped == skip_interval:
                    n_frames_skipped = 0
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    df_common = pd.DataFrame([[frame, frame/video_fps]], columns=['frame', 'time'])
                    
                    # one liner for bbox list
                    bbox = bbox_list[frame] if bbox_list_passed else None
                    df_landmark = process_and_format_face_mesh(
                        img_rgb,
                        face_mesh,
                        df_common,
                        bbox=bbox
                    )

            except Exception as e:
                logger.info(f'error processing frame: {frame} in file: {path} & Error: {e}')
                df_landmark = get_undected_markers(frame, video_fps)

            df_list.append(df_landmark)
            frame += 1

    except Exception as e:
        logger.info(f'Face error process file in facemesh for file:{path} & Error: {e}')

    finally:
        # Empty dataframe in case of insufficient datapoints
        if len(df_list) == 0:
            df_landmark = get_empty_dataframe()
            df_list.append(df_landmark)
            logger.info(f'Face not detected by facemesh in: {path}')

    return df_list, skip_interval

def get_undected_markers(frame,fps):
    """
    ---------------------------------------------------------------------------------------------------

    This function creates a dataframe with NaN values representing facial landmarks that were not detected
    in a frame of the video.

    Parameters:
    ............
    frame : int
        Frame number
    fps : int
        Frames per second of the video

    Returns:
    ............
    df_landmark : pandas.DataFrame
        Dataframe with NaN values for undetected facial landmarks in a frame of the video

    ---------------------------------------------------------------------------------------------------
    """
    df_common = pd.DataFrame([[frame, frame/fps]], columns=['frame','time'])
    df_coord = get_column()

    col_list = list(range(0, 468))
    cols_x = ['lmk' + str(s+1).zfill(3) + '_x' for s in col_list]
    cols_y = ['lmk' + str(s+1).zfill(3) + '_y' for s in col_list]
    cols_z = ['lmk' + str(s+1).zfill(3) + '_z' for s in col_list]

    cols = cols_x + cols_y + cols_z
    df_coord.columns = cols

    df_landmark = pd.concat([df_common, df_coord], axis=1)
    return df_landmark

def get_empty_dataframe():
    """
    ---------------------------------------------------------------------------------------------------

    This function creates an empty dataframe containing columns for frame number, landmark position
    variables, and overall displacement measurement.

    Parameters:
    ............
    None

    Returns:
    ............
    empty_df : pandas.DataFrame
        Empty displacement dataframe

    ---------------------------------------------------------------------------------------------------
    """
    columns = ['frame','time'] + ['lmk' + str(col+1).zfill(3) for col in range(0, 468)] + ['overall']
    empty_df = pd.DataFrame(columns=columns)
    return empty_df

def get_landmarks(path, frames_per_second=3, bbox_list=[]):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to an image file as input, runs Facemesh on the image, and returns a
    dataframe containing the landmark coordinates for each frame of the video.

    Parameters:
    ............
    path : str
        Path to image file
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    bbox_list : list, optional
        List of bounding boxes for each frame in the video

    Returns:
    ............
    df_landmark : pandas.DataFrame
        Dataframe containing landmark coordinates for each frame of the video

    ---------------------------------------------------------------------------------------------------
    """

    landmark_list, skip_interval = run_facemesh(
        path,
        bbox_list=bbox_list,
        frames_per_second=frames_per_second
    )

    if len(landmark_list) > 0:
        df_landmark = pd.concat(landmark_list).reset_index(drop=True)
    else:
        df_landmark = get_empty_dataframe()

    return df_landmark, skip_interval


# -----------------------------------------------------------------------------
# Reference strategies
# -----------------------------------------------------------------------------
import warnings
from pathlib import Path
from typing import Iterable, Sequence, Literal, Tuple, List

import numpy as np
import pandas as pd

try:
    from scipy.spatial.transform import Rotation as _SciPyRot
    _HAS_SCIPY = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_SCIPY = False

# Optional, only if canonical .obj loading requested
try:
    import trimesh  # type: ignore
    _HAS_TRIMESH = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_TRIMESH = False

# -----------------------------------------------------------------------------
# Landmark subsets
# -----------------------------------------------------------------------------
_DEFAULT_IDS: Tuple[int, ...] = (2, 34, 264, 169, 61,  341, 112, 10)

RefStrategy = Literal["first", "auto", "canonical"]
def _euler_xyz(Rmat: np.ndarray, *, degrees: bool = True) -> Tuple[float, float, float]:
    """Convert 3×3 *Rmat* to xyz Tait‑Bryan angles (rx, ry, rz)."""
    return tuple(_SciPyRot.from_matrix(Rmat).as_euler("xyz", degrees=degrees))



# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def _row_to_pts(row: pd.Series, ids: Sequence[int]) -> np.ndarray:
    return np.asarray([[row[f"lmk{idx:03d}_x"], -row[f"lmk{idx:03d}_y"], row[f"lmk{idx:03d}_z"]] for idx in ids])

def _build_reference(
    landmarks_df: pd.DataFrame,
    ids: Sequence[int],
    strategy: RefStrategy,
    n_ref: int = 5,
    n_auto: int = 300,
    canonical_path: Path | None = None,
) -> np.ndarray:
    """Return reference landmarks array (len(ids)×3)."""
    req_cols = [f"lmk{idx:03d}_{ax}" for idx in ids for ax in ("x", "y", "z")]

    if strategy == "first":
        frames = landmarks_df[landmarks_df[req_cols].notna().all(1)].head(n_ref)
        if frames.empty:
            raise RuntimeError("No complete frame found for reference build")
        return np.stack([_row_to_pts(r, ids) for _, r in frames.iterrows()]).mean(0)

    if strategy == "canonical":
        if canonical_path is None:
            raise ValueError("canonical reference requires --canonical_csv/obj")
        can_df = pd.read_csv(canonical_path)
        if can_df.empty:
            raise ValueError("Canonical reference file has no rows")

        li, ri = 34, 264  # default eye-outer corners indices
        wide_format = all(col in can_df.columns for col in req_cols)
        xyz_format = {"x", "y", "z"}.issubset(can_df.columns)

        if not (wide_format or xyz_format):
            raise ValueError(
                "Canonical CSV must either contain lmk*** columns or simple x,y,z columns"
            )

        if wide_format:
            row = can_df.iloc[0]
            ref_pts = _row_to_pts(row, ids)

            def _canonical_point(idx: int) -> np.ndarray:
                return np.array(
                    [
                        row[f"lmk{idx:03d}_x"],
                        row[f"lmk{idx:03d}_y"],
                        row[f"lmk{idx:03d}_z"],
                    ],
                    dtype=float,
                )

        else:  # xyz_format
            arr = can_df[["x", "y", "z"]].to_numpy(dtype=float)
            max_idx = max(ids)
            if arr.shape[0] <= max_idx:
                raise ValueError(
                    "Canonical mesh CSV must supply ≥ max landmark index rows"
                )
            ref_pts = arr[np.array(ids, dtype=int)].copy()
            ref_pts[:, 1] *= -1.0  # match _row_to_pts camera-space convention

            def _canonical_point(idx: int) -> np.ndarray:
                return arr[idx]

        scale = 1.0
        if li in ids and ri in ids:
            can_d = np.linalg.norm(_canonical_point(li) - _canonical_point(ri))
            if can_d:
                valid = landmarks_df[landmarks_df[req_cols].notna().all(1)]
                if valid.empty:
                    raise RuntimeError(
                        "No complete frame found for canonical reference scaling"
                    )
                df_first = valid.iloc[0]
                cur_li = np.array(
                    [
                        df_first[f"lmk{li:03d}_x"],
                        df_first[f"lmk{li:03d}_y"],
                        df_first[f"lmk{li:03d}_z"],
                    ],
                    dtype=float,
                )
                cur_ri = np.array(
                    [
                        df_first[f"lmk{ri:03d}_x"],
                        df_first[f"lmk{ri:03d}_y"],
                        df_first[f"lmk{ri:03d}_z"],
                    ],
                    dtype=float,
                )
                cur_d = np.linalg.norm(cur_li - cur_ri)
                scale = cur_d / can_d if can_d else 1.0
        return ref_pts * scale

    raise ValueError("Unknown reference strategy")

def _procrustes_rigid(
    A: np.ndarray, B: np.ndarray, with_scaling: bool = False
) -> Tuple[np.ndarray, float]:
    """Return best-fit rotation (and optional scale) from A → B."""
    centroid_A = A.mean(0)
    centroid_B = B.mean(0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, s, Vt = np.linalg.svd(H)
    Rmat = Vt.T @ U.T
    if np.linalg.det(Rmat) < 0:
        Vt[2] *= -1
        Rmat = Vt.T @ U.T
    if with_scaling:
        var_A = (AA**2).sum()
        scale = (s @ np.ones_like(s)) / var_A
    else:
        scale = 1.0
    return Rmat, scale

# <existing _euler_xyz and _row_to_pts unchanged>

# -----------------------------------------------------------------------------
# Public pose estimator
# -----------------------------------------------------------------------------

def estimate_pose(
    landmarks_df: pd.DataFrame,
    landmark_ids: Iterable[int] | None = None,
    *,
    ref_strategy: RefStrategy = "first",
    n_reference_frames: int = 5,
    n_auto: int = 300,
    canonical_mesh: Path | None = None,
    use_procrustes_scale: bool = False,
) -> pd.DataFrame:
    """Return pitch, yaw, roll per frame (optionally using full Procrustes)."""
    ids = tuple(landmark_ids) if landmark_ids is not None else _DEFAULT_IDS
    ref_pts = _build_reference(
        landmarks_df, ids, ref_strategy, n_reference_frames, n_auto, canonical_mesh
    )
    req_cols = [f"lmk{idx:03d}_{ax}" for idx in ids for ax in ("x", "y", "z")]
    pose_rows = []
    skipped = 0
    for _, row in landmarks_df.iterrows():
        if row[req_cols].isna().any():
            pose_rows.append((np.nan, np.nan, np.nan))
            skipped += 1
            continue
        cur_pts = _row_to_pts(row, ids)

        Rm, _ = _procrustes_rigid(cur_pts, ref_pts, with_scaling=use_procrustes_scale)
        rx, ry, rz = _euler_xyz(Rm, degrees=True)

        detected_pitch = rz
        detected_yaw = -rx
        detected_roll = -ry

        pose_rows.append((detected_pitch, detected_yaw, detected_roll))


    if skipped:
        warnings.warn(f"Skipped {skipped} incomplete frames")
    return pd.DataFrame(pose_rows, columns=["pitch_deg", "yaw_deg","roll_deg"], index=landmarks_df.index)

#################################################
# Head movement calculations
#################################################

def get_fw_displacement(sampled_frames, include_z=False):
    """Calculates the frame-wise displacement in the x-y plane or x-y-z space.

    Args:
        sampled_frames (pd.DataFrame): A DataFrame containing the bounding box center coordinates
            in the 'face_center_x' and 'face_center_y' columns for each frame.
        include_z (bool): If True, includes z-coordinate ('face_center_z') in displacement calculation.

    Returns:
        pd.Series: A Series containing the frame-wise displacement magnitudes.
            Returns an empty series if the input DataFrame has fewer than 2 rows.
    """
    coords = ['face_center_x', 'face_center_y']
    if include_z and 'face_center_z' in sampled_frames.columns:
        coords.append('face_center_z')
    
    # Drop rows where any required coordinate is null
    valid_frames = sampled_frames[coords].dropna()
    
    if len(valid_frames) < 2:
        return pd.Series()
        
    displacement = valid_frames.diff()
    
    displacement_magnitudes = (displacement**2).sum(axis=1)**0.5
    
    return displacement_magnitudes

def compute_rotation_angles_vectorized(pitch: np.ndarray, yaw: np.ndarray, roll: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """
    Computes the total rotation angles for multiple sets of pitch, yaw, and roll angles in degrees,
    with correct matrix multiplication order. Calculates total angle following openface convention:
        https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format#featureextraction

    Args:
    - pitch (np.ndarray): Array of rotations about the X-axis in degrees (pitch).
    - yaw (np.ndarray): Array of rotations about the Y-axis in degrees (yaw).
    - roll (np.ndarray): Array of rotations about the Z-axis in degrees (roll).
    - order (str): Rotation order (e.g., "XYZ" means pitch->yaw->roll).

    Returns:
    - np.ndarray: Array of total rotation angles in degrees.
    """
    # Convert degrees to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)

    # Compute cosines and sines in batch
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)

    # Define batch rotation matrices (shape: (N, 3, 3))
    # Pitch around X-axis
    R_x = np.stack([
        np.stack([np.ones_like(pitch_rad), np.zeros_like(pitch_rad), np.zeros_like(pitch_rad)], axis=-1),
        np.stack([np.zeros_like(pitch_rad), cos_p, -sin_p], axis=-1),
        np.stack([np.zeros_like(pitch_rad), sin_p, cos_p], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    # Yaw around Y-axis
    R_y = np.stack([
        np.stack([cos_y, np.zeros_like(yaw_rad), sin_y], axis=-1),
        np.stack([np.zeros_like(yaw_rad), np.ones_like(yaw_rad), np.zeros_like(yaw_rad)], axis=-1),
        np.stack([-sin_y, np.zeros_like(yaw_rad), cos_y], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    # Roll around Z-axis
    R_z = np.stack([
        np.stack([cos_r, -sin_r, np.zeros_like(roll_rad)], axis=-1),
        np.stack([sin_r, cos_r, np.zeros_like(roll_rad)], axis=-1),
        np.stack([np.zeros_like(roll_rad), np.zeros_like(roll_rad), np.ones_like(roll_rad)], axis=-1)
    ], axis=1)  # Shape (N, 3, 3)

    # Apply rotations in specified order correctly
    # Convention: R = Rx * Ry * Rz (pitch -> yaw -> roll)
    order_map = {
        "XYZ": lambda: np.einsum("nij,njk,nkl->nil", R_x, R_y, R_z),  # pitch -> yaw -> roll
        "YXZ": lambda: np.einsum("nij,njk,nkl->nil", R_y, R_x, R_z),  # yaw -> pitch -> roll
        "ZXY": lambda: np.einsum("nij,njk,nkl->nil", R_z, R_x, R_y),  # roll -> pitch -> yaw
        "ZYX": lambda: np.einsum("nij,njk,nkl->nil", R_z, R_y, R_x),  # roll -> yaw -> pitch
    }

    if order not in order_map:
        raise ValueError("Invalid rotation order. Use 'XYZ', 'YXZ', 'ZXY', or 'ZYX'.")

    R = order_map[order]()  # Shape: (N, 3, 3)

    # Compute the total rotation angles from the trace of R
    trace_R = np.einsum("nii->n", R)  
    rotation_angles = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0))  

    return np.degrees(rotation_angles)  # Convert to degrees

def compute_fw_disp(sampled_df, normalize_by_bb_size=False):
    """Computes the xy displacement of the face in the video.
        Parameters:
        ----------
            sampled_df (pd.DataFrame): Dataframe containing the face detections.
            normalize_by_bb_size (bool): Whether to normalize the displacement by the bounding box size.
        
        Returns:
        ----------
            pd.DataFrame: Dataframe with the xy displacement added.
        """

    sampled_df['fw_disp'] = get_fw_displacement(sampled_df)

    if normalize_by_bb_size:
        sampled_df['fw_disp'] /= (sampled_df['bb_x2'] - sampled_df['bb_x1']).abs()

    return sampled_df

def compute_summary_stats(df):
    """Computes the mean and standard deviation of selected metrics.
        Parameters:
        ----------
            df (pd.DataFrame): DataFrame containing the computed metrics.
        
        Returns:
        ----------
            pd.DataFrame: DataFrame containing the mean and standard deviation of selected metrics.
    """

    stats_cols = ['xy_disp', 'pitch', 'yaw', 'roll', 'euclidean_angle_disp']
    mean_series = df[stats_cols].mean()
    std_series = df[stats_cols].std()

    summary_df = pd.DataFrame({
        'xy_disp_mean': [mean_series['xy_disp']],
        'pitch_mean': [mean_series['pitch']],
        'yaw_mean': [mean_series['yaw']],
        'roll_mean': [mean_series['roll']],
        'euclidean_angle_disp_mean': [mean_series['euclidean_angle_disp']],
        'xy_disp_std': [std_series['xy_disp']],
        'pitch_std': [std_series['pitch']],
        'yaw_std': [std_series['yaw']],
        'roll_std': [std_series['roll']],
        'euclidean_angle_disp_std': [std_series['euclidean_angle_disp']]
    })
    return summary_df

#################################################
# Main function to extract head movement
#################################################
def head_movement(video_path, method = 'mediapipe', frames_per_second=3, normalize_by_bb_size=False, bbox_list=[], padding_percent=0.1):
    """
    Extract bounding boxes and facial landmark coordinates from a video using py-feat's Detector,
    sampling at a specified number of frames per second, and compute various head movement metrics.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    method : str, optional
        The method to use for face detection and landmark extraction. Default is 'mediapipe', also accepts 'pyfeat'.
    frames_per_second : float, optional
        The number of frames to sample per second of video. Default is 3.
        This determines the temporal resolution of the analysis.
    normalize_by_bb_size : bool, optional
        If True, normalize the xy displacement by the bounding box width. Default is False.
    bbox_list : list of dict, optional
        A list of bounding boxes, each represented as a dictionary with keys:
        ['bb_x1', 'bb_y1', 'w', 'h']. If provided, these bounding boxes will be used 
        instead of detecting them.
    padding_percent : float, optional
        Amount of padding to add around the bounding box as a percentage of its size.
        Default is 0.1 (10%).

    Returns
    -------
    out_df : pd.DataFrame
        DataFrame containing per-frame results including bounding boxes, landmarks,
        displacement, and computed rotation metrics.
    summary_df : pd.DataFrame
        A one-row DataFrame summarizing mean and std of selected metrics.
    """
    ##
    
    if method == 'pyfeat':
        out_df = extract_landmarks_and_bboxes(
            video_path,
            frames_per_second=frames_per_second,
            bbox_list=bbox_list,
            padding_percent=padding_percent
        )
        out_df = compute_face_centers(out_df)
    elif method == 'mediapipe':
        out_df, skip_interval = get_landmarks(
            video_path,
            frames_per_second=frames_per_second,
            bbox_list=bbox_list
        )
        return out_df
    #  out_df = pd.DataFrame(
    #     out_array,
    #     columns=[
    #     'frame',
    #     'bb_x1',
    #     'bb_y1',
    #     'bb_x2',
    #     'bb_y2',
    #     'face_confidence',
    #     'pitch',
    #     'roll',
    #     'yaw']
    # )

    

    

# df should have face center x and y, (sometimes z if mediapipe)
    sampled_frames = out_df.copy()
    #### compute displacement
    sampled_frames = compute_fw_disp(
        sampled_frames, 
        normalize_by_bb_size=normalize_by_bb_size
    )
    out_df['xy_disp'] = sampled_frames['xy_disp']

    ### compute angle displacement
    out_df['euclidean_angle'] = compute_rotation_angles_vectorized(
        out_df['pitch'], 
        out_df['yaw'], 
        out_df['roll']
    )
    
    sampled_angles = out_df['euclidean_angle'].dropna()
    
    out_df['euclidean_angle_disp'] = sampled_angles.diff().abs()

    

    summary_df = compute_summary_stats(out_df)

    return out_df, summary_df
