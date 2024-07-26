def crop_img(img,bb_dict):
    """
    ---------------------------------------------------------------------------------------------------

    This function takes an RGB image, a Facemesh object, a frame number, and a list of bounding boxes as
    input, runs Facemesh on the image, and returns a Pandas dataframe with the filtered 3D coordinates of
    each facial landmark detected.

    Parameters:
    ............
    rgb_img : numpy.ndarray
        RGB image
    face_mesh : Mediapipe object
        Facemesh object
    frame : int
        Frame number
    bbox_list : list
        List of bounding boxes

    Returns:
    ............
    df_coord : pandas.DataFrame
        Dataframe with the filtered 3D coordinates of each facial landmark detected

    ---------------------------------------------------------------------------------------------------
    """

    x = bb_dict['bb_x']
    y = bb_dict['bb_y']
    w = bb_dict['bb_w']
    h = bb_dict['bb_h']
    roi = img[y:y+h, x:x+w]
    return roi