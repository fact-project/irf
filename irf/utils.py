def camera_distance_to_angle(theta):
    '''
    Convert a distance in mm in the fact camera
    to a distance in degrees

    This formula contains at two approximations.
    1. The mirror is a perfect parabola
    2. small angle approximation
    '''
    pixelsize = 9.5  # mm
    fov_per_pixel = 0.11  # degree
    return (theta * (fov_per_pixel / pixelsize))**2
