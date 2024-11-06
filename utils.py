from imagededup.methods import PHash, AHash, DHash, WHash, CNN

from functools import wraps
import time
from tqdm import tqdm


VALID_METHODS = {'PHash': PHash, 
                 'AHash': AHash,
                 'DHash': DHash, 
                 'WHash': WHash, 
                 'CNN': CNN
                 }

def timeit(text=''):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'{text}{total_time:.4f}s')
            return result
        return wrapper
    return decorator

def get_duplicate_map(frames, method, max_distance_threshold=10) -> dict:
    """
    Get duplicate map from frames

    Args:
        frames (list): list of frames
        method (str): method to use for finding duplicates
        max_distance_threshold (int): max distance threshold (default: 10)

    Returns:
        dict: duplicate map
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Valid methods are: {', '.join(VALID_METHODS.keys())}")
    
    hasher = VALID_METHODS[method]()
    encodings = {}
    for frame_idx, frame in tqdm(enumerate(frames)):
        encodings.update({str(frame_idx): hasher.encode_image(image_array=frame)})
    duplicates = hasher.find_duplicates(encoding_map=encodings, max_distance_threshold=max_distance_threshold)
    return duplicates

@timeit(text='Time to remove duplicate: ')
def remove_duplicate(video_array, method="PHash", max_distance_threshold=10):
    """
    Remove duplicate from video array

    Args:
        video_array (np.array): video array
        methods (str): method to use for finding duplicates (default: "PHash")
        max_distance_threshold (int): max distance threshold (default: 10)

    Returns:
        tuple: unique frames, duplicates
    """
    duplicates = get_duplicate_map(video_array, method, max_distance_threshold)
    unique_frames = dict(enumerate(video_array))
    for frame_idx, dup_frames in duplicates.items():
        if int(frame_idx) in unique_frames.keys():
            try:
                for dup_frame in dup_frames:
                    unique_frames.pop(int(dup_frame)) 
            except:
                pass
    return unique_frames, duplicates
