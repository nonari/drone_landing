import sys

import cv2
from os import path
from glob import glob

import fire
import h5py
import numpy as np

vid_len = {
    43: 7041,
    44: 758,
    45: 817,
    46: 3306,
    47: 1789,
    50: 3451,
    51: 556,
    53: 953,
    56: 2687,
    61: 2143,
    85: 1192,
    86: 3839,
    88: 1212,
    89: 1778,
    93: 856,
    97: 8261,
    101: 702,
    114: 2695,
    116: 2829,
    118: 3975
}

split_test_ids = {'0051', '0056', '0061', '0086', '0088', '0089', '0116'}


def extract_ruralscapes(**kwargs):
    vid = kwargs['vid'] if 'vid' in kwargs else None
    start = kwargs['start'] if 'start' in kwargs else None
    end = kwargs['end'] if 'end' in kwargs else None
    root = '~/ruralscapes/light_videos'
    videos = glob(root+'/*')
    videos = sorted(videos)

    if vid is not None:
        videos = [path.join(root, f'DJI_{vid}.MP4')]

    for v in videos:
        name = path.basename(v).split('.')[0]
        num = name[-4:]
        if num in split_test_ids:
            continue
        prefix = path.join(path.dirname(root), 'flow_farneback', name)
        extract_frames(v, prefix, start, end)


def extract_frames(video_file, prefix, start=None, end=None):
    forward_db = h5py.File(f'{prefix}_forward.h5', 'w')
    backward_db = h5py.File(f'{prefix}_backward.h5', 'w')

    vid_id = int(path.basename(video_file).split('.')[0][4:8])
    if end is None:
        max_frame = vid_len[vid_id]
    else:
        max_frame = end + 1
    forward_db.create_dataset('flow', (max_frame, 1280, 720, 2), chunks=(1, 1280, 720, 2), compression="gzip", compression_opts=4)
    backward_db.create_dataset('flow', (max_frame, 1280, 720, 2), chunks=(1, 1280, 720, 2), compression="gzip", compression_opts=4)
    cap = cv2.VideoCapture(video_file)
    print(prefix)
    is_read, frame = cap.read()
    last_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            print(f'ENDED {prefix} at {count}')
            break
        else:
            if start is not None and count == start - 1:
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if start is None or count >= start:
                print(f'{vid_id}: {count}/{max_frame}')
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                forward_flow = cv2.calcOpticalFlowFarneback(last_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)[..., :2]
                forward_flow = np.around(np.swapaxes(forward_flow, 0, 1), decimals=3)
                forward_db['flow'][count] = forward_flow
                backward_flow = cv2.calcOpticalFlowFarneback(frame, last_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)[..., :2]
                backward_flow = np.around(np.swapaxes(backward_flow, 0, 1), decimals=3)
                backward_db['flow'][count] = backward_flow
                if count == max_frame - 1:
                    break

        last_frame = frame
        count += 1

    forward_db.close()
    backward_db.close()


if __name__ == '__main__':
    fire.Fire(extract_ruralscapes)
