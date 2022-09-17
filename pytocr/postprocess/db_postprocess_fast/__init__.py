import os
import subprocess
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

def cpp_boxes_from_bitmap(pred, bitmap, 
                          box_thresh, 
                          det_db_unclip_ratio,
                          src_w, src_h,
                          use_padding_resize=False):
    from .db_postprocess import db_postprocess
    bitmap = bitmap.astype(np.uint8)
    bboxes = db_postprocess(pred, bitmap, 
                            box_thresh, 
                            det_db_unclip_ratio, 
                            src_w, src_h, 
                            use_padding_resize)
    return bboxes
    