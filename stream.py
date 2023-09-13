import streamlit as st
import cv2
import numpy as np
from pyngrok import ngrok
import tempfile
from PIL import Image
from numpy import ndarray
import time
from typing import List, Optional, Tuple, Union
from tensorrt_cuda import TRTEngine

ngrok.set_auth_token("2TxUrE2wuWsHeuiJi9nhZ5EV7Xm_57HPbhgwBA4ufgKtPwZGE")

def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (0, 0, 0)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)
def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (0, 0, 0)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)

def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im
        
def det_postprocess(data: Tuple[ndarray, ndarray, ndarray, ndarray]):
    assert len(data) == 4
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    nums = num_dets.item()
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]
    return bboxes, scores, labels    
    
def engine_run(engine, image):
    H, W = engine.inp_info[0].shape[-2:]
    
    # image = cv2.imread(image_path)
    bgr, ratio, dwdh = letterbox(image, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = np.array(dwdh * 2, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor)
    
    # Detection
    results = engine(tensor)
    
    bboxes, scores, labels = det_postprocess(results)
    bboxes -= dwdh
    bboxes /= ratio

    
    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        color = (0,255,0)
        cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        cv2.putText(image,
                f'face:{score:.3f}', (bbox[0], bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, [225, 255, 255],
                thickness=2)
    return image


 

def main():
    engine_path = 'best.engine'
    engine = TRTEngine(engine_path)  
# Create a video file uploader
    st.header("Upload a video")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        # Load the video with cv2
        cap = cv2.VideoCapture(video_path)
        outputing2 = st.empty()
        outputing = st.empty()
                
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            
            output = engine_run(image = frame, engine = engine)
            
            end = time.time()
            # Convert the output to an image that can be displayed
            outputing2.write(f"FPS : {round(1.0 / (end - start) , 2)}" , key = 0)
      
            output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            
            # Display the image
            time.sleep(1.5)
            outputing.image(output_image)

        cap.release() 
    else:
        st.write("Please upload a video file ")
      


if __name__=="__main__":
    main()
    

    
    
    
