import cv2
import yaml
import pandas as pd
import time
from ultralytics import YOLO
from PIL import Image
import cv2
# from cv2 import imread
import time
import numpy as np
import pandas as pd
import os

def scale_bounding_box(x1, y1, x2, y2, scale_factor=0.5):
    try:
        # Calculate the center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate the original width and height
        width = x2 - x1
        height = y2 - y1

        # Scale the width and height
        new_width = width * scale_factor
        new_height = height * scale_factor

        # Calculate the new top-left and bottom-right coordinates
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        return int(new_x1), int(new_y1), int(new_x2), int(new_y2)
    except:
        print("ERROR",x1, y1, x2, y2)
        return x1, y1, x2, y2

def crop_black_bars(image,cutpercent=0.24,targetsize=640,grid=4):
    height, width, _ = image.shape
    sq = int(targetsize/grid)
    bar_height = int(height * cutpercent)  # 10% of image height
    cropped_image = image[bar_height:height - bar_height, :]
    resized_image = Image.fromarray(cropped_image).resize((sq, sq))
    return cropped_image

def getindice(fin = 5,fps = 25, bufferseconds = 16, grid=16,  step = 1):
    # gapframe = step * fin
    # windowtime = step * fin/fps*bufferseconds
    # print(f"Gap {gapframe/fps}s / {gapframe} frames  X16= windowtime {windowtime}")
    start = (bufferseconds*25/5)-1
    array = np.arange(start, start - step * grid, -step)
    reversed_array = array[::-1]
    # print(reversed_array)
    return reversed_array


def createSeq(seq_frame=None,list_seq_frame=None,frame_counter=0,
                    cropSQ=0.24, gridtotaltime = 16,
                    fps=25,length=None,windows = [16,10,7],
                    frame_interval=5,verbose=False):
        grid_collector = []
        if int(frame_counter % frame_interval == 0):
            crop_frame = crop_black_bars(seq_frame,cropSQ)
            list_seq_frame.append(crop_frame)
        # print("frame_counter",self.frame_counter)
        if len(list_seq_frame) >  gridtotaltime*int(fps/frame_interval)+1:
            # print("##########################")
            for window in windows:
                Tgrid = []
                step = int(window/(gridtotaltime*int(fps/frame_interval)+1))
                print(f"Step SQ{step}")
                for dx in getindice(step=step):
                    Tgrid.append(list_seq_frame[dx])
           
                grid = np.concatenate([np.concatenate(Tgrid[j:j+4], axis=1) for j in range(0, 16, 4)], axis=0)
                # Save the grid as a JPEG image
                grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                grid = Image.fromarray(grid)
                grid = grid.resize((640,640))
                grid_collector.append(grid)
             
        # if len(grid_collector) > 0:
        #     print("************************")

        return list_seq_frame,grid_collector


def createSeqCUSTOM(seq_frame=None,drawobjs=[],list_seq_frame=None,frame_counter=0,
                    cropSQ=0.24, gridtotaltime = 32,
                    fps=25,length=None,windows = [16,10,7,3],
                    frame_interval=5,verbose=False,
                    CustomGap = [-148, -128, -108, -88, 
                                 -68, -58, -48, -38,
                                   -28, -23, -18, -13,
                                     -7, -5, -3, -1]):
        seq_frame = draw_boxes_for_seq(seq_frame,drawobjs)
        grid_collector = []
        if int(frame_counter % frame_interval == 0):
            crop_frame = crop_black_bars(seq_frame,cropSQ)
            list_seq_frame.append(crop_frame)
        # print("frame_counter",self.frame_counter)
        if len(list_seq_frame) >  gridtotaltime*int(fps/frame_interval):
            # print("################w##########")
            for window in windows:
                lengS = window/gridtotaltime
                Tgrid = []
                # print("Crafting Windows:________________________ ", window, lengS)
                
                for c in CustomGap:
                    Tgrid.append(list_seq_frame[c])

                grid = np.concatenate([np.concatenate(Tgrid[j:j+4], axis=1) for j in range(0, 16, 4)], axis=0)
                # Save the grid as a JPEG image
                grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                grid = Image.fromarray(grid)
                grid = grid.resize((640,640))
                grid_collector.append(grid)
             
        # if len(grid_collector) > 0:
        #     print("************************")

        return list_seq_frame,grid_collector

# Load the data.yaml file
def load_yaml_namemap(file_path):
    with open(file_path, 'r') as file:
        namemap = yaml.safe_load(file)
    return namemap
# Function to get the name from the id
def get_names(id, namemap):
    try:
        return namemap['names'][id]
    except KeyError:
        return None

def loadYOLO(MDL_path,gpu_idx,name="best.pt"):

    model = YOLO(f"{MDL_path}/weights/{name}")
    try:
        yaml_file_path = f'{MDL_path}/data.yaml'  # Replace with your file path
        namemap = load_yaml_namemap(yaml_file_path)
        names = [item[key] for item in namemap['names'] for key in item]
    except:
        # Create a 480x640 3-channel (RGB) image canvas with dtype uint8
        print("not data.yaml supplied, getting names from model")
        canvas_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in model(canvas_rgb, verbose=False):
            DICT = x.names
            names = list(DICT.values())
        # print(names)
    model.cuda(device=gpu_idx)

    return model, names

def getDet(frameinput,model,frame_count,names,cumulative_det,verbose=False,batchtag=[]):
    detection_results_batch = model(frameinput,verbose=verbose)
    bid = 0 # BACTH NAMING INDEX
    det = [] 

    try: 
        for detection_results in detection_results_batch:
            for it in detection_results.boxes:
                conf = round(it.conf.item(),3)
                cls = int(it.cls.item())
                name = names[cls]
                for bb in it.xyxy:
                    x1, y1, x2, y2 = bb
                    if len(batchtag) > 0: # Batch tag must match input
                        temp = [frame_count,cls,name,conf,int(x1), int(y1), int(x2), int(y2),batchtag[bid]]
                    else:
                        temp = [frame_count,cls,name,conf,int(x1), int(y1), int(x2), int(y2)]
                    cumulative_det.append(temp)
                    det.append(temp)
            bid = bid +1
    except:

        for x in detection_results_batch:
            for idx , conf in zip(x.probs.top5,x.probs.top5conf):
                label,score = names[idx],round(conf.item(),2)
                if score > 0.1:
                    # print(label,score)
                    if len(batchtag) > 0: # Batch tag must match input
                        temp = [frame_count,idx,label,score,0,0,0,0,batchtag[bid]]
                    else:
                        temp = [frame_count,idx,label,score,0,0,0,0]
                    cumulative_det.append(temp)
                    det.append(temp)
            bid = bid +1

    return cumulative_det, det


def get_color(obj_name):
    """
    Returns the color associated with the given object name.
    Parameters:
    obj_name (str): The name of the object.
    Returns:
    tuple: A tuple representing the RGB color.
    """
    # Define a dictionary to map object names to their corresponding colors
    color_mapping = {
        'left': (255, 255, 0),   # Yellow color for 'left'
        'right': (0, 255, 255),  # Cyan color for 'right'
        'back': (255, 0, 0),     # Red color for 'back'
        'front': (0, 255, 0),    # Green color for 'front'
        'BR': (0, 125, 125),     # Teal color for 'BR' (back right)
        'BL': (125, 125, 0),     # Olive color for 'BL' (back left)
        'FR': (0, 125, 255),     # Sky blue color for 'FR' (front right)
        'FL': (255, 125, 0),      # Orange color for 'FL' (front left)



        ## TRUCK DIRECTION
        'WL': (255, 255, 0),   # Yellow color for 'left'
        'L': (255, 255, 0),   # Yellow color for 'left'
        'WR': (0, 255, 255),  # Cyan color for 'right'
        'R': (0, 255, 255),  # Cyan color for 'right'
        'WB': (255, 0, 0),     # Red color for 'back'
        'B': (255, 0, 0),     # Red color for 'back'
        'WF': (0, 255, 0),    # Green color for 'front'
        'F': (0, 255, 0),    # Green color for 'front'
        'WBR': (0, 125, 125),     # Teal color for 'BR' (back right)
        'WBL': (125, 125, 0),     # Olive color for 'BL' (back left)
        'WFR': (0, 125, 255),     # Sky blue color for 'FR' (front right)
        'WFL': (255, 125, 0),      # Orange color for 'FL' (front left)

        'O': (255, 255, 255),      # Orange color for 'FL' (front left)
        'S': (255, 240, 255),      # Orange color for 'FL' (front left)
    }

    # Get the color from the dictionary, default to black if obj_name is not found
    return color_mapping.get(obj_name, (0, 0, 0))  # Default color is black

def draw_boxes_for_seq(image, yoloboxes):
    for box in yoloboxes:
        #"filename","cls_id","name","score","xmin","ymin","xmax","ymax"
        _, conf, x1, y1, x2, y2, obj_name, _, _ = box
        color = get_color(obj_name)
        thickness= 2

        if obj_name =="HORSE":
                        # Example usage
            color = (255,255,255)
            # print(x1, y1, x2, y2)
            scaled_bb = scale_bounding_box(int(x1), int(y1), int(x2), int(y2),scale_factor=0.9)
            # print("Scaled bounding box:", scaled_bb)
            x1, y1, x2, y2  = scaled_bb
            thickness = 4
        
        if obj_name =="TRUCK":
                        # Example usage
            color = (0,0,0)
            # print(x1, y1, x2, y2)
            scaled_bb = scale_bounding_box(int(x1), int(y1), int(x2), int(y2),scale_factor=1.2)
            # print("Scaled bounding box:", scaled_bb)
            x1, y1, x2, y2  = scaled_bb
            thickness= 2
         
        

        # Draw rectangle on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)  # Green color box with thickness 2
         # Draw arrow on top of the bounding box
        arrow_start = (int((int(x1) + int(x2)) / 2), int(y1)-25)
        # print(arrow_start)
        arrow_end = arrow_start

        arrow_length = 25
        if obj_name in ['left','L',"WL",]:
            arrow_end = (arrow_start[0] - arrow_length, arrow_start[1])
        elif obj_name in ['R',"WL",'right']:
            arrow_end = (arrow_start[0] + arrow_length, arrow_start[1])
        elif obj_name in ['B',"WL",'back']:
            arrow_end = (arrow_start[0], arrow_start[1] - arrow_length)
        elif obj_name in ['F',"WF",'front']:
            arrow_end = (arrow_start[0], arrow_start[1] + arrow_length)
        elif obj_name in ['FR',"WFR", 'FR']:
            arrow_end = (arrow_start[0] + int(0.7 * arrow_length), arrow_start[1] + int(0.7 * arrow_length))
        elif obj_name in ['BL',"WBL",'BL']:
            arrow_end = (arrow_start[0] - int(0.7 * arrow_length), arrow_start[1] - int(0.7 * arrow_length))
        elif obj_name in ['FL',"WFL",'FL']:
            arrow_end = (arrow_start[0] - int(0.7 * arrow_length), arrow_start[1] + int(0.7 * arrow_length))
        elif obj_name in ['BR',"WBR", 'BR']:
            arrow_end = (arrow_start[0] + int(0.7 * arrow_length), arrow_start[1] - int(0.7 * arrow_length))
        arrow_thickness = 4
        arrow_tip_size = 0.4  # Increas
        cv2.arrowedLine(image, arrow_start, arrow_end, color, arrow_thickness, tipLength=arrow_tip_size)
    
    return image


def crop_black_bars(image,cutpercent=0.24,targetsize=640,grid=4):
    height, width, _ = image.shape
    sq = int(targetsize/grid)
    bar_height = int(height * cutpercent)  # 10% of image height
    cropped_image = image[bar_height:height - bar_height, :]
    resized_image = Image.fromarray(cropped_image).resize((sq, sq))
    return cropped_image


# def createSeq(seq_frame=None,drawobjs=[],list_seq_frame=None,frame_counter=0,
#                     cropSQ=0.24, gridtotaltime = 16,
#                     fps=25,length=None,windows = [16,10,7,3],
#                     frame_interval=5,verbose=False):
#         seq_frame = draw_boxes_for_seq(seq_frame,drawobjs)
#         grid_collector = []
#         if int(frame_counter % frame_interval == 0):
#             crop_frame = crop_black_bars(seq_frame,cropSQ)
#             list_seq_frame.append(crop_frame)
#         # print("frame_counter",self.frame_counter)
#         if len(list_seq_frame) >  gridtotaltime*int(fps/frame_interval):
#             # print("##########################")
#             for window in windows:
#                 lengS = window/gridtotaltime
#                 Tgrid = []
#                 # print("Crafting Windows:________________________ ", window, lengS)
#                 if window == 16:
    
#                     for i in range(16):
#                         if i == 15:
#                             Tgrid.append(list_seq_frame[-1])
#                         else:
#                             dx = int(i*int(fps/frame_interval))
#                             Tgrid.append(list_seq_frame[dx+1])
#                             # print(dx)
#                 else:
#                     samplingGap = (fps/frame_interval)/(1/lengS)
#                     # print("GAP : ",samplingGap, window)
#                     for i in range(16):
#                         dx = int(15-i)*int(samplingGap)
#                         if  dx == 0:
#                             dx = 1
#                         Tgrid.append(list_seq_frame[-dx])
#                         # print(dx)

#                 grid = np.concatenate([np.concatenate(Tgrid[j:j+4], axis=1) for j in range(0, 16, 4)], axis=0)
#                 # Save the grid as a JPEG image
#                 grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
#                 grid = Image.fromarray(grid)
#                 grid = grid.resize((640,640))
#                 grid_collector.append(grid)
             
#         # if len(grid_collector) > 0:
#         #     print("************************")

#         return list_seq_frame,grid_collector


def make_time_canvas(size, frame_number, fps=25):
    """
    Create a canvas that shows the time in seconds corresponding to a given frame number.

    Parameters:
        size (tuple): Size of the canvas (height, width).
        frame_number (int): Frame number to display the corresponding time.
        fps (int): Frames per second (default is 25).
    Returns:
        numpy.ndarray: The created canvas as an image.
    """
    # Calculate time in seconds
    time_in_seconds = frame_number / fps

    # Create a blank white canvas
    canvas = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(size) / 300  # Adjust font size based on canvas size
    thickness = 4
    color = (0, 0, 0)  # Black text

    # Convert time to string
    time_text = f"{time_in_seconds:.1f} seconds"

    # Calculate text size and position
    text_size = cv2.getTextSize(time_text, font, font_scale, thickness)[0]
    text_x = (size[1] - text_size[0]) // 3
    text_y = (size[0] + text_size[1]) // 3
    cv2.putText(canvas, time_text, (text_x, text_y), font, font_scale, color, thickness)
    time_text = f"{frame_number} "
    

    # Calculate text size and position
    text_size = cv2.getTextSize(time_text, font, font_scale, thickness)[0]
    text_x = (size[1] - text_size[0]) // 2
    text_y = (size[0] + text_size[1]) // 2

    # Put the time text on the canvas
    cv2.putText(canvas, time_text, (text_x, text_y), font, font_scale, color, thickness)

    return canvas

def createSeqGeneral(seq_frame=None,drawobjs=[],list_seq_frame=None,frame_counter=0,
                    cropSQ=0.24,fps=25,windows = [16],
                    frame_interval=5,verbose=False, modelName='', CustomGaps = [[-1,-2,-3,-4]], targetsize=640, gridsize=4):
    # odelName=r'{}/{}'.format(self.model_path, model_name)
    if "FRV_ESDSEQ" in modelName:
        windows = [2]
        gridsize= 3
        num_images = int(gridsize*gridsize)
        # CustomGaps = [[-9 +1 * i for i in range(num_images)]]
        # DRONE SEQ IS INVERTED TOP LEFT TO TOP TO BOTTOM RIGHT FIRST IMAGE IS TOP LEFT 
        CustomGaps = [[-9, -8, -7, -6, -5, -4, -3, -2, -1]]
    if "FRA_SEQ" in modelName:
        gridsize=4
        windows = [32]
        CustomGaps = [[-148, -128, -108, -88, 
                    -68, -58, -48, -38,
                    -28, -23, -18, -13,
                    -7, -5, -3, -1]]
 
    if "FRV_SEQ" in modelName or "FRV_AUX_ACTIONSEQ" in modelName:
        gridsize=4
        num_images = int(gridsize*gridsize)
        CustomGaps = []
        windows = [16]
        for wd in windows:
            gap = int(fps * wd // num_images // frame_interval)  # Calculate gap in frame units
            CustomGaps.append([(- (num_images - i - 1) * gap) - 1 for i in range(num_images)])
            # print(CustomGaps)
    
    if "FRV_AUX_DRONESEQ" in modelName:
        windows = [2]
        gridsize= 3
        num_images = int(gridsize*gridsize)
        # CustomGaps = [[-9 +1 * i for i in range(num_images)]]
        # DRONE SEQ IS INVERTED TOP LEFT TO TOP TO BOTTOM RIGHT FIRST IMAGE IS TOP LEFT 
        CustomGaps = [[-9, -8, -7, -6, -5, -4, -3, -2, -1]]

    #Create a buffer for sequence
    if int(frame_counter % frame_interval == 0):
        if len(drawobjs) > 0: # Only Draw when objects are given
            seq_frame = cv2.UMat(seq_frame)
            seq_frame = draw_boxes_for_seq(seq_frame,drawobjs)
            seq_frame = seq_frame.get()
        if cropSQ > 0:
            seq_frame = crop_black_bars(seq_frame,cutpercent=cropSQ,targetsize=targetsize,grid=gridsize)
        list_seq_frame.append(seq_frame)

        
    grid_collector = []


    for gap in CustomGaps:
        Tgrid = []
        for c in gap:
            # print(abs(min(gap)))
            if len(list_seq_frame) > abs(c):
                Tgrid.append(list_seq_frame[c])
            else:
                Tgrid.append(np.zeros((seq_frame.shape[0], seq_frame.shape[1], 3), dtype=np.uint8))
        #Create SEQ GRID
        
        grid = np.concatenate([np.concatenate(Tgrid[j:j+gridsize], axis=1) for j in range(0, gridsize*gridsize, gridsize)], axis=0)
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        grid = Image.fromarray(grid)
        grid = grid.resize((int(targetsize),int(targetsize))) # Resize Image
        grid_collector.append(grid) 

    if CustomGaps and (len(list_seq_frame) > abs(min([min(x) for x in CustomGaps]))+1):
        list_seq_frame = list_seq_frame[1:] # Move Window 1+ frame and remove last

    # # Convert RGB to BGR for OpenCV
    # image_np = np.array(grid_collector[0]) 
    # opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # # Display the image using OpenCV
    # cv2.imshow("Image", opencv_image)
    # cv2.waitKey(50)

    return list_seq_frame,grid_collector


def TestSeuence(seconds=32,modelName="DRONE"):
    # TestSeuence(seconds=3,modelName="DRONE")
    # TestSeuence(seconds=16,modelName="FRV")
    # TestSeuence(seconds=32,modelName="FRA")
    list_seq_frame = []
    for i in range(25*seconds):
        # frame = make_canvas((480, 640), i)
        if i % 5 == 0:
            frame  =make_time_canvas((480, 640), i, fps=25)
            list_seq_frame, grids = createSeqGeneral(seq_frame=frame,drawobjs=[],list_seq_frame=list_seq_frame, modelName=modelName)
            for grid in grids:
                image_np = np.array(grid) 
            # Convert RGB to BGR for OpenCV
            opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            # Display the image using OpenCV
            cv2.imshow("Image", opencv_image)
            cv2.waitKey(50)

    cv2.destroyAllWindows()