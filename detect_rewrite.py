import glob
from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper_original import *
# from PIL import Image
detector = Detector(classes = [0]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./weights/best.pt',) # pass the path to the trained weight file

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

video_paths = glob.glob("./IO_data/input/video/*")

# output = None will not save the output video
for video_path in video_paths:
    basename = os.path.basename(video_path)
    file_name = os.path.splitext(basename)[0]
    tracker.track_video(video_path, output="./IO_data/output/" + str(file_name) + ".avi", show_live = True, skip_frames = 0, count_objects = True, verbose=1)


# images_paths = glob.glob("./IO_data/input/images/*.jpg")
#
# # output = None will not save the output video
# for video_path in images_paths:
#     basename = os.path.basename(video_path)
#     file_name = os.path.splitext(basename)[0]
#     tracker.track_video(video_path, output="./IO_data/output/" + str(file_name) + ".jpg", show_live = True, skip_frames = 0, count_objects = True, verbose=1)

