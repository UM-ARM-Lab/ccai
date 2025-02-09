import cv2
import os
from natsort import natsorted
import pathlib
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')

def display_and_save_video(image_dir, output_video_path=None, frame_rate=30):

    images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]
    images = natsorted(images)

    if not images:
        print("No PNG images found in the directory.")
        return
    
    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", width, height)
    delay = int(1000 / frame_rate) 

    for image_path in images:
        frame = cv2.imread(image_path)
        cv2.imshow("Video", frame)
        # Write the frame to the video file if saving
        if video_writer:
            video_writer.write(frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release resources
    if video_writer:
        video_writer.release()
        print(f"Video saved to {output_video_path}")
    cv2.destroyAllWindows()
 

# output_path = '/home/newuser/Desktop/Honda/ccai/data/plots/vid_results_vf_'
# for i in range(8):
#     for j in range(1):
#         j=0
#         image_directory = f"/home/newuser/Desktop/Honda/ccai/data/experiments/imgs/trial_results{i}_{j}_vf"
#         op = output_path + str(i) + '.mp4'
#         display_and_save_video(image_directory, op, frame_rate=50)
 

def convert_dirs_to_videos(source_dir, output_dir, frame_rate):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subdirs = [os.path.join(source_dir, d) for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    subdirs = natsorted(subdirs)  # Ensure natural sorting
    for idx, subdir in enumerate(subdirs):
        print(subdir)
        output_video_path = os.path.join(output_dir, f"vid_{idx}.mp4")
        display_and_save_video(subdir, output_video_path, frame_rate=frame_rate)


source_dir = f'{fpath.resolve()}/experiments/imgs'
output_dir = f'{fpath.resolve()}/plots/test_method_vids' 
# output_dir = f'{fpath.resolve()}/plots/5_3_test_80iter'
convert_dirs_to_videos(source_dir, output_dir, frame_rate=100)
