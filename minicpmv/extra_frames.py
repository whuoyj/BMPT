import os
import cv2


def extract_frames(video_path, output_folder, num_frames=4):
    """
    Uniformly sample a fixed number of frames from a video and save them.

    :param video_path: Path to the input video file.
    :param output_folder: Directory where sampled frames will be saved.
    :param num_frames: Number of frames to sample. Default: 4.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"No frames found in video file: {video_path}")
        cap.release()
        return

    frame_indices = [int(total_frames * (i + 1) / (num_frames + 1)) for i in range(num_frames)]

    os.makedirs(output_folder, exist_ok=True)

    current_frame = 0
    saved_frames = 0

    while saved_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == frame_indices[saved_frames]:
            frame_filename = os.path.join(output_folder, f"{saved_frames + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")
            saved_frames += 1

        current_frame += 1

    cap.release()


def process_videos(input_dir, output_dir, num_frames=4):
    """
    Process all video files in a directory and save uniformly sampled frames.

    :param input_dir: Directory containing input video files.
    :param output_dir: Root directory for saving sampled frame folders.
    :param num_frames: Number of frames to sample from each video. Default: 4.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".webm"):
            video_path = os.path.join(input_dir, filename)
            video_name = os.path.splitext(filename)[0]
            video_output_folder = os.path.join(output_dir, video_name)
            print(f"Processing video: {video_path}")
            extract_frames(video_path, video_output_folder, num_frames)


if __name__ == "__main__":
    input_directory = "../Yourdir/somethingV2/20bn-something-something-v2/"
    output_directory = "../Yourdir/ssv2_frames/"

    if not os.path.exists(input_directory):
        print(f"Input directory does not exist: {input_directory}")
    else:
        os.makedirs(output_directory, exist_ok=True)
        process_videos(input_directory, output_directory, num_frames=4)
