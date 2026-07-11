# -*- coding: utf-8 -*-
import os
import json
import torch
from torch.nn.parallel import DataParallel
from chat import MiniCPMVChat, img2base64


torch.manual_seed(0)

# Load MiniCPM-V. Change this path to the local MiniCPM-Llama3-V-2.5 checkpoint.
chat_model = MiniCPMVChat('../MiniCPM-V-main/MiniCPM-Llama3-V-2_5/')


def process_video(video_path, output_dir):
    """
    Process a sampled-frame folder, generate triples for each frame, and save them.

    :param video_path: Path to a video frame folder, e.g., ssv2_frames/99532.
    :param output_dir: Directory for saving raw triple descriptions, e.g., ssv2_des.
    """
    video_name = os.path.basename(video_path)
    description_file = os.path.join(output_dir, f"{video_name}.txt")

    if os.path.exists(description_file):
        print(f"Description file already exists, skipping video: {video_name}")
        return

    descriptions = []

    for frame_number in sorted(os.listdir(video_path), key=lambda x: int(os.path.splitext(x)[0])):
        if frame_number.endswith('.jpg'):
            frame_path = os.path.join(video_path, frame_number)
            im_64 = img2base64(frame_path)
            msgs = [{
                "role": "user",
                "content": (
                    "The visual relationship between entities depicted in an image and entities is represented by entities: ...  "
                    "relation: ...  Entity: ...  The form of the triples is presented and printed, for example, an image of a woman sitting on a stool "
                    "is printed strictly in the following format: Entity: Woman relation: sitting Entity: stool. "
                    "Entity: Woman relation: sitting Entity: stool are on the same row. "
                    "Don't print other words or sentences. Be sure to print only triples"
                )
            }]
            inputs = {"image": im_64, "question": json.dumps(msgs)}

            with torch.no_grad():
                if isinstance(chat_model, DataParallel):
                    answer = chat_model.module.chat(inputs)
                else:
                    answer = chat_model.chat(inputs)

            if isinstance(answer, str):
                descriptions.append(answer)
            else:
                descriptions.append(str(answer))

    with open(description_file, 'w', encoding='utf-8') as f:
        for desc in descriptions:
            f.write(desc + '\n')
    print(f"Saved description file: {description_file}")


def main():
    input_root = '../ssv2_frames/'
    output_root = '../ssv2_des/'

    os.makedirs(output_root, exist_ok=True)

    video_count = 0

    for video_folder in os.listdir(input_root):
        video_path = os.path.join(input_root, video_folder)
        if os.path.isdir(video_path):
            process_video(video_path, output_root)
            video_count += 1
            print(f"Completed videos count: {video_count}")


if __name__ == "__main__":
    main()
