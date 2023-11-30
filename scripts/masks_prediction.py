import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


def preprocess_raw_frame(frame):
    frame = cv2.resize(frame, (112, 112))
    frame = frame / 255.0
    frame = np.array(frame)

    return frame


def frame_to_tensor(frame, device="cuda"):
    frame = cv2.resize(frame, (112, 112))
    frame = np.array(frame)
    frame = torch.from_numpy(frame)
    frame = frame.to(device)
    frame = frame.permute(2, 1, 0)
    frame = frame.unsqueeze(0)
    frame = frame.float()

    return frame


def predict_mask(frame, model):
    with torch.inference_mode():
        pred_tensor = torch.sigmoid(model(frame))

    pred = pred_tensor.squeeze(0)
    pred = pred.squeeze(0)
    pred = pred.to("cpu").numpy()
    pred = pred >= 0.5

    return pred


def show_frame_and_predicted_mask(frame, mask):
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def overlay_mask_on_frame(frame, mask, color_tuple=(255, 0, 0), alpha=0.5):
    mask = mask[:, :, np.newaxis] * color_tuple
    frame = (frame * 255).astype(np.uint8)
    mask = mask.astype(np.uint8)
    return cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0)


def read_video_and_predict(video_path, model, amount_of_frames=None, predict=True):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the time interval between frames
    frame_interval_ms = int(1000 / fps)

    # Initialize a variable to keep track of the current frame number
    frames_list = []
    predictions_list = []
    current_frame = 0

    # Loop through the video frames
    while True:
        # Set the video's position to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Read the frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        if amount_of_frames != None and isinstance(amount_of_frames, int):
            if current_frame == amount_of_frames:
                break

        frame_to_predict = frame_to_tensor(frame)
        frames_list.append(frame)
        if predict == True:
            predicted_mask = predict_mask(frame_to_predict, model)
            predictions_list.append(predicted_mask)

        # Increment the current frame number based on the frame interval
        current_frame += 1

    # Release the video capture object
    cap.release()

    return frames_list, predictions_list


def generate_video_from_list_of_frames(frames, output_path, frame_rate=10.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        frame_rate,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    # Write each frame to the video file
    for frame in frames:
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()
