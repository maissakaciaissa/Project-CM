import cv2
import numpy as np
from tkinter import Tk, filedialog

def MSE(block1, block2):
    """
    Calculates the mean squared error (MSE) between two blocks.
    """
    return np.sum((block1.astype("float") - block2.astype("float")) ** 2) / float(
        block1.shape[0] * block1.shape[1]
    )

def resize_video_if_needed(cap, max_width=640, max_height=360):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if frame_width > max_width or frame_height > max_height:
        new_width = max_width
        new_height = int((new_width / frame_width) * frame_height)
        
        if new_height > max_height:
            new_height = max_height
            new_width = int((new_height / frame_height) * frame_width)
        
        return new_width, new_height
    
    return frame_width, frame_height



def logarithmic_search(block, search_area, block_width, block_height, max_step):
    """
    Performs a logarithmic search to find the matching block within a sub-image of the reference (next) frame.
    'block' is from the current frame, 'search_area' is the portion of the next frame to search.
    Returns local coordinates (within search_area) of the best match, relative to (0,0) in search_area.
    """
    center_x = search_area.shape[1] // 2
    center_y = search_area.shape[0] // 2
    step = max_step
    best_coordinates = None
    min_mse = float('inf')

    while step >= 1:
        candidate_positions = [
            (center_x, center_y),
            (center_x + step, center_y),
            (center_x - step, center_y),
            (center_x, center_y + step),
            (center_x, center_y - step),
            (center_x + step, center_y + step),
            (center_x + step, center_y - step),
            (center_x - step, center_y + step),
            (center_x - step, center_y - step),
        ]

        for x, y in candidate_positions:
            if (0 <= x <= search_area.shape[1] - block_width) and (0 <= y <= search_area.shape[0] - block_height):
                candidate_block = search_area[y:y + block_height, x:x + block_width]
                if candidate_block.shape == block.shape:
                    mse = MSE(block, candidate_block)
                    if mse < min_mse:
                        min_mse = mse
                        best_coordinates = [(x, y), (x + block_width, y + block_height)]

        if best_coordinates:
            # Re-center search around the best match
            center_x = best_coordinates[0][0]
            center_y = best_coordinates[0][1]

        step //= 2

    return best_coordinates

def process_video(video_path, block_width, block_height, max_step, output_video_path, max_width=640, max_height=360):
    """
    Process the video with block-based motion estimation:
    - Finds best-matching blocks in the just-read frame (t+1) for each block of the previous frame (t).
    - Computes a residual image (difference).
    - Reconstructs frame (t+1) from matched blocks + residual.
    - Displays & saves the reconstructed video.
    """
    #Opening the Video File
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Resize video if necessary
    frame_width, frame_height = resize_video_if_needed(cap, max_width, max_height)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))
    if not out.isOpened():
        print("Error: VideoWriter not opened. Check your codecs or output path.")
        cap.release()
        return

    print(f"Output video will be saved to '{output_video_path}'")

    prev_frame = None

    def add_title_to_frame(frame, title):
        """
        Add a text bar (title) on top of the frame.
        """
        title_height = 40
        frame_with_title = np.copy(frame)
        cv2.rectangle(frame_with_title, (0, 0), (frame.shape[1], title_height), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_with_title, title, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame_with_title

    def pad_frame_to_size(frame, target_width, target_height):
        """
        Pad frame to target size with black borders if needed.
        """
        padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_frame[:frame.shape[0], :frame.shape[1]] = frame
        return padded_frame

    def process_frame_pair(frame1, frame2):
        """
        For consecutive frames (frame1 -> frame2):
          - Perform block matching.
          - Compute and display the "residual" (difference).
          - Construct a reconstructed frame2 from matched blocks + residual.
          - Show them all in a combined preview and save only the reconstructed frame.
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Prepare placeholders
        residual_image = np.zeros_like(gray1, dtype=np.float32)
        reconstructed_frame = np.copy(frame1)  # We'll reconstruct "frame2" based on frame1 blocks

        for y in range(0, gray1.shape[0], block_height):
            for x in range(0, gray1.shape[1], block_width):
                block_current = gray1[y : y + block_height, x : x + block_width]

                # Define local search region in frame2
                x1 = max(0, x - block_width)
                y1 = max(0, y - block_height)
                x2 = min(gray2.shape[1], x + block_width + block_width)
                y2 = min(gray2.shape[0], y + block_height + block_height)

                search_sub_image = gray2[y1:y2, x1:x2]

                # Logarithmic search
                best_local_coords = logarithmic_search(
                    block_current,
                    search_sub_image,
                    block_width,
                    block_height,
                    max_step
                )

                if best_local_coords is not None:
                    (lx1, ly1), (lx2, ly2) = best_local_coords
                    # Convert local coords back to global
                    gx1, gy1 = x1 + lx1, y1 + ly1
                    gx2, gy2 = x1 + lx2, y1 + ly2

                    # Extract matched block from frame2 in full color
                    matched_block = frame2[gy1:gy2, gx1:gx2]

                    # Resize if block dimension mismatch
                    if matched_block.shape[:2] != block_current.shape:
                        matched_block = cv2.resize(matched_block, (block_current.shape[1], block_current.shape[0]))

                    # Compute residual (gray)
                    matched_block_gray = cv2.cvtColor(matched_block, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    diff = block_current.astype(np.float32) - matched_block_gray
                    residual_image[y : y + block_height, x : x + block_width] = diff

                    # Reconstruct (matched block + residual)
                    # We'll apply the same residual to each color channel
                    for c in range(3):
                        channel_data = matched_block[:, :, c].astype(np.float32)
                        channel_data += diff  # add the difference in grayscale
                        channel_data = np.clip(channel_data, 0, 255).astype(np.uint8)
                        reconstructed_frame[y : y + block_height, x : x + block_width, c] = channel_data

        # Convert residual to display as black for no change
        # Taking absolute value so negative diffs don't get clipped to black
        residual_abs = np.abs(residual_image)
        # Clip to [0,255], convert to uint8
        residual_display = np.clip(residual_abs, 0, 255).astype(np.uint8)
        residual_bgr = cv2.cvtColor(residual_display, cv2.COLOR_GRAY2BGR)

        # --- CREATE TILED DISPLAY ---
        frame1_with_title = add_title_to_frame(frame1, "Frame t")
        frame2_with_title = add_title_to_frame(frame2, "Frame t+1")
        residual_with_title = add_title_to_frame(residual_bgr, "Residual (Absolute Difference)")
        recon_with_title = add_title_to_frame(reconstructed_frame, "Reconstructed (t+1)")

        # Determine max dims to pad all suitably for display (to ensure consistent padding)
        max_width = max(
            frame1_with_title.shape[1],
            frame2_with_title.shape[1],
            residual_with_title.shape[1],
            recon_with_title.shape[1]
        )
        max_height = max(
            frame1_with_title.shape[0],
            frame2_with_title.shape[0],
            residual_with_title.shape[0],
            recon_with_title.shape[0]
        )

        # Pad to uniform size
        frame1_with_title = pad_frame_to_size(frame1_with_title, max_width, max_height)
        frame2_with_title = pad_frame_to_size(frame2_with_title, max_width, max_height)
        residual_with_title = pad_frame_to_size(residual_with_title, max_width, max_height)
        recon_with_title = pad_frame_to_size(recon_with_title, max_width, max_height)

        # 2x2 layout
        top_row = cv2.hconcat([frame1_with_title, frame2_with_title])
        bottom_row = cv2.hconcat([residual_with_title, recon_with_title])
        combined_frame = cv2.vconcat([top_row, bottom_row])

        cv2.imshow("Block Matching Analysis (Press 'q' to Quit)", combined_frame)

        # Write reconstructed frame
        out.write(reconstructed_frame)
        print("Saving reconstructed frame to output video...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    # Main loop
    while True:
        #If no frame is read (i.e., the video has ended), it breaks the loop.
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Resize each frame to the target size
        frame_resized = cv2.resize(frame, (frame_width, frame_height))

        if prev_frame is not None:
            if not process_frame_pair(prev_frame, frame_resized):
                break

        prev_frame = frame_resized #the current frame becomes the previous frame for the next iteration

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done. Reconstructed video saved to:", output_video_path)


def choose_video_file():
    """Open a file dialog to choose a video file."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    return file_path

# Choose the video file
video_path = choose_video_file()

if video_path:
    output_video_path = "output_video_log.mp4"  # Save the output to this file (only reconstructed frames)
    block_width, block_height = 16, 16
    max_step =32

    process_video(video_path, block_width, block_height, max_step, output_video_path)
    print("Finished processing.")
else:
    print("No video file selected.")