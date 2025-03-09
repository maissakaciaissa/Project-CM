from tkinter import Tk, filedialog
import cv2
import numpy as np
import math

def MSE(block1, block2):
    """Calculate the Mean Squared Error between two blocks."""
    return np.sum((block1.astype("float") - block2.astype("float")) ** 2) / float(
        block1.shape[0] * block1.shape[1]
    )

def create_pyramid(image, levels):
    """Create a pyramid of images with decreasing resolutions."""
    # each time we take an image we down sized it by pyrDown and repeat it for as many levels as we want
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def hierarchical_search(bloc1, searchBox, pyramids_b, bloc_width, bloc_height, levels):
    """
    Hierarchical search for the matching block.
    The process starts at the lowest resolution and refines progressively.
    """
    best_coordinates = None
    mse = math.inf

    for level in range(levels - 1, -1, -1):  # Traverse levels (low to high)
        img_b = pyramids_b[level]
        scale_factor = 2 ** level

        # Adjust block dimensions and coordinates for the current level
        bloc1_scaled = cv2.resize(
            bloc1, (bloc_width // scale_factor, bloc_height // scale_factor)
        )
        scaled_searchBox = [
            [coord // scale_factor for coord in searchBox[0]],
            [coord // scale_factor for coord in searchBox[1]],
        ]

        # Search in a window around the previous position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                x = scaled_searchBox[0][0] + dx * (bloc_width // scale_factor)
                y = scaled_searchBox[0][1] + dy * (bloc_height // scale_factor)

                # Check if the block is within image bounds
                if (
                    0 <= x < img_b.shape[1] - bloc_width // scale_factor
                    and 0 <= y < img_b.shape[0] - bloc_height // scale_factor
                ):
                    bloc2 = img_b[
                        y : y + bloc_height // scale_factor,
                        x : x + bloc_width // scale_factor,
                    ]
                    temp_mse = MSE(bloc1_scaled, bloc2)

                    if temp_mse < mse:
                        mse = temp_mse
                        best_coordinates = [
                            (x * scale_factor, y * scale_factor),
                            (
                                (x + bloc_width // scale_factor) * scale_factor,
                                (y + bloc_height // scale_factor) * scale_factor,
                            ),
                        ]

    return best_coordinates

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

def process_video(video_path, bloc_width, bloc_height, levels, output_video_path, max_width=1024, max_height=576):
    """
    Process a video using hierarchical search and calculate residuals and reconstructions,
    and save only the reconstructed video to a file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get original video resolution
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original video resolution: {original_width}x{original_height}")

    # Resize video if necessary
    frame_width, frame_height = resize_video_if_needed(cap, max_width, max_height)
    print(f"Processing video with resized resolution: {frame_width}x{frame_height}")

    # Initialize the video writer for .mp4 output (saving only reconstructed frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the MP4V codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))  # Write output at resized dimensions

    prev_frame = None

    def add_title_to_frame(frame, title):
        """Add a title above the given frame similar to plt.title() in matplotlib."""
        title_height = 40  # Height of the title bar
        frame_with_title = np.copy(frame)
        
        cv2.rectangle(frame_with_title, (0, 0), (frame.shape[1], title_height), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_with_title, title, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame_with_title

    def pad_frame_to_size(frame, target_width, target_height):
        """Pad the frame to the target size with black borders."""
        padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_frame[:frame.shape[0], :frame.shape[1]] = frame
        return padded_frame

    def process_frame_pair(frame1, frame2):
        print("Processing frame pair...")
        a = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Create image pyramids
        print("Creating pyramids...")
        pyramid_a = create_pyramid(a, levels)
        pyramid_b = create_pyramid(b, levels)

        residual_image = np.zeros_like(a, dtype=np.int16)
        reconstructed_image = np.copy(frame1)  # Preserve color reconstruction

        for y in range(0, a.shape[0], bloc_height):
            for x in range(0, a.shape[1], bloc_width):
                print(f"Processing block at ({x}, {y})...")
                bloc1 = a[y : y + bloc_height, x : x + bloc_width]

                # Define an initial search box
                searchBox = [
                    (max(0, x - bloc_width), max(0, y - bloc_height)),
                    (min(a.shape[1], x + 2 * bloc_width), min(a.shape[0], y + 2 * bloc_height)),
                ]

                # Find the best matching block
                print(f"Performing hierarchical search for block at ({x}, {y})...")
                best_match = hierarchical_search(
                    bloc1, searchBox, pyramid_b, bloc_width, bloc_height, levels
                )

                if best_match is not None:
                    print(f"Found best match for block at ({x}, {y}).")
                    matched_block = frame2[best_match[0][1] : best_match[1][1], best_match[0][0] : best_match[1][0]]

                    if matched_block.shape[:2] != bloc1.shape:
                        matched_block = cv2.resize(matched_block, (bloc1.shape[1], bloc1.shape[0]))

                    residual_image[y : y + bloc_height, x : x + bloc_width] = (
                        bloc1.astype(np.int16) - cv2.cvtColor(matched_block, cv2.COLOR_BGR2GRAY).astype(np.int16)
                    )

                    residual_abs = np.abs(residual_image)
                    residual_display = np.clip(residual_abs, 0, 255).astype(np.uint8)
                    residual_bgr = cv2.cvtColor(residual_display, cv2.COLOR_GRAY2BGR)

                    reconstructed_image[y : y + bloc_height, x : x + bloc_width] = np.clip(
                        matched_block + residual_image[y : y + bloc_height, x : x + bloc_width][:, :, np.newaxis], 0, 255
                    ).astype(np.uint8)
                else:
                    print(f"No match found for block at ({x}, {y}).")

        print("Completed processing current frame pair.")

        frame1_with_title = add_title_to_frame(np.copy(frame1), "Original Frame")
        frame2_with_title = add_title_to_frame(np.copy(frame2), "Next Frame")
        residual_bgr_with_title = add_title_to_frame(residual_bgr, "Residual Image")
        reconstructed_image_with_title = add_title_to_frame(np.copy(reconstructed_image), "Reconstructed Frame")

        max_width = max(frame1_with_title.shape[1], frame2_with_title.shape[1], residual_bgr_with_title.shape[1], reconstructed_image_with_title.shape[1])
        max_height = max(frame1_with_title.shape[0], frame2_with_title.shape[0], residual_bgr_with_title.shape[0], reconstructed_image_with_title.shape[0])

        frame1_with_title = pad_frame_to_size(frame1_with_title, max_width, max_height)
        frame2_with_title = pad_frame_to_size(frame2_with_title, max_width, max_height)
        residual_bgr_with_title = pad_frame_to_size(residual_bgr_with_title, max_width, max_height)
        reconstructed_image_with_title = pad_frame_to_size(reconstructed_image_with_title, max_width, max_height)

        combined_frame = cv2.vconcat([
            cv2.hconcat([frame1_with_title, frame2_with_title]),
            cv2.hconcat([residual_bgr_with_title, reconstructed_image_with_title]),
        ])

        # Resize the combined frame to fit the display window
        combined_frame_resized = cv2.resize(combined_frame, (800, 600))

        cv2.imshow("Frame Analysis (Original, Residual, Reconstructed)", combined_frame_resized)

        out.write(reconstructed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        if prev_frame is not None:
            if not process_frame_pair(prev_frame, frame):
                print("Exiting video processing loop.")
                break

        prev_frame = frame

    print("Releasing resources...")
    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()
    print("Processing complete.")


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
    output_video_path = "output_video_hier.mp4"  # Save the output to this file (only reconstructed frames)
    bloc_width, bloc_height = 16, 16
    levels = 3
    process_video(video_path, bloc_width, bloc_height, levels, output_video_path)
else:
    print("No video file selected.")
