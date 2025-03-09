import cv2
import numpy as np
import math
import os
from tkinter import Tk, Label, Button, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk

def MSE(block1, block2):
    """Calculate the Mean Squared Error between two blocks."""
    return np.sum((block1.astype("float") - block2.astype("float")) ** 2) / float(
        block1.shape[0] * block1.shape[1]
    )


def create_pyramid(image, levels):
    """Create a pyramid of images with decreasing resolutions."""
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def hierarchical_search(bloc1, searchBox, pyramids_b, bloc_width, bloc_height, levels):
    """Hierarchical search for the matching block."""
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

def manual_block_selection(image):
    """Allow the user to manually select a block using the mouse."""
    roi = None
    drawing = False
    x_start, y_start = -1, -1

    def select_block(event, x, y, flags, param):
        nonlocal roi, drawing, x_start, y_start
        if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
            drawing = True
            x_start, y_start = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Update the rectangle
            temp_image = image.copy()
            cv2.rectangle(temp_image, (x_start, y_start), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Block", temp_image)

        elif event == cv2.EVENT_LBUTTONUP:  # Finalize the rectangle
            drawing = False
            x_end, y_end = x, y
            roi = (min(x_start, x_end), min(y_start, y_end),
                   abs(x_end - x_start), abs(y_end - y_start))
            print(f"Selected block: {roi}")
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("Select Block", image)

    cv2.imshow("Select Block", image)
    cv2.setMouseCallback("Select Block", select_block)

    print("Draw a rectangle around the desired block, then press 'q' to confirm.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        print("true")
        if key == ord("q") and roi is not None:
            break

    cv2.destroyAllWindows()
    return roi



def process_selected_block(img1, img2, roi, levels, img_label_residual, img_label_reconstructed):
    """Process a manually selected block using hierarchical search."""
    if roi is None:
        print("Error: No block selected.")
        return

    # Convert images to grayscale
    a = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create image pyramids
    pyramid_a = create_pyramid(a, levels)
    pyramid_b = create_pyramid(b, levels)

    # Extract the selected block from img1 (grayscale)
    x, y, w, h = roi
    bloc1 = a[y:y+h, x:x+w]

    # Define the search box
    searchBox = [
        (max(0, x - w), max(0, y - h)),
        (min(a.shape[1], x + 2 * w), min(a.shape[0], y + 2 * h)),
    ]

    # Find the best matching block in img2 (grayscale)
    best_match = hierarchical_search(bloc1, searchBox, pyramid_b, w, h, levels)

    # Initialize the residual image and reconstructed image (both grayscale)
    residual_image = np.zeros_like(bloc1, dtype=np.int16)
    reconstructed_image = np.zeros_like(bloc1, dtype=np.uint8)

    if best_match is not None:
        matched_block = b[  # Get the matched block in grayscale
            best_match[0][1]:best_match[1][1],
            best_match[0][0]:best_match[1][0]
        ]

        # Ensure matched block is the same size as the selected block
        if matched_block.shape != bloc1.shape:
            matched_block = cv2.resize(matched_block, (bloc1.shape[1], bloc1.shape[0]))

        # Calculate the residual (grayscale difference)
        residual_image = bloc1.astype(np.int16) - matched_block.astype(np.int16)
        reconstructed_block = matched_block.astype(np.int16) + residual_image.astype(np.int16)

        # Take the absolute value of the residual to avoid negative differences
        residual_abs = np.abs(residual_image)

        # Clip the residual to the 0-255 range and convert to uint8
        residual_display = np.clip(residual_abs, 0, 255).astype(np.uint8)

        # Reconstruct the block by adding residual to the matched block
        #reconstructed_image = np.clip(reconstructed_block, 0, 255).astype(np.uint8)

        # Now map the reconstructed grayscale block to the original color block from img2
        reconstructed_image_color = img2[  # Use the color version of the matched block
            best_match[0][1]:best_match[1][1],
            best_match[0][0]:best_match[1][0]
        ]

        # Ensure the reconstructed color block matches the selected block's position and size
        if reconstructed_image_color.shape != (h, w, 3):
            reconstructed_image_color = cv2.resize(reconstructed_image_color, (w, h))

    # Convert residual and reconstructed images for Tkinter display
    residual_display_rgb = cv2.cvtColor(residual_display, cv2.COLOR_BGR2RGB)
    reconstructed_image_color_rgb = cv2.cvtColor(reconstructed_image_color, cv2.COLOR_BGR2RGB)

    # Display the results in Tkinter
    photo_residual = ImageTk.PhotoImage(Image.fromarray(residual_display_rgb))
    img_label_residual.config(image=photo_residual)
    img_label_residual.image = photo_residual
    img_label_residual.place(x=0, y=300)

    photo_reconstructed = ImageTk.PhotoImage(Image.fromarray(reconstructed_image_color_rgb))
    img_label_reconstructed.config(image=photo_reconstructed)
    img_label_reconstructed.image = photo_reconstructed
    img_label_reconstructed.place(x=620, y=300)





    # cv2.imshow("Residual Image (Grayscale)", residual_display)  # Display the absolute residual in grayscale
    # cv2.imshow("Reconstructed Block (Color)", reconstructed_image_color)  # Display the reconstructed block in color
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Global variables for images and ROI
img1 = None
img2 = None
roi = None
residual_image = None
reconstructed_image_color = None

def select_file(img_label, is_img1):
    """Choose an image file and display it in the GUI."""
    global img1, img2
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = cv2.imread(file_path)
        # resized = cv2.resize(img, (300, 300))  # Resize to fit GUI
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        
        img_label.config(image=photo)
        img_label.image = photo

        if is_img1:
            img1 = img
            img_label.place(x=0, y=0)
            return img1
        else:
            img2 = img
            img_label.place(x=620, y=0)
            return img2
        
def choose_block_action():
    global roi, img1
    if img1 is None:
        print("Error: Frame 1 not selected.")
        return
    roi = manual_block_selection(img1)
def choose_frame(is_img1, img_label):
    global img1, img2
    if is_img1:
        img1 = select_file(img_label, True)
    else:
        img2 = select_file(img_label, False)



# GUI Setup
root = Tk()
root.geometry("1100x600")  # Set a specific size for the window
root.title("Hierarchical Block Matching")
levels = 3
# Labels for images
x, y = 0, 0
width, height = 300, 300
img_label1 = Label(root, text="Frame 1", font=('Arial', 16))
img_label1.place(x=x + width / 2 - 50, y=y + height / 2 - 20)  # Adjust position to center the label

x2, y2 = 620,0  # For Frame 2
img_label2 = Label(root, text="Frame 2", font=('Arial', 16))
img_label2.place(x=x2 + width / 2 - 50, y=y2 + height / 2 - 20)  # Centered for Frame 2

x2, y2 = 0,300
img_label_residual =Label(root, text="Residual", font=('Arial', 16)) 
img_label_residual.place(x=x2 + width / 2 - 50, y=y2 + height / 2 - 20)

x2, y2 = 620,300
img_label_reconstructed =Label(root, text="Reconstructed", font=('Arial', 16)) 
img_label_reconstructed.place(x=x2 + width / 2 - 50, y=y2 + height / 2 - 20)


# Buttons

# Using place() to position buttons one below the other
btn_select_frame1 = Button(root, text="Choose Frame 1", command=lambda: choose_frame(True, img_label1))
btn_select_frame1.place(x=500, y=150)  # Positioning at a specific x and y

btn_select_frame2 = Button(root, text="Choose Frame 2", command=lambda: choose_frame(False, img_label2))
btn_select_frame2.place(x=500, y=200)  # 50 pixels below the first button

btn_choose_block = Button(root, text="Choose Block", command=choose_block_action)
btn_choose_block.place(x=500, y=250)  # 50 pixels below the second button

btn_process = Button(root, text="Process", command=lambda: process_selected_block(img1, img2, roi, levels,img_label_residual,img_label_reconstructed))
btn_process.place(x=500, y=300)  # 50 pixels below the third button

root.mainloop()