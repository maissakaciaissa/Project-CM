import cv2
import numpy as np
import math
from tkinter import Tk, Label, Button, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk

def MSE(block1, block2):
    """Calculate the Mean Squared Error between two blocks."""
    return np.sum((block1.astype("float") - block2.astype("float")) ** 2) / float(
        block1.shape[0] * block1.shape[1]
    )


def logarithmic_search(block, search_area, block_width, block_height, max_step):
    """
    Performs a logarithmic search to find the matching block within a sub-image of the reference (next) frame.
    'block' is from the current frame, 'search_area' is the portion of the next frame to search.
    Returns local coordinates (within search_area) of the best match, relative to (0,0) in search_area.
    """
    """
    block: A block (subset of pixels) from the current frame (the template to match).
    search_area: The area in the next frame where the function will search for the best match.
    block_width and block_height: Dimensions of the block.
    max_step: The largest step size to begin the search; this controls the initial search resolution.
    """
    center_x = search_area.shape[1] // 2
    center_y = search_area.shape[0] // 2
    step = max_step
    best_coordinates = None
    min_mse = float('inf') # Initialize MSE to infinity

    while step >= 1:
        #neighbors 
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
            # the if checks if the block is within the search area
            if (0 <= x <= search_area.shape[1] - block_width) and (0 <= y <= search_area.shape[0] - block_height):
                candidate_block = search_area[y:y + block_height, x:x + block_width]
                if candidate_block.shape == block.shape:
                    mse = MSE(block, candidate_block)
                    if mse < min_mse:
                        #if the new calculated mse is less than the previous one, update the min_mse and best_coordinates
                        min_mse = mse
                        best_coordinates = [(x, y), (x + block_width, y + block_height)]

        if best_coordinates:
            # Re-center search around the best match
            center_x = best_coordinates[0][0]
            center_y = best_coordinates[0][1]

        step //= 2

    return best_coordinates


def manual_block_selection(image):
    """Allow the user to manually select a block using the mouse."""
    roi = None #selected block
    drawing = False
    x_start, y_start = -1, -1

    def select_block(event, x, y, flags, param):
        nonlocal roi, drawing, x_start, y_start 
        if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing cv2.EVENT_LBUTTONDOWN mouse button pressed
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
                   abs(x_end - x_start), abs(y_end - y_start)) # calcule the selected block from the coordinaes , calc width,height
            print(f"Selected block: {roi}")
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2) # Draw the final rectangle on the image
            cv2.imshow("Select Block", image)

    cv2.imshow("Select Block", image)
    cv2.setMouseCallback("Select Block", select_block)

    print("Draw a rectangle around the desired block, then press 'q' to confirm.")
    #Wait for User Confirmation
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") and roi is not None:
            print("here")
            break

    cv2.destroyAllWindows()
    return roi

def process_selected_block(img1, img2, roi, img_label_residual, img_label_reconstructed):
    """Process a manually selected block using logarithmic search."""
    if roi is None:
        print("Error: No block selected.")
        return

    # Convert images to grayscale
    a = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Extract the selected block from img1 (grayscale)
    x, y, w, h = roi
    # Extracts a subimage (block) from the grayscale image 'a'
    bloc1 = a[y:y + h, x:x + w]

    # Define the search box coordinates
    searchBox = [
        max(0, x - w), max(0, y - h),  # Top-left corner
        min(b.shape[1], x + 2 * w), min(b.shape[0], y + 2 * h)  # Bottom-right corner
    ]

    # Extract the search area from img2 (grayscale)
    search_area = b[searchBox[1]:searchBox[3], searchBox[0]:searchBox[2]]

    # Perform logarithmic search to find the best matching block
    best_match = logarithmic_search(bloc1, search_area, w, h, max_step=16)  # Adjust max_step as needed

    if best_match is not None:
        # Extract the matched block from the **color image (img2)**
        matched_block_color = img2[
            best_match[0][1] + searchBox[1]:best_match[1][1] + searchBox[1],
            best_match[0][0] + searchBox[0]:best_match[1][0] + searchBox[0]
        ]

        # Ensure the matched block is resized to match the selected block size
        if matched_block_color.shape[:2] != (h, w):
            matched_block_color = cv2.resize(matched_block_color, (w, h))

        # Compute the residuals per channel (between the selected block and the matched block)
        residual_image = img1[y:y + h, x:x + w].astype(np.int16) - matched_block_color.astype(np.int16) #converted to int16 to avoid overflow(negative values or large ones)

        # Reconstruct the block by adding residuals back to the matched block
        reconstructed_image_color = np.clip(matched_block_color + residual_image, 0, 255).astype(np.uint8)

        # Display the residual image (convert to grayscale for visualization)
        residual_display = np.clip(np.abs(residual_image), 0, 255).astype(np.uint8) # convert the residual image to uint8 for display in the range 0-255 , abs for the black color
        residual_display_gray = cv2.cvtColor(residual_display, cv2.COLOR_BGR2GRAY)#converts the image to grayscale to simplify the visualization of the residuals.

        # Update residual image in the GUI
        photo_residual = ImageTk.PhotoImage(Image.fromarray(residual_display_gray)) # from np to pil
        img_label_residual.config(image=photo_residual)
        img_label_residual.image = photo_residual

        # Update reconstructed color image in the GUI
        photo_reconstructed = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(reconstructed_image_color, cv2.COLOR_BGR2RGB)))
        img_label_reconstructed.config(image=photo_reconstructed)
        img_label_reconstructed.image = photo_reconstructed

        print("Processing completed successfully.")
    else:
        print("No matching block found.")



# GUI Setup (same as before)
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



# Global variables for images and ROI
img1 = None
img2 = None
roi = None
residual_image = None
reconstructed_image_color = None
# GUI Setup
root = Tk()
root.geometry("1100x600")  # Set a specific size for the window
root.title("Logarithmic Block Matching")

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

btn_process = Button(root, text="Process", command=lambda: process_selected_block(img1, img2, roi,img_label_residual,img_label_reconstructed))
btn_process.place(x=500, y=300)  # 50 pixels below the third button

root.mainloop() 