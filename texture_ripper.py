#Notes for improvements
#Bezier curves for each edge (probably realistically impossible - It seems only openCV supports 4 point corrections, not... with bezier)
#rotate an input image
#rotate render output
#ability to animate/batch export a sequence from the placed coordinates
#ability to drag select all the coordinates at once

import sys
from tkinterdnd2 import DND_FILES, TkinterDnD
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import os
import json

theImagePath = None

class TextureRipperApp:
    def __init__(self, root, sys_arg_content=None):
        self.root = root
        self.root.title("Texture Ripper")
        self.sys_arg_content = sys_arg_content
        if self.sys_arg_content:
            print("Content from the file:")
            print(self.sys_arg_content)
        else:
            print("No content provided.")
        
        # Force-load tkdnd
        self.root.tk.call('package', 'require', 'tkdnd')

        self.label = tk.Label(self.root, text="Drop an image file here", width=40, height=10, bg="lightgray")
        self.label.pack(padx=10, pady=10)

        self.label.drop_target_register(DND_FILES)
        self.label.dnd_bind('<<Drop>>', self.drop)
        self.label.pack(expand=True, fill='both', padx=10, pady=10)
        self.label.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Initialize variables
        self.image = None
        self.image_path = None
        self.canvas_image = None
        self.selection_sets = []  # List of selection sets
        self.current_selection_set_index = None
        self.zoom_level = 1.0  # Default zoom level (1.0 = no zoom)
        self.canvas_offset_x = 0  # For tracking canvas movement during zoom/pan
        self.canvas_offset_y = 0
        self.pan_start_x = 0  # For tracking the starting point of the pan
        self.pan_start_y = 0
        self.is_panning = False  # Whether we are currently panning
        self.map_image = None  # Composite image (map) of all extracted textures
        self.zoom_active = False  # Whether zoom mode is active

        # Top button frames
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side=tk.TOP, pady=5)

        # Create the main frames (lol)
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Bottom button frames
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, pady=10)

        # Set initial canvas dimensions
        self.canvas_width = 800
        self.canvas_height = 600


        button_style = {
            'bg': 'black',
            'fg': 'white',
            'font': ('Tahoma', 10),
            'relief': tk.RAISED,
            'bd': 2,
            'activebackground': 'lightgray',
            'activeforeground': 'black'
        }


        self.load_button = tk.Button(self.top_frame, text="Load Image", command=self.load_image, **button_style)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(self.top_frame, text="Load Project", command=self.load_project, **button_style)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.extract_button = tk.Button(self.top_frame, text="Save Project", command=self.save_project, **button_style)
        self.extract_button.pack(side=tk.LEFT, padx=5)

        self.extract_button = tk.Button(self.top_frame, text="Render Texture", command=self.extract_texture, **button_style)
        self.extract_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(self.top_frame, text="Save File As", command=self.save_texture_map, **button_style)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Create canvas for displaying the main image
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg='gray')
        self.canvas.pack(side=tk.LEFT)

        # Create area for displaying the extracted texture map
        self.extracted_canvas = tk.Canvas(self.main_frame, width=400, height=400, bg="gray")
        self.extracted_canvas.pack(side=tk.LEFT, padx=10)



        # First row buttons
        self.first_row_frame = tk.Frame(self.button_frame)
        self.first_row_frame.pack(side=tk.TOP, pady=5)

        self.add_selection_set_button = tk.Button(self.first_row_frame, text="Add Selection Set", command=self.add_selection_set, **button_style)
        self.add_selection_set_button.pack(side=tk.LEFT, padx=5)

        self.add_selection_set_button = tk.Button(self.first_row_frame, text="Del Selection Set", command=self.del_selection_set, **button_style)
        self.add_selection_set_button.pack(side=tk.LEFT, padx=5)

        self.prev_selection_set_button = tk.Button(self.first_row_frame, text="Previous Set", command=self.prev_selection_set, **button_style)
        self.prev_selection_set_button.pack(side=tk.LEFT, padx=5)

        self.next_selection_set_button = tk.Button(self.first_row_frame, text="Next Set", command=self.next_selection_set, **button_style)
        self.next_selection_set_button.pack(side=tk.LEFT, padx=5)

        # Second row buttons
        self.second_row_frame = tk.Frame(self.button_frame)
        self.second_row_frame.pack(side=tk.TOP, pady=5)

        self.reset_button = tk.Button(self.second_row_frame, text="Reset View", command=self.reset_view, **button_style)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.clear_points_button = tk.Button(self.second_row_frame, text="Clear Points", command=self.clear_points, **button_style)
        self.clear_points_button.pack(side=tk.LEFT, padx=5)

        self.clear_map_button = tk.Button(self.second_row_frame, text="Clear Map", command=self.clear_map, **button_style)
        self.clear_map_button.pack(side=tk.LEFT, padx=5)

        # Bind mouse events for adding points, panning, zooming, and dragging
        self.canvas.bind("<Button-1>", self.add_or_select_point)
        self.canvas.bind("<B1-Motion>", self.drag_point)
        self.canvas.bind("<ButtonRelease-1>", self.release_point)
        self.canvas.bind("<MouseWheel>", self.zoom_image)
        self.canvas.bind("<Button-4>", self.zoom_image)  # For Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom_image)  # For Linux scroll down
        self.canvas.bind("<ButtonPress-2>", self.start_pan)  # Middle mouse button press
        self.canvas.bind("<B2-Motion>", self.pan_image)  # Middle mouse button drag
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)  # Middle mouse button release
        self.canvas.bind("<ButtonPress-3>", self.start_pan)  # Right mouse button press as alternative to middle button
        self.canvas.bind("<B3-Motion>", self.pan_image)  # Right mouse button drag
        self.canvas.bind("<ButtonRelease-3>", self.end_pan)  # Right mouse button release
        self.root.bind("<Control_L>", self.enable_zoom_mode)
        self.root.bind("<Control_R>", self.enable_zoom_mode)
        self.root.bind("<KeyRelease-Control_L>", self.disable_zoom_mode)
        self.root.bind("<KeyRelease-Control_R>", self.disable_zoom_mode)

        if sys_arg_content:
            self.root.after(40, self.init_gui) #20 ms delay forces tinkerer to spawn the image after GUI has loaded

    def init_gui(self):
        # After mainloop starts, call load_image if sys_arg_content exists
        if self.sys_arg_content:
            print(f"Loading image from: {self.sys_arg_content}")
            #time.sleep(3)
            self.load_image(self.sys_arg_content)

    def drop(self, event):
        filepath = event.data.strip("{}")  # handles filenames with spaces
        print(f"filepath: {filepath}")
        self.load_image(filepath)

    def load_image(self, path=None):
        global theImagePath
        #time.sleep(0.5)
        """Load an image file."""
        self.image_path = path or filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")])
        if not self.image_path:
            return
        try:
            #time.sleep(2)
            self.image = Image.open(self.image_path)
            theImagePath = self.image

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return

        # Reset variables
        self.zoom_level = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.selection_sets = []
        self.current_selection_set_index = None
        self.map_image = None
        self.extracted_canvas.delete("all")
        self.canvas.delete("all")

        # Get image dimensions
        self.img_width, self.img_height = self.image.size

        # Create low-res image (max 2048 width)
        max_width = 2048
        if self.img_width > max_width:
            scale_factor = max_width / self.img_width
            lowres_height = int(self.img_height * scale_factor)
            self.lowres_image = self.image.resize((max_width, lowres_height), Image.BILINEAR)
        else:
            self.lowres_image = self.image.copy()

        # Adjust canvas size based on image size while maintaining aspect ratio
        self.scale = min(self.canvas_width / self.img_width, self.canvas_height / self.img_height)
        self.canvas.config(width=self.canvas_width, height=self.canvas_height)


        #######################
        self.display_image()

        # Automatically add the first selection set
        self.add_selection_set(first_set=True)

        self.reset_view()
        print("got this far")


    def display_image(self):
        """Display image efficiently using viewport-based rendering and low-res fallback."""
        if not self.image:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        scale = self.scale * self.zoom_level

        # Use low-res image if zoomed out
        use_lowres = self.zoom_level < 2
        img = self.lowres_image if use_lowres else self.image
        img_width = img.width
        img_height = img.height
        effective_scale = scale * (self.img_width / img_width)  # Adjust for lowres scale if used

        # Compute visible region in image coords
        left = int(max(0, (0 - self.canvas_offset_x) / effective_scale))
        top = int(max(0, (0 - self.canvas_offset_y) / effective_scale))
        right = int(min(img_width, (canvas_width - self.canvas_offset_x) / effective_scale))
        bottom = int(min(img_height, (canvas_height - self.canvas_offset_y) / effective_scale))

        # Crop and resize
        cropped = img.crop((left, top, right, bottom))
        display_width = int((right - left) * effective_scale)
        display_height = int((bottom - top) * effective_scale)
        resized = cropped.resize((display_width, display_height), Image.BILINEAR)

        self.canvas_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            max(0, self.canvas_offset_x),
            max(0, self.canvas_offset_y),
            anchor=tk.NW,
            image=self.canvas_image
        )
        self.draw_grid()


    def image_to_canvas_coords(self, x, y):
        """Convert image coordinates to canvas coordinates."""
        canvas_x = x * self.scale * self.zoom_level + self.canvas_offset_x
        canvas_y = y * self.scale * self.zoom_level + self.canvas_offset_y
        return canvas_x, canvas_y

    def canvas_to_image_coords(self, x, y):
        """Convert canvas coordinates to image coordinates."""
        img_x = (x - self.canvas_offset_x) / (self.scale * self.zoom_level)
        img_y = (y - self.canvas_offset_y) / (self.scale * self.zoom_level)
        return img_x, img_y

    def draw_grid(self):
        """Draw the quadrilateral grid for the current selection set."""
        self.canvas.delete("grid")
        if self.current_selection_set_index is not None:
            selection_set = self.selection_sets[self.current_selection_set_index]
            points = selection_set['points']
            # Draw primary points
            for i, point in enumerate(points):
                canvas_x, canvas_y = self.image_to_canvas_coords(*point)
                self.canvas.create_oval(canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5,
                                        outline='red', width=2, tags="grid")

            if len(points) == 4:
                # Draw lines between points
                for i in range(4):
                    p1 = points[i]
                    p2 = points[(i + 1) % 4]

                    # Draw straight lines
                    canvas_x1, canvas_y1 = self.image_to_canvas_coords(*p1)
                    canvas_x2, canvas_y2 = self.image_to_canvas_coords(*p2)
                    self.canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                                            fill='green', width=2, tags="grid")

    def add_selection_set(self, first_set=False):
        """Add a new selection set."""
        selection_set = {'points': [], 'texture': None}
        self.selection_sets.append(selection_set)
        self.current_selection_set_index = len(self.selection_sets) - 1
        self.selected_point = None
        self.draw_grid()
        if not first_set:
            messagebox.showinfo("Info", f"Added new selection set {self.current_selection_set_index + 1}")

    def prev_selection_set(self):
        """Switch to the previous selection set."""
        if self.selection_sets and self.current_selection_set_index > 0:
            self.current_selection_set_index -= 1
            self.selected_point = None
            self.draw_grid()
        else:
            messagebox.showinfo("Info", "No previous selection set.")

    def next_selection_set(self):
        """Switch to the next selection set."""
        if self.selection_sets and self.current_selection_set_index < len(self.selection_sets) - 1:
            self.current_selection_set_index += 1
            self.selected_point = None
            self.draw_grid()
        else:
            messagebox.showinfo("Info", "No next selection set.")

    def del_selection_set(self):
        # Check if there is a current selection set to remove
        if len(self.selection_sets) > 0:
            # Remove the current selection set
            del self.selection_sets[self.current_selection_set_index]

            # Adjust the current selection set index
            if len(self.selection_sets) > 0:
                # If there are still selection sets, move the index to the last one
                self.current_selection_set_index = len(self.selection_sets) - 1
            else:
                # If no selection sets are left, reset the index
                self.current_selection_set_index = -1

            # Reset the selected point
            self.selected_point = None

            # Redraw the grid
            self.draw_grid()

            # Show a confirmation message
            messagebox.showinfo("Info", "Selection set removed.")
        else:
            messagebox.showwarning("Warning", "No selection sets to remove.")


    def add_or_select_point(self, event):
        """Add a point or select a point to drag."""
        if self.current_selection_set_index is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        selection_set = self.selection_sets[self.current_selection_set_index]
        points = selection_set['points']

        x, y = self.canvas_to_image_coords(event.x, event.y)

        # Constrain the point within the image boundaries
        x = max(0, min(x, self.img_width))
        y = max(0, min(y, self.img_height))

        # If there are less than 4 points, add new points
        if len(points) < 4:
            points.append((x, y))
            self.draw_grid()
        else:
            # Otherwise, check if a primary point is selected for dragging
            for i, point in enumerate(points):
                canvas_x, canvas_y = self.image_to_canvas_coords(*point)
                if abs(canvas_x - event.x) < 10 and abs(canvas_y - event.y) < 10:
                    self.selected_point = i
                    return

    def drag_point(self, event):
        """Drag the selected point."""
        if self.current_selection_set_index is None:
            return

        selection_set = self.selection_sets[self.current_selection_set_index]
        points = selection_set['points']

        if self.selected_point is not None:
            index = self.selected_point
            x, y = self.canvas_to_image_coords(event.x, event.y)

            # Constrain the point within the image boundaries
            x = max(0, min(x, self.img_width))
            y = max(0, min(y, self.img_height))

            points[index] = (x, y)
            self.draw_grid()

    def release_point(self, event):
        """Release the dragged point."""
        self.selected_point = None

    def clear_points(self):
        """Clear the selected points in the current selection set."""
        if self.current_selection_set_index is not None:
            self.selection_sets[self.current_selection_set_index]['points'] = []
            self.canvas.delete("grid")
            self.display_image()
        else:
            messagebox.showwarning("Warning", "No selection set to clear.")

    def clear_map(self):
        """Clear the extracted textures and reset the map."""
        for selection_set in self.selection_sets:
            selection_set['texture'] = None
        self.map_image = None
        self.extracted_canvas.delete("all")
        #messagebox.showinfo("Info", "Texture map cleared.")

    def zoom_image(self, event):
        """Zoom in or out based on the mouse wheel while Control is held."""
        if self.zoom_active:
            # For Windows and MacOS
            if hasattr(event, 'delta'):
                if event.delta > 0:
                    zoom_factor = 1.1
                else:
                    zoom_factor = 0.9
            else:
                # For Linux
                if event.num == 4:
                    zoom_factor = 1.1
                elif event.num == 5:
                    zoom_factor = 0.9
                else:
                    return

            new_zoom_level = self.zoom_level * zoom_factor

            # Limit zoom levels
            if new_zoom_level < 0.1 or new_zoom_level > 10:
                return

            # Get mouse position in image coordinates
            mouse_x, mouse_y = self.canvas_to_image_coords(event.x, event.y)

            # Update zoom level
            self.zoom_level = new_zoom_level

            # Adjust canvas offsets to keep the image centered at the cursor
            canvas_mouse_x, canvas_mouse_y = self.image_to_canvas_coords(mouse_x, mouse_y)
            self.canvas_offset_x += event.x - canvas_mouse_x
            self.canvas_offset_y += event.y - canvas_mouse_y

            self.display_image()

    def enable_zoom_mode(self, event):
        """Enable zoom mode when Control is pressed."""
        self.zoom_active = True

    def disable_zoom_mode(self, event):
        """Disable zoom mode when Control is released."""
        self.zoom_active = False

    def start_pan(self, event):
        """Start panning when the middle or right mouse button is pressed."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.is_panning = True

    def pan_image(self, event):
        """Pan the image by dragging with the middle or right mouse button."""
        if self.is_panning:
            delta_x = event.x - self.pan_start_x
            delta_y = event.y - self.pan_start_y
            self.canvas_offset_x += delta_x
            self.canvas_offset_y += delta_y
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.display_image()

    def end_pan(self, event):
        """End panning when the middle or right mouse button is released."""
        self.is_panning = False

    def reset_view(self):
        """Reset the zoom and panning to the default view."""
        self.zoom_level = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.display_image()

    def order_points(self, pts):
        """Order points in the following order: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and diff of points
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference

        return rect

    def extract_texture(self):
        """Extract the texture using the selected quadrilateral points."""
        if self.current_selection_set_index is None: 
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        if self.current_selection_set_index is 0: #0 fixes a bug in console and popup
            messagebox.showwarning("Warning", "Please add a selection set and 4 points.")
            return
        
        selection_set = self.selection_sets[self.current_selection_set_index]
        points = selection_set['points']

        if len(points) != 4:
            messagebox.showwarning("Warning", "Please select exactly 4 points.")
            return

        if not self.image_path:
            messagebox.showerror("Error", "No image loaded.")
            return

        src_img = cv2.imread(self.image_path)
        if src_img is None:
            messagebox.showerror("Error", "Failed to load the image for processing.")
            return

        # Convert the image to RGB
        src_img_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        # Use perspective transform for quadrilateral
        src_pts = np.array(points, dtype=np.float32)
        src_pts = self.order_points(src_pts)

        # Determine the size of the output image
        (tl, tr, br, bl) = src_pts

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = int(max(height_left, height_right))

        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp the image using the perspective transform
        warped = cv2.warpPerspective(src_img_rgb, M, (max_width, max_height))

        # Convert back to PIL Image and store in the selection set
        extracted_image = Image.fromarray(warped)
        selection_set['texture'] = extracted_image

        # Update the texture map
        self.update_texture_map()

        # Points remain for further editing

    def update_texture_map(self):
        """Update the composite texture map with all extracted textures."""
        textures = [s['texture'] for s in self.selection_sets if s['texture'] is not None]
        if not textures:
            return

        # Define the map dimensions
        # For simplicity, we'll arrange textures in a grid
        texture_widths = [img.width for img in textures]
        texture_heights = [img.height for img in textures]
        max_texture_width = max(texture_widths)
        max_texture_height = max(texture_heights)

        # Calculate grid size
        num_textures = len(textures)
        grid_cols = int(np.ceil(np.sqrt(num_textures)))
        grid_rows = int(np.ceil(num_textures / grid_cols))

        map_width = grid_cols * max_texture_width
        map_height = grid_rows * max_texture_height

        # Create a new blank image for the map
        self.map_image = Image.new('RGB', (map_width, map_height), color=(0, 0, 0))

        # Paste each texture into the map
        for idx, texture in enumerate(textures):
            row = idx // grid_cols
            col = idx % grid_cols
            x_offset = col * max_texture_width
            y_offset = row * max_texture_height

            # Center the texture in the grid cell
            x_center = x_offset + (max_texture_width - texture.width) // 2
            y_center = y_offset + (max_texture_height - texture.height) // 2

            self.map_image.paste(texture, (x_center, y_center))

        # Display the updated texture map
        self.display_texture_map()

    def display_texture_map(self):
        """Display the texture map on the extracted canvas."""
        if self.map_image:
            # Calculate aspect ratio and fit the image to the extracted canvas (400x400)
            canvas_width, canvas_height = 400, 400
            map_width, map_height = self.map_image.size
            ratio = min(canvas_width / map_width, canvas_height / map_height)
            new_width = int(map_width * ratio)
            new_height = int(map_height * ratio)

            # Resize while preserving aspect ratio
            img_resized = self.map_image.resize((new_width, new_height), Image.LANCZOS)
            extracted_image_tk = ImageTk.PhotoImage(img_resized)

            # Clear the canvas and display the resized image
            self.extracted_canvas.delete("all")
            self.extracted_canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2,
                                               anchor=tk.NW, image=extracted_image_tk)
            self.extracted_canvas.image = extracted_image_tk  # Keep reference

    def save_texture_map(self):
        """Save the texture map to a user-specified location."""
        if self.map_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg;*.jpeg"), ("TIF Files", "*.tif;*.tiff")])
            if file_path:
                try:
                    self.map_image.save(file_path)
                    messagebox.showinfo("Success", "Texture map saved successfully.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save texture map:\n{e}")
        else:
            messagebox.showwarning("Warning", "No texture map to save.")

    
    def save_project(self):
        """Save the current project, including image path and selection sets."""
        global theImagePath
        theImagePath = self.image_path  # Set the global image path
        
        # Ask user for a directory to save the .TexR file
        save_path = filedialog.asksaveasfilename(defaultextension=".TexR", filetypes=[("TexR Files", "*.TexR")])
        if not save_path:
            return
        
        project_data = {
            "image_path": self.image_path,
            "selection_sets": []
        }

        # Store each selection set's points
        for selection_set in self.selection_sets:
            project_data["selection_sets"].append({
                "points": selection_set['points']
            })

        # Save the data as plaintext in the .TexR file
        try:
            with open(save_path, 'w') as file:
                file.write(json.dumps(project_data, indent=4))
            #messagebox.showinfo("Info", "Project saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save project:\n{e}")

    def load_project(self, path=None):
        """Load a project from a .TexR file."""
        load_path = path or filedialog.askopenfilename(filetypes=[("TexR Files", "*.TexR")])
        if not load_path:
            return
        
        # Load the project data
        try:
            with open(load_path, 'r') as file:
                project_data = json.load(file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load project:\n{e}")
            return
        
        # Load image path
        self.image_path = project_data["image_path"]
        global theImagePath
        theImagePath = self.image_path  # Set the global image path
        
        self.load_image(self.image_path)

        # Load selection sets
        self.selection_sets = []
        for set_data in project_data["selection_sets"]:
            selection_set = {'points': set_data["points"], 'texture': None}
            self.selection_sets.append(selection_set)
        
        # Update the UI or canvas here (you can customize this part based on your needs)
        self.display_image()  # Assuming this is the function that updates the image view
        #messagebox.showinfo("Info", "Project loaded.")
    


if __name__ == "__main__":
    #root = tk.Tk()

    #system argument
    if len(sys.argv) < 2:
        print("No file provided, continuing without a file.")
        file_path = None  # or handle as needed, maybe use a default file path or skip file-related operations
    else:
        file_path = sys.argv[1]

    SysArgContent = None

    if file_path:
        # Check if the provided file is an image file (optional check)
        if os.path.isfile(file_path):
            print(f"Image file provided: {file_path}")
            SysArgContent = file_path
        else:
            print(f"Provided path is not a valid file: {file_path}")
    else:
        # Handle the case when no file is provided
        print("Continuing with other operations.")

    root = TkinterDnD.Tk()
    app = TextureRipperApp(root, SysArgContent)
    root.mainloop()

