# Texture-Ripper
An attempt to recreate [ShoeBox](https://renderhjs.net/shoebox/) with the power of ChatGPT.

# How to Use:

- **Saving/Loading**
  - **Load an image:**
    - Click the **"Load Image"** button and select the image file.
    - Drag an image on the GUI
    - Drag an image onto the Python file itself
  - **Load project files**
    - Save/Load Project buttons (TexR format, plaintext. Stands for TextureRipper)
- **Select Points:**
    - Click anywhere on the image to place a point.
    - Strategically place four points to perspective correct a desired texture extraction.
    - Adjust points by clicking and dragging them.
- **Render Texture:**
    - Click **"Render Texture"** to extract the selected area.
    - The points remain on the image, allowing further adjustments if needed.
    - If you modify the points and click **"Render Texture"** again, the texture in the map is updated.
- **Add More Selection Sets:**
    - Click **"Add Selection Set"** to define additional textures.
    - Use **"Previous Set"** and **"Next Set"** to navigate between selection sets.
    - Delete selection sets as well.
- **Save the Texture Map:**
    - Once all desired textures are rendered, click **"Save File As"** to save the map.
- **Clear:**
    - Use **"Clear Points"** to reset points in the current selection set.
    - Use **"Clear Map"** to clear all extracted textures and start over.
    - Use **Del Selection Set** to delete current selection set.
    - Use **Reset View** to reset the current zoom.
- **Zoom and Pan:**
    - **Hold Control and use the mouse wheel to zoom.**
    - Use the middle or right mouse button to pan.


# Installation:

**Python:**
Download the python file.
Install the dependenices via cmd or however you like:
The commands I would run:
    `pip install tkinterdnd2`
    `pip install tkinter`
    `pip install cv2`
    `pip install numpy`
    `pip install pillow`

# Limitations:
- Animations do not render properly with multiple sets (unlikely to ever change - Only animate one set at a time, only have 1 set at a time for animations, ideally).
- Does not support using bezier curves for texture extraction like ShoeBox (unlikely to be implemented - requires total restructuring of code).
- Texture packing method is simple, unoptimized.
