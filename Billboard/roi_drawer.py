import cv2

# Path to your video file
VIDEO_PATH = r"C:\Users\MSI\Desktop\Billboard\WhatsApp Video 2026-05-01 at 23.15.31.mp4"

# Global variables for drawing
drawing = False
start_point = None
end_point = None

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

def main():
    global start_point, end_point

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_PATH}'")
        return

    # Set up mouse callback
    cv2.namedWindow("Video with ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Video with ROI", mouse_callback)

    print("Playing video. Click and drag to draw ROI rectangle.")
    print("Press 'p' to print ROI coordinates as fractions.")
    print("Press 'r' to reset rectangle.")
    print("Press 'q' to quit.")

    frame_resized = False
    display_scale = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        frame_h, frame_w = frame.shape[:2]

        # Scale frame to fit screen — same logic as test.py
        display_scale = min(1.0, 1280 / frame_w, 800 / frame_h)
        if display_scale < 1.0:
            new_w = int(frame_w * display_scale)
            new_h = int(frame_h * display_scale)
            display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_frame = frame.copy()

        # Size the window to match the scaled frame exactly (once)
        # so mouse coordinates map 1-to-1 with display_frame pixels
        if not frame_resized:
            dh, dw = display_frame.shape[:2]
            cv2.resizeWindow("Video with ROI", dw, dh)
            frame_resized = True

        # Draw the rectangle on the display frame
        if start_point and end_point:
            cv2.rectangle(display_frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(display_frame, "ROI", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Video with ROI", display_frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            start_point = None
            end_point = None
        elif key == ord('p'):
            if start_point and end_point:
                x1 = max(0, min(start_point[0], end_point[0]))
                y1 = max(0, min(start_point[1], end_point[1]))
                x2 = max(0, min(max(start_point[0], end_point[0]), display_frame.shape[1]))
                y2 = max(0, min(max(start_point[1], end_point[1]), display_frame.shape[0]))
                if x1 >= x2 or y1 >= y2:
                    print("Invalid ROI, please draw a rectangle inside the frame.")
                    continue

                # Map display coordinates back to original frame coordinates
                orig_x1 = int(x1 / display_scale)
                orig_y1 = int(y1 / display_scale)
                orig_x2 = int(x2 / display_scale)
                orig_y2 = int(y2 / display_scale)

                roi_fractions = (
                    orig_x1 / frame_w,
                    orig_y1 / frame_h,
                    orig_x2 / frame_w,
                    orig_y2 / frame_h,
                )
                fx1, fy1, fx2, fy2 = roi_fractions
                print()
                print("=" * 50)
                print("  ROI POINTS (original video pixel coordinates)")
                print("=" * 50)
                print(f"  Top-Left:     ({orig_x1}, {orig_y1})")
                print(f"  Top-Right:    ({orig_x2}, {orig_y1})")
                print(f"  Bottom-Right: ({orig_x2}, {orig_y2})")
                print(f"  Bottom-Left:  ({orig_x1}, {orig_y2})")
                print()
                print(f"  Frame size:   {frame_w} x {frame_h}")
                print()
                print("  For test.py (auto-loaded from roi_config.txt):")
                print(f"  BILLBOARD_ROI = ({fx1}, {fy1}, {fx2}, {fy2})")
                print("=" * 50)
                print("  Saved to roi_config.txt")
                print()
                with open("roi_config.txt", "w") as f:
                    f.write(f"{fx1} {fy1} {fx2} {fy2}\n")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()