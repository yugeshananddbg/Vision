import cv2
import numpy as np
import sys


def auto_calibrate(frame):
    """
    Automatically detects the archery target to find its center and radius.
    """
    # Convert to grayscale and apply a blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use HoughCircles to detect the target
    # These parameters might need tuning for different video resolutions/lighting
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of accumulator resolution
        minDist=100,  # Minimum distance between detected centers
        param1=50,  # Upper threshold for the internal Canny edge detector
        param2=60,  # Threshold for center detection
        minRadius=50,  # Minimum radius to be detected
        maxRadius=300  # Maximum radius to be detected
    )

    if circles is not None:
        # Find the circle with the largest radius
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])

        center = (largest_circle[0], largest_circle[1])
        radius = largest_circle[2]

        print(f"✅ Target auto-detected! Center: {center}, Radius: {radius}")
        return center, radius

    print("❌ Auto-calibration failed: No target found.")
    return None, None


def get_score(distance, outer_radius, rings=5):
    """
    Calculates the score based on distance from the center.
    The target in the video has 5 scoring zones (Yellow, Red, Blue, Black, White).
    """
    if distance > outer_radius:
        return 0  # Missed the target

    # Assuming 5 rings, each worth 2 points more than the one outside it (e.g., 2, 4, 6, 8, 10)
    ring_width = outer_radius / rings
    ring_index = int(distance // ring_width)
    score = (rings - ring_index) * 2  # Example scoring: 10, 8, 6, 4, 2

    # Bullseye gets max score
    if score > 10:
        score = 10

    return max(0, score)


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # --- Automatic Calibration Stage ---
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    target_center, outer_radius = auto_calibrate(first_frame)

    if target_center is None:
        print("Exiting due to calibration failure.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # --- Detection Stage ---
    # Reset video capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_score = 0
    detected_arrows = []  # To avoid recounting arrows in the same spot

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()

        # Draw the detected target for visualization
        cv2.circle(output, target_center, outer_radius, (255, 0, 255), 3)
        cv2.circle(output, target_center, 5, (0, 255, 0), -1)

        # Use grayscale for arrow detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Detect arrows (small circles)
        # Parameters are tuned for small arrow holes
        arrow_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                                         minDist=20, param1=50, param2=25,
                                         minRadius=2, maxRadius=15)

        if arrow_circles is not None:
            arrow_circles = np.uint16(np.around(arrow_circles))
            for (x, y, r) in arrow_circles[0, :]:

                # Check if this arrow is already scored
                is_new_arrow = True
                for (ax, ay, ascore) in detected_arrows:
                    # If a detected arrow is too close to an old one, ignore it
                    if np.sqrt((x - ax) ** 2 + (y - ay) ** 2) < 20:
                        is_new_arrow = False
                        # Redraw old score
                        cv2.putText(output, f"+{ascore}", (ax + 10, ay),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
                        break

                if is_new_arrow:
                    # Distance from target center
                    distance = np.sqrt((x - target_center[0]) ** 2 + (y - target_center[1]) ** 2)

                    # Only score arrows that are on the target
                    if distance <= outer_radius:
                        score = get_score(distance, outer_radius)
                        total_score += score
                        detected_arrows.append((x, y, score))
                        print(f"New arrow at ({x},{y}), Score = +{score}, Total = {total_score}")
                        # Display score next to the arrow
                        cv2.putText(output, f"+{score}", (x + 10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

        # Display total score
        cv2.putText(output, f"Total Score: {total_score}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(output, f"Total Score: {total_score}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Archery Score Detection", output)
        key = cv2.waitKey(60) & 0xFF  # Wait a bit longer between frames
        if key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Get the video file path from the command line or use a default
    video_file = "archey.mp4"  # Make sure this file is in the same directory
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    main(video_file)