import cv2

print("Scanning for available cameras...")
found = []

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  Camera {i} → FOUND and working")
            found.append(i)
        else:
            print(f"  Camera {i} → Found but cannot read frames")
        cap.release()
    else:
        print(f"  Camera {i} → Not available")

print()
if found:
    print(f"Working cameras: {found}")
    print(f"Use --camera {found[-1]} for your external camera")
else:
    print("No cameras found at all")