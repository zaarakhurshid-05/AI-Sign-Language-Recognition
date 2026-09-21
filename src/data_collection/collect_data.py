from pathlib import Path
import csv
import re
import time

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT PATHS
# ============================================================

# collect_data.py
# -> data_collection
# -> src
# -> AI-Sign-Language-Recognition

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
ALPHABET_DIR = RAW_DATA_DIR / "alphabet"
SIGNS_DIR = RAW_DATA_DIR / "signs"

MODEL_PATH = BASE_DIR / "src" / "data_collection" / "hand_landmarker.task"


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

SEQUENCES_PER_CLASS = 40
SEQUENCE_LENGTH = 40

# Minimum number of frames in which at least one hand
# must be detected. Otherwise sequence is discarded.
MIN_HAND_FRAMES = 5

# Keep this True to match the previous mirrored-camera
# handedness behavior.
SWAP_HANDEDNESS = True

WINDOW_NAME = "ASL Data Collection"


# ============================================================
# ALPHABET CLASSES
# ============================================================

ALPHABET_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]


# ============================================================
# CONVERSATIONAL ASL CLASSES
# ============================================================

SIGN_CLASSES = [

    # Greetings / basic interaction
    "HI_HELLO",
    "BYE",
    "THANK_YOU",
    "PLEASE",
    "SORRY",
    "WELCOME",
    "NICE",
    "MEET",

    # People / pronouns
    "I_ME",
    "YOU",
    "MY",
    "YOUR",
    "WE_US",
    "THEY_THEM_THOSE",

    # Common conversation
    "HOW",
    "WHAT",
    "WHO",
    "WHERE",
    "WHEN",
    "WHY",
    "WHICH",

    # Responses / feelings
    "YES",
    "NO",
    "GOOD",
    "BAD",
    "FINE",
    "OKAY",
    "LIKE",
    "LOVE",
    "WANT",
    "NEED",

    # Everyday activities
    "GO",
    "COME",
    "EAT",
    "DRINK",
    "SLEEP",
    "WORK",
    "STUDY",
    "LIVE",
    "HELP",
    "WAIT",
    "STOP",
    "START",

    # Common things
    "HOME",
    "SCHOOL",
    "FOOD",
    "WATER",
    "MONEY",
    "PHONE",
    "CAR",
    "BOOK",

    # People / relationships
    "FAMILY",
    "FRIEND",
    "MOTHER",
    "FATHER",
    "BROTHER",
    "SISTER",

    # Time
    "TODAY",
    "TOMORROW",
    "YESTERDAY",
    "NOW",
    "LATER",
    "MORNING",
    "NIGHT",

    # Additional everyday signs
    "SEE_YOU_LATER",
    "DOG",
    "CAT",
    "AGAIN",
    "REPEAT",
    "MORE",
    "MILK",
    "GO_TO",
    "BATHROOM",
    "LEARN",
    "SIGN",
    "NAME",
    "ALL_DONE",
    "FINISH",
]


# ============================================================
# MEDIAPIPE HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


# ============================================================
# CREATE FOLDERS
# ============================================================

def create_directories():
    ALPHABET_DIR.mkdir(parents=True, exist_ok=True)
    SIGNS_DIR.mkdir(parents=True, exist_ok=True)


def create_class_directories():
    """
    Create folders for all alphabet and conversational signs.
    Existing data will NOT be deleted.
    """

    for class_name in ALPHABET_CLASSES:
        folder = ALPHABET_DIR / class_name
        folder.mkdir(parents=True, exist_ok=True)

    for class_name in SIGN_CLASSES:
        folder = SIGNS_DIR / class_name
        folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# SEQUENCE FILE FUNCTIONS
# ============================================================

def get_class_folder(class_name, is_alphabet):
    if is_alphabet:
        return ALPHABET_DIR / class_name

    return SIGNS_DIR / class_name


def get_existing_sequence_numbers(class_folder):
    """
    Returns all existing sequence numbers.

    Example:
        sequence_001.csv
        sequence_002.csv
        sequence_007.csv

    returns:
        [1, 2, 7]
    """

    numbers = []

    if not class_folder.exists():
        return numbers

    pattern = re.compile(r"sequence_(\d+)\.csv$")

    for file in class_folder.iterdir():

        if not file.is_file():
            continue

        match = pattern.match(file.name)

        if match:
            numbers.append(int(match.group(1)))

    return sorted(numbers)


def get_sequence_count(class_folder):
    return len(get_existing_sequence_numbers(class_folder))


def get_next_sequence_number(class_folder):

    numbers = get_existing_sequence_numbers(class_folder)

    if not numbers:
        return 1

    return max(numbers) + 1


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

def create_hand_landmarker():

    if not MODEL_PATH.exists():

        print("\nERROR: hand_landmarker.task not found.")
        print("Expected location:")
        print(MODEL_PATH)

        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}"
        )

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def empty_hand():

    # 21 landmarks × 3 values
    return [0.0] * 63


def extract_landmarks(result):
    """
    Always returns exactly 126 values.

    Left hand:
        21 × (x, y, z) = 63

    Right hand:
        21 × (x, y, z) = 63

    Total:
        126 features
    """

    left_hand = None
    right_hand = None

    detected_hands = getattr(result, "hand_landmarks", [])
    handedness_list = getattr(result, "handedness", [])

    for index, landmarks in enumerate(detected_hands):

        label = None

        if index < len(handedness_list):

            categories = handedness_list[index]

            if categories:
                label = categories[0].category_name

        if label is None:
            label = "Unknown"

        label = label.lower()

        # Swap because webcam is being displayed mirrored.
        if SWAP_HANDEDNESS:

            if label == "left":
                label = "right"

            elif label == "right":
                label = "left"

        # Extract 21 landmarks
        features = []

        for landmark in landmarks[:21]:

            features.append(float(landmark.x))
            features.append(float(landmark.y))
            features.append(float(landmark.z))

        # Safety padding
        while len(features) < 63:
            features.append(0.0)

        features = features[:63]

        # Assign hand
        if label == "left" and left_hand is None:

            left_hand = features

        elif label == "right" and right_hand is None:

            right_hand = features

        else:

            # Fallback if handedness is unknown
            if left_hand is None:
                left_hand = features

            elif right_hand is None:
                right_hand = features

    if left_hand is None:
        left_hand = empty_hand()

    if right_hand is None:
        right_hand = empty_hand()

    return left_hand + right_hand


# ============================================================
# DRAW LANDMARKS
# ============================================================

def draw_landmarks(frame, result):

    detected_hands = getattr(result, "hand_landmarks", [])
    handedness_list = getattr(result, "handedness", [])

    for hand_index, landmarks in enumerate(detected_hands):

        points = []

        for landmark in landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # Draw connections
        for start, end in HAND_CONNECTIONS:

            if start < len(points) and end < len(points):

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 255, 255),
                    2
                )

        # Hand label
        label = "Hand"

        if hand_index < len(handedness_list):

            categories = handedness_list[hand_index]

            if categories:
                label = categories[0].category_name

                if SWAP_HANDEDNESS:

                    if label.lower() == "left":
                        label = "Right"

                    elif label.lower() == "right":
                        label = "Left"

        if points:

            wrist_x, wrist_y = points[0]

            cv2.putText(
                frame,
                label,
                (wrist_x, wrist_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


# ============================================================
# TEXT UI
# ============================================================

def put_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def put_center_text(
    frame,
    text,
    y,
    scale=1.5,
    thickness=3
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    x = (frame.shape[1] - text_width) // 2

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera. "
            "Try changing CAMERA_INDEX from 0 to 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap


# ============================================================
# MANUAL READY SCREEN
# ============================================================

def countdown(
    cap,
    landmarker,
    class_name,
    sequence_number
):
    """
    Manual start screen.

    SPACE = start recording
    ESC   = exit

    The function name is kept as 'countdown' so the rest
    of the existing collection code does not need changes.
    """

    while True:

        ret, frame = cap.read()

        if not ret:
            return False

        # Mirror camera
        frame = cv2.flip(frame, 1)

        put_text(
            frame,
            f"Sign: {class_name}",
            (30, 45),
            0.8,
            2
        )

        put_text(
            frame,
            f"Sequence: {sequence_number:03d}",
            (30, 80),
            0.7,
            2
        )

        put_center_text(
            frame,
            "READY?",
            190,
            1.8,
            4
        )

        put_center_text(
            frame,
            "PRESS SPACE TO START",
            300,
            1.0,
            2
        )

        put_text(
            frame,
            "SPACE = Start",
            (30, frame.shape[0] - 55),
            0.7,
            2
        )

        put_text(
            frame,
            "ESC = Exit",
            (30, frame.shape[0] - 20),
            0.7,
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # SPACE = start recording
        if key == 32:
            return True

        # ESC = exit
        if key == 27:
            return False


# ============================================================
# SAVE CSV
# ============================================================

def save_sequence(
    class_folder,
    sequence_number,
    rows
):

    class_folder.mkdir(parents=True, exist_ok=True)

    file_path = (
        class_folder /
        f"sequence_{sequence_number:03d}.csv"
    )

    # Never overwrite
    if file_path.exists():

        raise FileExistsError(
            f"File already exists: {file_path}"
        )

    header = [
        "frame"
    ] + [
        f"feature_{i}"
        for i in range(126)
    ]

    with open(
        file_path,
        "x",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

        writer.writerows(rows)

    return file_path


# ============================================================
# COLLECT ONE SEQUENCE
# ============================================================

def collect_one_sequence(
    cap,
    landmarker,
    class_name,
    class_folder,
    sequence_number
):

    # ----------------------------------------
    # Manual ready screen
    # ----------------------------------------

    ready = countdown(
        cap,
        landmarker,
        class_name,
        sequence_number
    )

    if not ready:
        return "exit"

    rows = []

    hand_detected_frames = 0

    # ----------------------------------------
    # Capture 40 consecutive frames
    # ----------------------------------------

    for frame_number in range(SEQUENCE_LENGTH):

        ret, frame = cap.read()

        if not ret:

            print("Camera frame could not be read.")

            return "exit"

        # Mirror
        frame = cv2.flip(frame, 1)

        # MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame = np.ascontiguousarray(
            rgb_frame
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = landmarker.detect(mp_image)

        detected_hands = getattr(
            result,
            "hand_landmarks",
            []
        )

        if detected_hands:
            hand_detected_frames += 1

        # Extract exactly 126 features
        features = extract_landmarks(result)

        # Safety check
        if len(features) != 126:

            print(
                f"ERROR: Expected 126 features, "
                f"got {len(features)}"
            )

            return "exit"

        # Save frame row
        row = [
            frame_number + 1
        ] + features

        rows.append(row)

        # Draw
        draw_landmarks(
            frame,
            result
        )

        # UI
        put_text(
            frame,
            f"Sign: {class_name}",
            (30, 40),
            0.8,
            2
        )

        put_text(
            frame,
            f"Sequence: {sequence_number:03d}",
            (30, 75),
            0.7,
            2
        )

        put_text(
            frame,
            f"Frame: {frame_number + 1:02d}/{SEQUENCE_LENGTH}",
            (30, 110),
            0.7,
            2
        )

        put_text(
            frame,
            "Recording...",
            (30, 150),
            0.8,
            2
        )

        put_text(
            frame,
            "ESC = Exit",
            (30, frame.shape[0] - 25),
            0.7,
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return "exit"

    # ========================================================
    # DATA QUALITY CHECK
    # ========================================================

    if hand_detected_frames < MIN_HAND_FRAMES:

        print(
            f"\nSequence {sequence_number:03d} discarded."
        )

        print(
            f"Only {hand_detected_frames}/"
            f"{SEQUENCE_LENGTH} frames had a hand."
        )

        print(
            "Please perform the sign again."
        )

        # Show warning for a moment
        start = time.time()

        while time.time() - start < 1.5:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            put_center_text(
                frame,
                "NO HAND DETECTED - RETRY",
                250,
                1.2,
                3
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                return "exit"

        return "retry"

    # ========================================================
    # SAVE
    # ========================================================

    file_path = save_sequence(
        class_folder,
        sequence_number,
        rows
    )

    print(
        f"Saved: {file_path}"
    )

    return "saved"


# ============================================================
# COLLECT CLASS
# ============================================================

def collect_class(
    cap,
    landmarker,
    class_name,
    is_alphabet,
    target_count
):

    class_folder = get_class_folder(
        class_name,
        is_alphabet
    )

    class_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    current_count = get_sequence_count(
        class_folder
    )

    print("\n" + "=" * 60)

    print(
        f"Collecting: {class_name}"
    )

    print(
        f"Existing: {current_count}"
    )

    print(
        f"Target:   {target_count}"
    )

    print("=" * 60)

    while current_count < target_count:

        sequence_number = get_next_sequence_number(
            class_folder
        )

        result = collect_one_sequence(
            cap,
            landmarker,
            class_name,
            class_folder,
            sequence_number
        )

        if result == "exit":

            return False

        if result == "retry":

            continue

        if result == "saved":

            current_count = get_sequence_count(
                class_folder
            )

            print(
                f"{class_name}: "
                f"{current_count}/{target_count}"
            )

            # Small pause
            time.sleep(0.3)

    print("\n" + "=" * 60)

    print(
        f"COMPLETED: {class_name}"
    )

    print(
        f"Sequences: {current_count}"
    )

    print("=" * 60)

    return True


# ============================================================
# MENU
# ============================================================

def show_menu():

    print("\n")
    print("=" * 70)
    print("             ASL DATA COLLECTION")
    print("=" * 70)

    print("\nALPHABET SIGNS (A-Z)")
    print("-" * 70)

    number = 1

    menu_items = []

    for class_name in ALPHABET_CLASSES:

        folder = get_class_folder(
            class_name,
            True
        )

        count = get_sequence_count(folder)

        status = "✓" if count >= SEQUENCES_PER_CLASS else ""

        print(
            f"{number:3d}. "
            f"{class_name:<15} "
            f"[{count:02d}/{SEQUENCES_PER_CLASS}] "
            f"{status}"
        )

        menu_items.append(
            (class_name, True)
        )

        number += 1

    print("\nCONVERSATIONAL ASL SIGNS")
    print("-" * 70)

    for class_name in SIGN_CLASSES:

        folder = get_class_folder(
            class_name,
            False
        )

        count = get_sequence_count(folder)

        status = "✓" if count >= SEQUENCES_PER_CLASS else ""

        print(
            f"{number:3d}. "
            f"{class_name:<25} "
            f"[{count:02d}/{SEQUENCES_PER_CLASS}] "
            f"{status}"
        )

        menu_items.append(
            (class_name, False)
        )

        number += 1

    print("\n")
    print("N.  Add another sign/phrase")
    print("0.  Exit")

    print("=" * 70)

    return menu_items


# ============================================================
# ADD NEW SIGN
# ============================================================

def add_new_sign():

    print("\n" + "=" * 60)
    print("ADD NEW ASL SIGN / PHRASE")
    print("=" * 60)

    print(
        "Use a clear class name, for example:"
    )

    print(
        "GOOD_MORNING"
    )

    print(
        "SEE_YOU"
    )

    print(
        "HOW_ARE_YOU"
    )

    raw_name = input(
        "\nEnter class name: "
    ).strip()

    if not raw_name:

        print("No class entered.")

        return

    # Clean class name
    class_name = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        raw_name
    )

    class_name = class_name.strip("_")

    if not class_name:

        print("Invalid class name.")

        return

    # Normalize to uppercase
    class_name = class_name.upper()

    # Check alphabet
    if class_name in ALPHABET_CLASSES:

        print(
            f"{class_name} is already an alphabet class."
        )

        return

    # Check existing sign
    if class_name in SIGN_CLASSES:

        print(
            f"{class_name} already exists."
        )

        return

    SIGN_CLASSES.append(class_name)

    folder = SIGNS_DIR / class_name

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nAdded: {class_name}"
    )

    print(
        f"Folder: {folder}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("       AI SIGN LANGUAGE RECOGNITION")
    print("             DATA COLLECTION")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"Camera              : {CAMERA_INDEX}")
    print(f"Sequences / class   : {SEQUENCES_PER_CLASS}")
    print(f"Frames / sequence   : {SEQUENCE_LENGTH}")
    print(f"Features / frame    : 126")
    print(f"Model               : {MODEL_PATH}")

    # Create directories
    create_directories()
    create_class_directories()

    # Check MediaPipe model
    if not MODEL_PATH.exists():

        print("\nERROR:")
        print(
            "hand_landmarker.task was not found."
        )

        print(
            f"\nPut it here:\n{MODEL_PATH}"
        )

        return

    # Open camera
    try:

        cap = open_camera()

    except RuntimeError as error:

        print("\nERROR:")
        print(error)

        return

    # Create MediaPipe landmarker
    try:

        landmarker = create_hand_landmarker()

    except Exception as error:

        cap.release()

        print("\nERROR creating MediaPipe Hand Landmarker:")
        print(error)

        return

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    # Main menu loop
    try:

        while True:

            menu_items = show_menu()

            choice = input(
                "\nSelect class number: "
            ).strip()

            # EXIT
            if choice == "0":

                print("\nExiting...")

                break

            # ADD NEW SIGN
            if choice.lower() == "n":

                add_new_sign()

                continue

            # Validate number
            if not choice.isdigit():

                print(
                    "Please enter a valid number."
                )

                continue

            index = int(choice) - 1

            if index < 0 or index >= len(menu_items):

                print(
                    "Invalid selection."
                )

                continue

            # Get class
            class_name, is_alphabet = menu_items[index]

            class_folder = get_class_folder(
                class_name,
                is_alphabet
            )

            current_count = get_sequence_count(
                class_folder
            )

            # If already complete
            if current_count >= SEQUENCES_PER_CLASS:

                print("\n" + "=" * 60)

                print(
                    f"{class_name} is already complete."
                )

                print(
                    f"Current sequences: {current_count}"
                )

                print("=" * 60)

                extra_choice = input(
                    "Add more sequences? [y/N]: "
                ).strip().lower()

                if extra_choice != "y":

                    continue

                extra_input = input(
                    "How many additional sequences? "
                ).strip()

                if not extra_input.isdigit():

                    print(
                        "Invalid number."
                    )

                    continue

                extra_count = int(extra_input)

                if extra_count <= 0:

                    continue

                target_count = (
                    current_count +
                    extra_count
                )

            else:

                target_count = SEQUENCES_PER_CLASS

            # Start collection
            success = collect_class(
                cap,
                landmarker,
                class_name,
                is_alphabet,
                target_count
            )

            if not success:

                print(
                    "\nCollection stopped."
                )

                break

    except KeyboardInterrupt:

        print(
            "\n\nProgram interrupted."
        )

    finally:

        cap.release()

        landmarker.close()

        cv2.destroyAllWindows()

        print(
            "\nCamera released."
        )

        print(
            "Data collection program closed."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
