"""Eye-controlled webcam demo: gaze moves the mouse, blinks click, drowsy
eye-closure triggers a sound alarm.

Hotkeys: Q quit | R reset | C recalibrate center | +/- mouse gain |
         M toggle mouse control on/off | S toggle sound on/off

Usage:
    python eye_control.py                  # mouse control + monitor, sound on
    python eye_control.py --no-mouse        # monitor-only, like eye_camera.py
    python eye_control.py --no-sound        # disable the audio alarm
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time
import urllib.request

from sound_alert import SoundAlert

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
# Anatomical left/right of the person, before mirroring the display.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


def find_pupil(frame, center, radius):
    """Find a dark pupil candidate INSIDE the iris; return None if ambiguous.

    Iris landmarks alone are not a segmentation of the pupil boundary.
    """
    import cv2
    import numpy as np
    cx, cy = center
    if radius < 5:
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(cx-radius)), max(0, int(cy-radius))
    x2, y2 = min(w, int(cx+radius)+1), min(h, int(cy+radius)+1)
    if x2 <= x1 or y2 <= y1:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    yy, xx = np.indices(gray.shape)
    mask = (xx-(cx-x1))**2 + (yy-(cy-y1))**2 < (radius*0.85)**2
    pixels = gray[mask]
    if pixels.size < 20 or float(np.percentile(pixels, 85)-np.percentile(pixels, 10)) < 18:
        return None
    threshold, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = ((gray <= threshold) & mask).astype(np.uint8)*255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 0.025*3.14159*radius**2 <= area <= 0.60*3.14159*radius**2:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or 4*3.14159*area/perimeter**2 < 0.40:
            continue
        moments = cv2.moments(contour)
        px = moments['m10']/moments['m00'] + x1
        py = moments['m01']/moments['m00'] + y1
        if (px-cx)**2 + (py-cy)**2 > (radius*0.45)**2:
            continue
        candidates.append((area, (round(px), round(py)), contour + np.array([[[x1, y1]]])))
    if not candidates:
        return None
    _, point, contour = max(candidates, key=lambda candidate: candidate[0])
    return point, contour


@dataclass
class EyeState:
    """Count only completed short closures; losing the face cancels a closure."""
    blinks: int = 0
    closed_since: float | None = None
    last_seen: float | None = None
    closed: bool = False
    just_blinked: bool = False

    def update(self, now, scores, threshold=0.55, alarm_seconds=1.5):
        self.just_blinked = False
        if scores is None:
            self.closed_since = None
            self.last_seen = None
            self.closed = False
            return 'NO FACE / EYES UNKNOWN', 0.0, False
        if self.last_seen is not None and now - self.last_seen > 0.5:
            self.closed_since = None
            self.closed = False
        self.last_seen = now
        limit = threshold - 0.10 if self.closed else threshold
        both_closed = min(scores) >= limit
        if both_closed:
            if self.closed_since is None:
                self.closed_since = now
            self.closed = True
            duration = now - self.closed_since
            alarm = duration >= alarm_seconds
            return ('LONG EYE CLOSURE' if alarm else 'BOTH EYES CLOSED'), duration, alarm
        if self.closed_since is not None:
            duration = now - self.closed_since
            if 0.08 <= duration <= 0.8:
                self.blinks += 1
                self.just_blinked = True
        self.closed_since = None
        self.closed = False
        status = 'ONE EYE CLOSED' if max(scores) >= threshold else 'EYES OPEN'
        return status, 0.0, False


@dataclass
class GazeMouse:
    """Maps the average iris offset inside the eye box to screen coordinates."""
    screen_w: int
    screen_h: int
    enabled: bool = True
    smoothing: float = 0.22        # fraction of target applied per frame; lower = smoother
    sensitivity: float = 2.2        # how far a small eye movement swings the cursor
    deadzone: float = 0.015        # ignore tiny iris-position changes near calibrated center
    center_x: float = 0.5
    center_y: float = 0.5
    cursor_x: float = field(init=False)
    cursor_y: float = field(init=False)
    last_click_time: float = 0.0

    def __post_init__(self):
        self.cursor_x = self.screen_w / 2
        self.cursor_y = self.screen_h / 2

    def recalibrate(self, norm_x, norm_y):
        self.center_x, self.center_y = norm_x, norm_y

    def update(self, norm_x, norm_y):
        """norm_x/norm_y: average iris position normalized 0..1 inside eye boxes."""
        import pyautogui
        # Raw camera image is unmirrored: looking to screen-right shifts iris left in the image.
        # Invert both axes to match the user's gaze direction to cursor movement.
        def apply_deadzone(value):
            if abs(value) <= self.deadzone:
                return 0.0
            return value - (self.deadzone if value > 0 else -self.deadzone)

        dx = apply_deadzone(self.center_x - norm_x) * self.sensitivity
        dy = apply_deadzone(self.center_y - norm_y) * self.sensitivity
        target_x = (0.5 + dx) * self.screen_w
        target_y = (0.5 + dy) * self.screen_h
        target_x = min(max(target_x, 0), self.screen_w - 1)
        target_y = min(max(target_y, 0), self.screen_h - 1)
        self.cursor_x += (target_x - self.cursor_x) * self.smoothing
        self.cursor_y += (target_y - self.cursor_y) * self.smoothing
        if self.enabled:
            pyautogui.moveTo(self.cursor_x, self.cursor_y, _pause=False)

    def click(self):
        import pyautogui
        now = time.monotonic()
        if now - self.last_click_time < 0.3:  # debounce accidental double-fire
            return
        self.last_click_time = now
        if self.enabled:
            pyautogui.click()


def ensure_model(path):
    if path.exists():
        return
    print('Downloading Face Landmarker model from Google (first run only)...')
    temporary = path.with_suffix('.download')
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def eye_iris_norm_position(landmarks, indices, iris_ids, w, h):
    """Normalized (0..1, 0..1) iris center position inside the eye's bounding box."""
    eye_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    iris_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_ids]
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    icx = sum(p[0] for p in iris_points) / len(iris_points)
    icy = sum(p[1] for p in iris_points) / len(iris_points)
    return (icx - x1) / (x2 - x1), (icy - y1) / (y2 - y1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.55)
    parser.add_argument('--alarm-seconds', type=float, default=1.5)
    parser.add_argument('--no-mouse', action='store_true', help='disable cursor control')
    parser.add_argument('--no-sound', action='store_true', help='disable the audio alarm')
    parser.add_argument('--sensitivity', type=float, default=2.2, help='gaze-to-cursor gain')
    args = parser.parse_args()
    if not 0.15 <= args.threshold <= 0.95 or args.alarm_seconds <= 0:
        parser.error('threshold: 0.15..0.95; alarm-seconds must be positive')

    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import vision

    mouse = None
    if not args.no_mouse:
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            screen_w, screen_h = pyautogui.size()
            mouse = GazeMouse(screen_w, screen_h, enabled=True, sensitivity=args.sensitivity)
        except Exception as error:
            print(f'Mouse control unavailable ({error}); continuing in monitor-only mode.')

    sound = None if args.no_sound else SoundAlert(interval_seconds=1.0)

    model = Path(__file__).resolve().with_name('face_landmarker.task')
    ensure_model(model)
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    cap = cv2.VideoCapture(args.camera)
    state = EyeState()
    threshold = args.threshold
    previous_timestamp = -1
    window = 'Minh - Eye Control'
    calibrated = False
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Close Zoom/Zalo camera or try --camera 1.')
        with vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Cannot read webcam frame. Check connection and permissions.')
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                now = time.monotonic()
                timestamp = max(previous_timestamp + 1, int(now * 1000))
                previous_timestamp = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                scores = None
                eye_labels = []
                pupil_status = []
                gaze_samples = []
                if result.face_landmarks and result.face_blendshapes:
                    landmarks = result.face_landmarks[0]
                    values = {item.category_name: item.score for item in result.face_blendshapes[0]}
                    if 'eyeBlinkLeft' in values and 'eyeBlinkRight' in values:
                        scores = (values['eyeBlinkLeft'], values['eyeBlinkRight'])
                    h, w = frame.shape[:2]
                    raw_frame = frame.copy()
                    for label, indices, iris_ids, blink_key in (
                        ('LEFT', LEFT_EYE, LEFT_IRIS, 'eyeBlinkLeft'),
                        ('RIGHT', RIGHT_EYE, RIGHT_IRIS, 'eyeBlinkRight')):
                        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h))
                                           for i in indices], dtype=np.int32)
                        cv2.polylines(frame, [points], True, (0, 255, 0), 1)
                        for point in points:
                            cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
                        x1 = max(0, int(points[:, 0].min()) - 10)
                        y1 = max(0, int(points[:, 1].min()) - 10)
                        x2 = min(w - 1, int(points[:, 0].max()) + 10)
                        y2 = min(h - 1, int(points[:, 1].max()) + 10)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_text = label + ': UNKNOWN'
                        eye_open = values.get(blink_key, 1.0) < threshold - 0.10
                        if eye_open and len(landmarks) >= 478:
                            iris = np.array([(landmarks[i].x*w, landmarks[i].y*h)
                                             for i in iris_ids], dtype=np.float32)
                            (cx, cy), radius = cv2.minEnclosingCircle(iris)
                            center = (round(cx), round(cy))
                            if 0 <= cx < w and 0 <= cy < h:
                                cv2.circle(frame, center, round(radius), (255, 255, 0), 1)
                                pupil = find_pupil(raw_frame, (cx, cy), radius)
                                label_text = label + ': IRIS ONLY'
                                if pupil is not None:
                                    point, contour = pupil
                                    cv2.drawContours(frame, [contour], -1, (0, 255, 255), 1)
                                    cv2.drawMarker(frame, point, (0, 0, 255),
                                                   cv2.MARKER_CROSS, 10, 1)
                                    label_text = label + ': PUPIL CANDIDATE'
                                    pupil_status.append(f'{label} pupil: ({w-1-point[0]}, {point[1]})')
                            if eye_open:
                                norm = eye_iris_norm_position(landmarks, indices, iris_ids, w, h)
                                if norm is not None:
                                    gaze_samples.append(norm)
                        eye_labels.append((label_text, w - 1 - x2, max(15, y1 - 6)))
                status, duration, alarm = state.update(now, scores, threshold, args.alarm_seconds)

                if sound is not None:
                    sound.set_armed(alarm)

                if mouse is not None and gaze_samples:
                    gx = sum(s[0] for s in gaze_samples) / len(gaze_samples)
                    gy = sum(s[1] for s in gaze_samples) / len(gaze_samples)
                    if not calibrated:
                        mouse.recalibrate(gx, gy)
                        calibrated = True
                    mouse.update(gx, gy)
                    if state.just_blinked:
                        mouse.click()

                frame = cv2.flip(frame, 1)
                for label, x, y in eye_labels:
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 0, 0), 3)
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 255, 0), 1)
                color = (0, 0, 255) if alarm else (0, 255, 0)
                mouse_state = 'OFF' if mouse is None else ('ON' if mouse.enabled else 'PAUSED')
                mouse_gain = '--' if mouse is None else f'{mouse.sensitivity:.1f}'
                sound_state = 'OFF' if sound is None else 'ON'
                lines = [status, f'Blinks: {state.blinks} | Closed: {duration:.2f}s',
                         f'Threshold: {threshold:.2f} | Alarm: {args.alarm_seconds:.1f}s',
                         f'Mouse: {mouse_state} | Gain: {mouse_gain} | Sound: {sound_state}',
                         'Q:quit R:reset C:recalibrate M:mouse S:sound +/-:gain']
                if scores is not None:
                    lines.append(f'Blink score L: {scores[0]:.2f}  R: {scores[1]:.2f}')
                lines.extend(pupil_status)
                for row, line in enumerate(lines):
                    y = 28 + row * 28
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)
                if alarm:
                    cv2.rectangle(frame, (2, 2), (frame.shape[1]-3, frame.shape[0]-3), color, 5)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('r'):
                    state = EyeState()
                if key == ord('c') and gaze_samples:
                    gx = sum(s[0] for s in gaze_samples) / len(gaze_samples)
                    gy = sum(s[1] for s in gaze_samples) / len(gaze_samples)
                    mouse.recalibrate(gx, gy) if mouse else None
                if key == ord('m') and mouse is not None:
                    mouse.enabled = not mouse.enabled
                if key == ord('s'):
                    if sound is not None:
                        sound.close()
                        sound = None
                    else:
                        sound = SoundAlert(interval_seconds=1.0)
                if key in (ord('+'), ord('='), ord('-')) and mouse is not None:
                    step = 0.1 if key != ord('-') else -0.1
                    mouse.sensitivity = max(0.6, min(4.0, mouse.sensitivity + step))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if sound is not None:
            sound.close()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(f'Error: {error}')
