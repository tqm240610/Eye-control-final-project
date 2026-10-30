"""Demo: dung mat (qua webcam + MediaPipe) de bat/tat den LED noi voi Arduino.

Y tuong: Arduino doc lenh tu cong Serial (USB) va dieu khien 1 chan digital
noi voi LED. May tinh dung webcam + MediaPipe Face Landmarker de phat hien
trang thai mat, roi gui lenh "1"/"0" qua Serial cho Arduino.

Ba che do dieu khien (chon bang --mode):
  hold   (mac dinh) Nham ca hai mat  -> den SANG. Mo mat -> den TAT.
  blink  Moi lan CHOP MAT hop le (nham roi mo trong ~0.08-0.8s) se
         DAO TRANG THAI den (dang tat -> sang, dang sang -> tat).
  wink   Nham mat trai -> den xanh; nham mat phai -> den do; mo ca hai -> tat.

Can nap san sketch arduino_led/arduino_led.ino vao board truoc khi chay file nay.

Vi du:
    python eye_led_control.py --port COM5
    python eye_led_control.py --port COM5 --mode blink
    python eye_led_control.py --port /dev/ttyACM0 --mode blink --threshold 0.6

Phim tat trong cua so webcam: Q thoat | R reset bo dem chop mat | +/- doi nguong
"""
import argparse
import time
from dataclasses import dataclass

try:
    import serial  # pyserial
except ImportError:
    serial = None


@dataclass
class EyeState:
    """Giong eye_control.py: chi tinh lan nham-mo hop le, mat mat thi huy."""
    blinks: int = 0
    closed_since: float | None = None
    last_seen: float | None = None
    closed: bool = False
    just_blinked: bool = False

    def update(self, now, scores, threshold=0.55):
        self.just_blinked = False
        if scores is None:
            self.closed_since = None
            self.last_seen = None
            self.closed = False
            return False
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
            return True
        if self.closed_since is not None:
            duration = now - self.closed_since
            if 0.08 <= duration <= 0.8:
                self.blinks += 1
                self.just_blinked = True
        self.closed_since = None
        self.closed = False
        return False


class LedLink:
    """Gui lenh bat/tat qua Serial; khong co Arduino thi chi in ra man hinh (che do mo phong)."""

    def __init__(self, port, baud=9600):
        self.port_name = port
        self.conn = None
        self._state = False
        if port and serial is not None:
            self.conn = serial.Serial(port, baud, timeout=0.2)
            time.sleep(2.0)  # cho Arduino khoi dong lai sau khi mo cong Serial

    def set(self, on):
        if on == self._state:
            return
        self._state = on
        command = '1\n' if on else '0\n'
        if self.conn is not None:
            self.conn.write(command.encode('ascii'))
        else:
            print(f'[MO PHONG - khong co Arduino] LED {"BAT" if on else "TAT"}')

    def close(self):
        if self.conn is not None:
            self.set(False)
            self.conn.close()


class DualLedLink:
    """Gui trang thai hai den: green/red, hoac none de tat ca hai."""

    def __init__(self, port, baud=9600):
        self.port_name = port
        self.conn = None
        self._state = None
        if port and serial is not None:
            self.conn = serial.Serial(port, baud, timeout=0.2)
            time.sleep(2.0)

    def set(self, state):
        if state == self._state:
            return
        self._state = state
        if self.conn is not None:
            self.conn.write((state + chr(10)).encode('ascii'))
        else:
            print(f'[MO PHONG] LED: {state.upper()}')

    def close(self):
        if self.conn is not None:
            self.set('none')
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', default=None,
                         help='Cong Serial cua Arduino, vi du COM5 (Windows) hoac /dev/ttyACM0 (Linux). '
                              'Bo trong de chay mo phong, chi in ra man hinh.')
    parser.add_argument('--baud', type=int, default=9600)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.55)
    parser.add_argument('--mode', choices=['gaze'], default='gaze')
    args = parser.parse_args()

    import cv2
    import mediapipe as mp
    from pathlib import Path
    from mediapipe.tasks.python import vision

    led = DualLedLink(args.port, args.baud)
    model = Path(__file__).resolve().with_name('face_landmarker.task')
    if not model.exists():
        raise SystemExit(f'Khong tim thay model: {model}. Chay eye_camera.py truoc de tai model.')

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
    led_on = False
    window = 'Minh - Mat dieu khien den (Arduino)'
    try:
        if not cap.isOpened():
            raise RuntimeError('Khong mo duoc webcam.')
        with vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Khong doc duoc khung hinh webcam.')
                now = time.monotonic()
                timestamp = max(previous_timestamp + 1, int(now * 1000))
                previous_timestamp = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)

                led_state = 'none'
                face_found = bool(result.face_landmarks)
                eyes_found = False
                pupil_x = None
                if face_found:
                    lm = result.face_landmarks[0]
                    # Iris centers, mapped into the mirrored preview's horizontal coordinates.
                    if len(lm) >= 478:
                        iris_r = sum(lm[i].x for i in range(468, 473)) / 5
                        iris_l = sum(lm[i].x for i in range(473, 478)) / 5
                        pupil_x = 1.0 - (iris_r + iris_l) / 2
                        eyes_found = True
                        if pupil_x < (1.0 / 3.0):
                            led_state = 'red'
                        elif pupil_x >= (2.0 / 3.0):
                            led_state = 'green'
                led.set(led_state)
                led_on = led_state != 'none'
                if not face_found:
                    status = 'KHONG NHAN RA MAT - NHIN RO WEBCAM'
                elif not eyes_found:
                    status = 'THAY MAT NHUNG MODEL KHONG CO DIEM CON NGUOI'
                else:
                    status = {'green': 'MAT O VUNG PHAI - DEN XANH',
                              'red': 'MAT O VUNG TRAI - DEN DO',
                              'none': 'MAT O VUNG GIUA - TAT DEN'}[led_state]

                frame = cv2.flip(frame, 1)
                # Draw three gaze zones over the mirrored webcam preview.
                h, w = frame.shape[:2]
                left_x, right_x = w // 3, (2 * w) // 3
                cv2.line(frame, (left_x, 0), (left_x, h), (0, 0, 255), 2)
                cv2.line(frame, (right_x, 0), (right_x, h), (0, 255, 0), 2)
                cv2.putText(frame, 'DO - TRAI', (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                cv2.putText(frame, 'GIUA - TAT', (left_x + 10, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.putText(frame, 'PHAI - XANH', (right_x + 10, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                if face_found and eyes_found:
                    lm = result.face_landmarks[0]
                    px = int(pupil_x * frame.shape[1])
                    py = int(((lm[468].y + lm[473].y) / 2) * frame.shape[0])
                    cv2.circle(frame, (px, py), 8, (0, 255, 255), -1)
                color = (0, 0, 255) if led_on else (0, 255, 0)
                lines = [
                    f'Che do: {args.mode} | Cong: {args.port or "mo phong"}',
                    status,
                    f'Den: {led_state.upper()}',
                    'Q: thoat',
                ]
                for row, line in enumerate(lines):
                    y = 28 + row * 28
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)
                if led_on:
                    cv2.rectangle(frame, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), color, 5)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        led.close()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(f'Loi: {error}')


