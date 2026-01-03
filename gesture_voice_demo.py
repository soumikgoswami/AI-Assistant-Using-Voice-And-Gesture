# gesture_voice_demo_improved.py
"""
Improved Gesture + Voice Demo
- Vosk for offline speech (sounddevice)
- MediaPipe hand detection for up to 2 hands
- Temporal smoothing for landmarks
- Relative thresholds for finger detection
- Debounce + command queue
"""
import os
import time
import sys
import queue
import threading
import collections
import difflib
import json # Moved json import to top

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ------------ Configuration ------------
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # <- update to your model path
# Note: set MAX_HANDS to 2 to enable two-hand detection
SAMPLE_RATE = 16000
COMMAND_COOLDOWN = 0.7       # reduced cooldown for more responsive gestures
SMOOTH_WINDOW = 6            # increased smoothing for more stable detection
MAX_HANDS = 2
# Global cooldown between any two gestures (seconds)
GLOBAL_GESTURE_COOLDOWN = 3.0  # seconds between any two gestures
# Debug / tuning options
DEBUG = True                 # Always show debug info for better feedback
HELP_STABLE_FRAMES = 4       # reduced frames needed for stability
HELP_MOVEMENT_THRESHOLD = 0.05 # increased threshold for more forgiving movement detection
GESTURE_DISPLAY_DURATION = 5.0 # seconds to display any gesture overlay

# Map recognized phrases (and fuzzy matches) to commands
VOICE_PHRASES = {
    # Movement commands
    "forward": "FORWARD",
    "move forward": "FORWARD",
    "go forward": "FORWARD",
    "walk forward": "FORWARD",
    
    "back": "BACK",
    "move back": "BACK",
    "go back": "BACK",
    "backward": "BACK",
    "walk back": "BACK",
    
    "left": "LEFT",
    "turn left": "LEFT",
    "go left": "LEFT",
    "move left": "LEFT",
    
    "right": "RIGHT",
    "turn right": "RIGHT",
    "go right": "RIGHT",
    "move right": "RIGHT",
    
    # Stop commands
    "stop": "STOP",
    "halt": "STOP",
    "freeze": "STOP",
    "stay": "STOP",
    "pause": "STOP",
    
    # New speed commands
    "faster": "SPEED_UP",
    "speed up": "SPEED_UP",
    "increase speed": "SPEED_UP",
    
    "slower": "SLOW_DOWN",
    "slow down": "SLOW_DOWN",
    "decrease speed": "SLOW_DOWN",
    
    # New action commands
    "start": "START",
    "begin": "START",
    "resume": "START",
    
    "help": "HELP",
    "what can i say": "HELP",
    "show commands": "HELP"
    ,
    # Circle gesture voice commands
    "circle": "CIRCLE",
    "draw circle": "CIRCLE",
    "make circle": "CIRCLE",
    "make a circle": "CIRCLE",
    "do a circle": "CIRCLE"
}
FUZZY_CUTOFF = 0.7

# ------------ MediaPipe setup ------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ------------ Helpers: smoothing and normalization ------------
def avg_landmarks(deque_of_lm):
    """Average landmarks from deque of landmark arrays (N, 21, 3) -> (21,3)."""
    stacked = np.stack(list(deque_of_lm), axis=0)  # (N,21,3)
    return np.mean(stacked, axis=0)              # (21,3)

def normalize_landmarks(lm_array):
    """
    Convert landmarks to relative coords normalized by palm size to reduce scale/camera distance effects.
    lm_array: (21,3) with x,y,z in normalized coords.
    returns normalized (21,3)
    """
    # Use wrist (0) as origin
    origin = lm_array[0, :2].copy()
    rel = lm_array[:, :2] - origin  # (21,2)
    # palm size: distance between wrist (0) and middle_finger_mcp (9) or index_mcp (5)
    p1 = lm_array[0, :2]
    p2 = lm_array[9, :2] if not np.allclose(lm_array[9,:2], 0) else lm_array[5,:2]
    palm_size = np.linalg.norm(p2 - p1)
    if palm_size < 1e-6:
        palm_size = 1.0
    rel = rel / palm_size
    # return x,y,z with normalized x,y and original z scaled by palm_size
    z = lm_array[:, 2:3] / (palm_size + 1e-9)
    return np.hstack([rel, z])

def fingers_up_from_norm(norm_lm, handedness_label="Right"):
    """
    Determine fingers up using normalized landmarks.
    norm_lm: (21,3) normalized coordinates
    handedness_label: "Left" or "Right" to interpret thumb direction
    Returns dict of finger booleans.
    """
    # indices: TIP and PIP/MCP indices for each finger
    tips = {"THUMB":4, "INDEX":8, "MIDDLE":12, "RING":16, "PINKY":20}
    dips = {"THUMB":3, "INDEX":6, "MIDDLE":10, "RING":14, "PINKY":18}
    mcps = {"THUMB":2, "INDEX":5, "MIDDLE":9, "RING":13, "PINKY":17}  # Added MCPs for better detection
    fingers = {}
    # For non-thumb fingers: tip y lower (more negative in normalized space) than pip -> finger up
    for f in ["INDEX", "MIDDLE", "RING", "PINKY"]:
        tip_y = norm_lm[tips[f], 1]
        dip_y = norm_lm[dips[f], 1]
        mcp_y = norm_lm[mcps[f], 1]  # Consider MCP position too
        # More lenient tolerance for finger bend and check overall finger direction
        is_pointing_up = (tip_y < mcp_y)  # Overall direction check
        is_extended = (tip_y < dip_y - 0.05)  # Reduced threshold from 0.08 to 0.05
        fingers[f] = 1 if (is_pointing_up and is_extended) else 0

    # Improved thumb detection using both x and y coordinates
    thumb_tip = norm_lm[tips["THUMB"], :2]  # x,y
    thumb_ip = norm_lm[dips["THUMB"], :2]  # x,y
    thumb_mcp = norm_lm[mcps["THUMB"], :2]  # x,y
    
    # Calculate thumb extension vector
    thumb_vec = thumb_tip - thumb_ip
    thumb_base_vec = thumb_ip - thumb_mcp
    
    # Magnitude of extension
    thumb_extension = np.linalg.norm(thumb_vec)
    
    if handedness_label.lower().startswith("right"):
        # For right hand: check if thumb is extended and pointing rightward/upward
        fingers["THUMB"] = 1 if (thumb_extension > 0.08 and thumb_vec[0] > 0.04) else 0
    else:
        # For left hand: check if thumb is extended and pointing leftward/upward
        fingers["THUMB"] = 1 if (thumb_extension > 0.08 and thumb_vec[0] < -0.04) else 0

    return fingers

def classify_gesture_from_fingers(fingers):
    total = sum(fingers.values())
    # Prioritize explicit gestures
    if total == 0:
        return "STOP"
    # Thumb + Index + Middle + Ring together -> SLOW_DOWN (four-finger signal for speed decrease)
    if fingers.get("THUMB", 0) == 1 and fingers.get("INDEX", 0) == 1 and fingers.get("MIDDLE", 0) == 1 and fingers.get("RING", 0) == 1 and \
       fingers.get("PINKY", 0) == 0:
        return "SLOW_DOWN"

    # Thumb + Index + Middle together -> SPEED_UP (three-finger signal for speed increase)
    if fingers.get("THUMB", 0) == 1 and fingers.get("INDEX", 0) == 1 and fingers.get("MIDDLE", 0) == 1 and \
       sum([fingers.get(k, 0) for k in ("RING", "PINKY")]) == 0:
        return "SPEED_UP"

    # Thumb + Index together -> CIRCLE (pinch-like circle intent)
    if fingers.get("THUMB", 0) == 1 and fingers.get("INDEX", 0) == 1 and \
       sum([fingers.get(k, 0) for k in ("MIDDLE", "RING", "PINKY")]) == 0:
        return "CIRCLE"
    # All five fingers up on a single hand -> HELP (open palm used as 'help' gesture)
    # NOTE: The multi-hand check handles this more reliably
    if total == 5:
        # Returning HELP here will make the single open hand trigger it.
        # This will be overridden by the two-hand check if applicable.
        return "HELP" 
    # Thumb-only (thumb up) -> START
    if fingers.get("THUMB", 0) == 1 and sum([fingers.get(k,0) for k in ("INDEX","MIDDLE","RING","PINKY")]) == 0:
        return "START"
    if total == 1 and fingers.get("INDEX",0):
        return "FORWARD"
    if total == 2 and fingers.get("INDEX",0) and fingers.get("MIDDLE",0):
        return "BACK"
    if total == 3:
        # Could be INDEX, MIDDLE, RING OR Thumb, Index, Middle (SPEED_UP)
        # SPEED_UP is prioritized above, so this is for other 3-finger combos
        return "LEFT"
    if total == 4:
        # Could be THUMB, INDEX, MIDDLE, RING (SLOW_DOWN)
        # SLOW_DOWN is prioritized above, so this is for other 4-finger combos
        return "RIGHT"
    return None


def detect_circle_from_path(path_deque):
    """Detect if points in path_deque form a circular motion.
    path_deque: deque of (x,y) normalized coords (wrist positions over time)
    Returns (True, center, mean_radius) if circle-like motion detected,
    otherwise (False, None, None).
    center is in normalized coords (x,y), mean_radius is in normalized units.
    """
    if len(path_deque) < 8:  # need at least a handful of samples
        return (False, None, None)
    pts = np.array(list(path_deque))  # (N,2) or (N,3)
    # use only x,y columns for circle detection
    if pts.shape[1] >= 2:
        pts2 = pts[:, :2]
    else:
        pts2 = pts
    # centroid
    c = pts2.mean(axis=0)
    vecs = pts2 - c
    radii = np.linalg.norm(vecs, axis=1)
    mean_r = radii.mean()

    # Ignore tiny motions
    if mean_r < 0.015:
        return (False, None, None)

    # radius stability: require roughly consistent distance from centroid
    radial_var = radii.std() / (mean_r + 1e-9)
    if radial_var > 0.9:
        return (False, None, None)

    # angles and total swept angle
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    ang_unwrapped = np.unwrap(angles)

    # total sweep in radians (span)
    total_span = ang_unwrapped.max() - ang_unwrapped.min()

    # require a substantial sweep (approx > ~4.5 radians ~= 260 degrees)
    if total_span < 4.0:
        return (False, None, None)

    # ensure angle progression is fairly smooth (not oscillating back-and-forth)
    ang_diffs = np.diff(ang_unwrapped)
    signs = np.sign(ang_diffs)
    # count sign changes (should be small for circular motion)
    sign_changes = np.sum(signs[1:] * signs[:-1] < 0)
    if sign_changes > max(2, len(ang_diffs) // 4):
        return (False, None, None)

    # check path length vs circumference estimate (rough sanity check)
    diffs = np.linalg.norm(np.diff(pts2, axis=0), axis=1)
    path_len = diffs.sum()
    circ_est = 2 * np.pi * mean_r
    if path_len < 0.6 * circ_est:  # must have traversed a decent portion of circumference
        return (False, None, None)

    return (True, c, mean_r)


def detect_vertical_swipe(path_deque, min_points=6):
    """Detect upward (speed up) or downward (slow down) swipe from wrist path.
    Returns 'UP' or 'DOWN' or None.
    """
    if len(path_deque) < min_points:
        return None
    pts = np.array(list(path_deque))
    if pts.shape[1] < 2:
        return None
    xs = pts[:, 0]
    ys = pts[:, 1]

    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]

    # require substantial vertical motion relative to horizontal motion
    if abs(dy) < 0.06:
        return None
    horiz_fraction = abs(dx) / (abs(dy) + 1e-9)
    if horiz_fraction > 0.45:  # too much horizontal motion
        return None

    # quick linearity check: fit y vs sample index and check residuals
    idx = np.arange(len(ys))
    coef = np.polyfit(idx, ys, 1)
    fitted = np.polyval(coef, idx)
    resid = np.sqrt(np.mean((fitted - ys) ** 2))
    dyn_range = ys.max() - ys.min()
    if dyn_range < 1e-6:
        return None
    resid_frac = resid / (dyn_range + 1e-9)
    if resid_frac > 0.35:  # too noisy / not a straight swipe
        return None

    # negative dy = upward (coordinate system: y increases downward)
    if dy < -0.08:
        return 'UP'
    if dy > 0.08:
        return 'DOWN'
    return None


def detect_forward_poke(path_deque, min_points=6):
    """Detect forward poke (hand moves toward camera) using wrist z values.
    Returns True if poke detected.
    """
    if len(path_deque) < min_points:
        return False
    pts = np.array(list(path_deque))
    if pts.shape[1] < 3:
        return False
    zs = pts[:, 2]

    # require a noticeable approach toward camera (z decreases)
    dz = zs[-1] - zs[0]
    if dz >= -0.04:
        return False

    # require that majority of samples are moving toward the camera (monotonicity)
    decreases = np.sum(np.diff(zs) < 0)
    if decreases < (0.6 * (len(zs) - 1)):
        return False

    return True

# ------------ Voice assistant using Vosk + sounddevice ------------
class VoiceAssistant(threading.Thread):
    # CORRECTED: Initializer name fixed to __init__
    def __init__(self, command_queue): 
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.speech_queue = queue.Queue()
        self.speech_thread = None
        self.running = False
        self.enabled = False
        self.model = None
        self.rec = None
        
        # Initialize pyttsx3 engine once
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            self.tts_engine_initialized = True
        except Exception as e:
            print(f"Warning: pyttsx3 engine failed to initialize: {e}. Voice feedback disabled.")
            self.tts_engine_initialized = False
            
        # If the Vosk model is missing, disable voice features
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"Warning: Vosk model not found at {VOSK_MODEL_PATH}. Voice recognition disabled.")
            self.enabled = False
            if self.tts_engine_initialized:
                 self.start_speech_thread()
            return

        self.enabled = True
        self.model = Model(VOSK_MODEL_PATH)
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.running = True
        self.last_spoken = time.time() - 10
        # Start speech processing thread
        if self.tts_engine_initialized:
            self.start_speech_thread()

    def start_speech_thread(self):
        def speech_worker():
            # Check if engine is ready before starting the worker
            if not self.tts_engine_initialized:
                return
            while self.running:
                try:
                    # Get next text to speak with a timeout
                    try:
                        text = self.speech_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    try:
                        # Use the pre-initialized engine
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                        
                    except Exception as e:
                        print(f"Speech synthesis error: {e}")
                        
                except Exception as e:
                    print(f"Speech worker error: {e}")
                    
        self.speech_thread = threading.Thread(target=speech_worker, daemon=True)
        self.speech_thread.start()
    
    def speak(self, text):
        if self.tts_engine_initialized:
            # Just queue the text, the speech thread will handle it
            self.speech_queue.put(text)

    def audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        if status:
            # can log status
            pass
        # indata is already a buffer when dtype='int16', convert directly to bytes
        data = bytes(indata)
        if self.rec.AcceptWaveform(data):
            res = self.rec.Result()
            # The result is a JSON string like {"text":"..."}
            try:
                # Removed json import from here, now at top level
                text = json.loads(res).get("text", "").strip().lower()
                if text:
                    # fuzzy match
                    best = difflib.get_close_matches(text, VOICE_PHRASES.keys(), n=1, cutoff=FUZZY_CUTOFF)
                    if best:
                        cmd = VOICE_PHRASES[best[0]]
                        self.command_queue.put(("VOICE", cmd, text))
                        # speak feedback in background
                        threading.Thread(target=self.speak, args=(f"Command {cmd}",), daemon=True).start()
            except Exception:
                pass
        else:
            # partial = self.rec.PartialResult(); ignore partials to avoid noise
            pass

    def run(self):
        # If voice is disabled (missing model), exit thread quietly.
        if not getattr(self, 'enabled', True):
            return

        # Start input stream
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize = 8000, dtype='int16',
                                   channels=1, callback=self.audio_callback):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            print("Voice thread error:", e)

    def stop(self):
        self.running = False
        # Stop the TTS engine
        if self.tts_engine_initialized:
            self.tts_engine.stop()
        # Clear any pending speech
        try:
            while True:
                self.speech_queue.get_nowait()
        except queue.Empty:
            pass

# ------------ Main program ------------
def main():
    # queue for commands from voice
    cmd_q = queue.Queue()
    va = VoiceAssistant(cmd_q)
    va.start()

    # keep last commands to avoid repeats
    last_cmd_time = {}
    combined_cmd_queue = queue.Queue()
    # global last-gesture timestamp to enforce GLOBAL_GESTURE_COOLDOWN between any gestures
    last_gesture_time = 0.0

    def emit_gesture(cmd, meta):
        """Helper to enqueue a gesture while respecting per-command and global cooldowns."""
        nonlocal last_gesture_time
        now = time.time()
        # global cooldown
        if now - last_gesture_time < GLOBAL_GESTURE_COOLDOWN:
            if DEBUG:
                print(f"Gesture '{cmd}' suppressed by global cooldown ({now - last_gesture_time:.2f}s)")
            return False
        # per-command debounce
        key = f"GESTURE_{cmd}"
        last_time = last_cmd_time.get(key, 0)
        if now - last_time <= COMMAND_COOLDOWN:
            if DEBUG:
                print(f"Gesture '{cmd}' suppressed by per-command debounce ({now - last_time:.2f}s)")
            return False
        combined_cmd_queue.put(("GESTURE", cmd, meta))
        last_cmd_time[key] = now
        last_gesture_time = now
        return True

    # Smoothing buffers per hand index
    hand_buffers = [collections.deque(maxlen=SMOOTH_WINDOW) for _ in range(MAX_HANDS)]
    # Track recent wrist positions per hand for motion-based gestures (circle/swipes/poke)
    # store (x,y,z) normalized coords
    hand_paths = [collections.deque(maxlen=30) for _ in range(MAX_HANDS)]
    # Circle overlay is unused (circle gesture is recognized via thumb+index)
    # Last help overlay: (text_lines, expiry_timestamp)
    last_help = None
    # Short-lived action overlay (e.g., Faster/Slower/Start)
    last_action = None
    # Last help overlay: (text_lines, expiry_timestamp)
    last_help = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                        min_detection_confidence=0.75,
                        min_tracking_confidence=0.75) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_texts = []
            # collect per-hand gesture candidates so we can detect both-hands-open -> HELP
            per_hand_candidates = []
            # Reset buffers if no hands
            if not results.multi_hand_landmarks:
                for b in hand_buffers:
                    b.clear()
                for p in hand_paths:
                    p.clear()

            # Process hands if present
            if results.multi_hand_landmarks:
                # multi_hand_landmarks and multi_handedness correspond by index
                for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks,
                                                                      results.multi_handedness)):
                    # safety: only process up to MAX_HANDS to match buffer sizes
                    if idx >= MAX_HANDS:
                        break
                    # collect landmarks into numpy array (21,3)
                    lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                    hand_buffers[idx].append(lm)

                    # append wrist position (normalized coords) for motion gestures
                    wrist = lm[0, :3]  # x,y,z normalized
                    hand_paths[idx].append(wrist.tolist())

                    # Circle motion detection disabled; circle is now recognized
                    # when Thumb + Index fingers are shown together (see classify_gesture_from_fingers).

                    # detect vertical swipe for speed control
                    vs = detect_vertical_swipe(hand_paths[idx])
                    if vs == 'UP':
                        last_time = last_cmd_time.get(f"GESTURE_SPEED_UP", 0)
                        now = time.time()
                        # Use emit_gesture for consistent debouncing/cooldown
                        if emit_gesture("SPEED_UP", f"hand{idx}_swipe"): 
                            last_action = ("Faster (Swipe)", now + GESTURE_DISPLAY_DURATION)
                            hand_paths[idx].clear()
                    elif vs == 'DOWN':
                        last_time = last_cmd_time.get(f"GESTURE_SLOW_DOWN", 0)
                        now = time.time()
                        if emit_gesture("SLOW_DOWN", f"hand{idx}_swipe"):
                            last_action = ("Slower (Swipe)", now + GESTURE_DISPLAY_DURATION)
                            hand_paths[idx].clear()

                    # detect forward poke for START
                    if detect_forward_poke(hand_paths[idx]):
                        last_time = last_cmd_time.get(f"GESTURE_START", 0)
                        now = time.time()
                        if emit_gesture("START", f"hand{idx}_poke"):
                            last_action = ("Start (Poke)", now + GESTURE_DISPLAY_DURATION)
                            hand_paths[idx].clear()

                    # draw landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if len(hand_buffers[idx]) >= 2:
                        avg_lm = avg_landmarks(hand_buffers[idx])
                        norm = normalize_landmarks(avg_lm)
                        hand_label = handedness.classification[0].label  # 'Right' or 'Left'
                        fingers = fingers_up_from_norm(norm, handedness_label=hand_label)
                        cmd = classify_gesture_from_fingers(fingers)
                        if cmd:
                            score = round(sum(fingers.values())/5.0, 2)
                            gesture_texts.append((cmd, idx, score))
                            # collect candidate for post-processing (to detect both-hands HELP)
                            per_hand_candidates.append({
                                "idx": idx,
                                "cmd": cmd,
                                "fingers_total": int(sum(fingers.values())),
                                "score": score
                            })

                # end for each hand
                # After processing all hands: if at least two hands show all 5 fingers, emit HELP once
                if len(per_hand_candidates) >= 2:
                    full_count = sum(1 for c in per_hand_candidates if c["fingers_total"] == 5)
                    if full_count >= 2:
                            # require both hands to be fairly stationary for HELP to avoid false positives
                            five_idxs = [c['idx'] for c in per_hand_candidates if c['fingers_total'] == 5]
                            # take first two five-finger hands (if more) for stability check
                            five_pair = five_idxs[:2]
                            both_stable = True
                            stability_details = []
                            for hidx in five_pair:
                                path = hand_paths[hidx]
                                pts = np.array(list(path)) if len(path) > 0 else np.zeros((0,3))
                                if pts.shape[0] < HELP_STABLE_FRAMES:
                                    both_stable = False
                                    stability_details.append((hidx, False, 1.0))
                                    continue
                                # compute movement statistics: mean velocity magnitude and z variation
                                diffs = np.diff(pts, axis=0)
                                vel = np.linalg.norm(diffs[:, :2], axis=1) if diffs.shape[0] > 0 else np.array([0.0])
                                mean_vel = float(np.mean(vel))
                                max_vel = float(np.max(vel)) if vel.size > 0 else 0.0
                                # z stability: avoid big forward/back movements
                                z_std = float(np.std(pts[:, 2])) if pts.shape[1] >= 3 else 0.0
                                # Adjusted stable_xy check to match original intent based on max velocity
                                stable_xy = (max_vel <= (HELP_MOVEMENT_THRESHOLD * 8)) and (mean_vel <= (HELP_MOVEMENT_THRESHOLD * 4))
                                stable_z = z_std <= 0.02
                                # mv_metric calculation retained for debug, though primarily unused in logic
                                mv_metric = float(np.sqrt(np.ptp(pts[:,0])**2 + np.ptp(pts[:,1])**2)) if pts.shape[1] >= 2 else 0.0 # Corrected **2 and removed *2 factor
                                stability_details.append((hidx, stable_xy and stable_z, mv_metric, mean_vel, z_std))
                                if not (stable_xy and stable_z):
                                    both_stable = False

                            if DEBUG:
                                print("HELP stability details:", stability_details)

                            last_time = last_cmd_time.get("GESTURE_HELP", 0)
                            now = time.time()
                            if both_stable:
                                # emit_gesture already includes the COMMAND_COOLDOWN check
                                if emit_gesture("HELP", "both_hands_stable"):
                                    # Clear paths only on success to make sure the stability window is full
                                    for hidx in five_pair:
                                        hand_paths[hidx].clear()
                            else:
                                # hands not stable enough: emit per-hand commands instead
                                # This block correctly prevents an unstably-held HELP from blocking other commands
                                for c in per_hand_candidates:
                                    # Only process single-hand gestures if the two-hand HELP wasn't stable
                                    # The emit_gesture function handles the debounce
                                    emit_gesture(c['cmd'], f"hand{c['idx']}")
                    else:
                        # fewer than 2 hands showing 5 fingers: emit any candidate normally
                        for c in per_hand_candidates:
                            emit_gesture(c['cmd'], f"hand{c['idx']}")
                else:
                    # fewer than 2 candidates: emit any candidate normally
                    for c in per_hand_candidates:
                        emit_gesture(c['cmd'], f"hand{c['idx']}")

            else:
                # No hands: show "No hand" overlay
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Draw gesture texts
            y0 = 40
            for (cmd, idx, score) in gesture_texts:
                cv2.putText(frame, f"H{idx}: {cmd} ({score})", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                y0 += 30

            # Debug overlay: per-hand finger totals and recent wrist movement
            if DEBUG:
                bx = w - 260
                by = 40
                cv2.rectangle(frame, (bx-6, by-18), (bx + 250, by + 26 + 30 * MAX_HANDS), (30,30,30), -1)
                for i in range(MAX_HANDS):
                    cand = next((c for c in per_hand_candidates if c['idx'] == i), None)
                    fingers_total = cand['fingers_total'] if cand else 0
                    # compute movement
                    mv = 0.0
                    pts = np.array(list(hand_paths[i])) if len(hand_paths[i]) > 0 else np.zeros((0,3))
                    if pts.shape[0] >= 2 and pts.shape[1] >= 2:
                        mv = float(np.sqrt(np.ptp(pts[:,0])**2 + np.ptp(pts[:,1])**2)) # Corrected **2 and removed *2 factor
                    line = f"H{i}: fingers={fingers_total} mv={mv:.3f}"
                    cv2.putText(frame, line, (bx, by + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            # Draw help overlay if present
            if last_help is not None:
                lines, expiry = last_help
                if time.time() < expiry:
                    # draw background box
                    box_w = 360
                    box_h = 20 + 22 * len(lines)
                    bx = 10
                    by = 10
                    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (50,50,50), -1)
                    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0,255,255), 1)
                    y = by + 20
                    for line in lines:
                        cv2.putText(frame, line, (bx + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        y += 22
                else:
                    last_help = None

            # Draw short action overlay (Faster/Slower/Start)
            if last_action is not None:
                text, expiry = last_action
                if time.time() < expiry:
                    # draw a small badge near bottom-right
                    bx = w - 220
                    by = h - 80
                    cv2.rectangle(frame, (bx, by), (bx + 200, by + 50), (40,40,40), -1)
                    cv2.rectangle(frame, (bx, by), (bx + 200, by + 50), (0,200,255), 1)
                    cv2.putText(frame, text, (bx + 18, by + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                else:
                    last_action = None

            # Check voice commands from queue
            try:
                while True:
                    src, vcmd, raw_text = cmd_q.get_nowait()
                    # Debounce voice commands too
                    last_time = last_cmd_time.get(f"VOICE_{vcmd}", 0)
                    now = time.time()
                    if now - last_time > COMMAND_COOLDOWN:
                        combined_cmd_queue.put(("VOICE", vcmd, raw_text))
                        last_cmd_time[f"VOICE_{vcmd}"] = now
            except queue.Empty:
                pass

            # Process combined commands (either from gesture or voice)
            try:
                # process multiple if present, but limit to avoid flooding
                processed = []
                while not combined_cmd_queue.empty() and len(processed) < 3:
                    src, cmd, meta = combined_cmd_queue.get_nowait()
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {src} -> {cmd} ({meta})")
                    # Special handling for HELP to show an overlay
                    if cmd == "HELP":
                        help_text = "Available commands: forward, back, left, right, stop, faster, slower, start, help, circle"
                        # split into lines for overlay
                        lines = [l.strip() for l in help_text.split(',')]
                        last_help = (lines, time.time() + GESTURE_DISPLAY_DURATION)  # show for configured duration
                        va.speak(help_text)
                    else:
                        if src == "GESTURE":
                            # Use meta for more descriptive speech feedback for gestures
                            va.speak(f"Detected {cmd}")
                        else:
                            # Use raw text for voice commands if available
                            va.speak(f"Voice command: {meta}") 
                    processed.append((src, cmd, meta))
                # show last processed command on screen
                if processed:
                    last = processed[-1]
                    cv2.putText(frame, f"{last[0]}: {last[1]}", (10, h-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
            except queue.Empty:
                pass

            cv2.imshow("Gesture + Vosk Voice Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    va.stop()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()