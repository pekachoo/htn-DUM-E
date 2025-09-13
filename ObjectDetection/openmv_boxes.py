import sensor, image, time

# -------------------- Setup --------------------
sensor.reset()
sensor.set_pixformat(sensor.RGB565)   # Color mode
sensor.set_framesize(sensor.QVGA)     # 320x240
sensor.skip_frames(time=2000)
clock = time.clock()

# -------------------- Parameters --------------------
MIN_AREA = 500       # ignore tiny blobs
MAX_AREA = 30000     # ignore huge blobs
MAX_MISSES = 3       # how many frames to keep missing objects
FRAME_SKIP = 2       # only run detection every N frames

# -------------------- Object Tracking --------------------
object_memory = []  # [{'rect':(x,y,w,h), 'misses':0}]

# -------------------- Helpers --------------------
def merge_boxes(boxes):
    merged = []
    for b in boxes:
        x, y, w, h = b
        added = False
        for i, m in enumerate(merged):
            mx, my, mw, mh = m
            # Check overlap
            if not (x > mx+mw or mx > x+w or y > my+mh or my > y+h):
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x+w, mx+mw) - nx
                nh = max(y+h, my+mh) - ny
                merged[i] = (nx, ny, nw, nh)
                added = True
                break
        if not added:
            merged.append(b)
    return merged

def boxes_overlap(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 > x2+w2 or x2 > x1+w1 or y1 > y2+h2 or y2 > y1+h1)

# -------------------- Main Loop --------------------
frame_count = 0
while(True):
    clock.tick()
    img = sensor.snapshot()
    frame_count += 1

    # Only process detection every FRAME_SKIP frames
    if frame_count % FRAME_SKIP == 0:
        new_boxes = []

        # --- 1. Blob detection (LAB thresholds wide to capture most objects) ---
        blobs = img.find_blobs([(0, 100, -50, 50, -50, 50)],
                               pixels_threshold=200, area_threshold=MIN_AREA, merge=True)
        for b in blobs:
            if MIN_AREA < b.area() < MAX_AREA:
                new_boxes.append(b.rect())

        # --- 2. Rectangles (contrast-based) ---
        rects = img.find_rects(threshold=10000)
        for r in rects:
            x, y, w, h = r.rect()
            if MIN_AREA < w*h < MAX_AREA:
                new_boxes.append(r.rect())

        # --- Merge overlapping boxes ---
        merged_boxes = merge_boxes(new_boxes)

        # --- Update object memory with tracking ---
        updated_memory = []
        for mb in merged_boxes:
            matched = False
            for obj in object_memory:
                if boxes_overlap(mb, obj['rect']):
                    # Update existing object
                    obj['rect'] = mb
                    obj['misses'] = 0
                    updated_memory.append(obj)
                    matched = True
                    break
            if not matched:
                # New object
                updated_memory.append({'rect': mb, 'misses': 0})

        # Increment misses for objects not seen this frame
        for obj in object_memory:
            if obj not in updated_memory:
                obj['misses'] += 1
                if obj['misses'] < MAX_MISSES:
                    updated_memory.append(obj)

        object_memory = updated_memory

    # --- Draw tracked boxes every frame ---
    for obj in object_memory:
        img.draw_rectangle(obj['rect'], color=(255,0,0), thickness=2)

    print("Objects:", len(object_memory), "FPS:", clock.fps())
