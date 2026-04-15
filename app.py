from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_engine = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_arabic_font(size=36):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

font = load_arabic_font(36)

sentence = ""
temp_letter = ""
count = 0


def get_sign(lm):
    # f[0]=إبهام، f[1]=سبابة، f[2]=وسطى، f[3]=بنصر، f[4]=خنصر
    f = []
    f.append(1 if lm[4].x < lm[3].x else 0)
    tips = [8, 12, 16, 20]
    for tip in tips:
        f.append(1 if lm[tip].y < lm[tip-2].y else 0)

    # مسافة: كف مفتوح بالكامل للأمام (كل الأصابع مرفوعة عالياً)
    if f == [1, 1, 1, 1, 1] and lm[8].y < lm[5].y - 0.1 and lm[12].y < lm[9].y - 0.1:
        return "مسافة"

    # حذف حرف: إبهام للأسفل مع باقي الأصابع مغلقة
    if lm[4].y > lm[3].y + 0.06 and f[1:] == [0, 0, 0, 0]:
        return "حذف_حرف"

    # مسح الكل: قبضة مغلقة محكمة (كل الأصابع منطوية والإبهام فوقها)
    if f == [0, 0, 0, 0, 0] and lm[4].x > lm[3].x and lm[8].y > lm[5].y + 0.05 and lm[12].y > lm[9].y + 0.05:
        return "مسح_الكل"

    # ============================================================
    # الحروف — حسب الوصف بالضبط
    # ============================================================

    # أ — رفع السبابة للأعلى بشكل مستقيم فقط
    if f == [0, 1, 0, 0, 0] and lm[8].y < lm[6].y - 0.06:
        return "أ"

    # ب — كف مفتوح للأعلى مع الإبهام تحت راحة اليد (كل الأصابع مرفوعة والإبهام منخفض)
    if f == [0, 1, 1, 1, 1] and lm[4].y > lm[3].y:
        return "ب"

    # ت — سبابة ووسطى مرفوعتين كنقطتين فوق الكف
    if f == [0, 1, 1, 0, 0] and abs(lm[8].x - lm[12].x) < 0.04:
        return "ت"

    # ث — ثلاثة أصابع (سبابة، وسطى، بنصر) مرفوعة
    if f == [0, 1, 1, 1, 0]:
        return "ث"

    # ج — أصابع مقوسة نصف دائرة مع الإبهام في المنتصف
    if f[1:] == [0, 0, 0, 0] and lm[4].x > lm[3].x and lm[8].y > lm[6].y and lm[12].y > lm[10].y:
        dist = abs(lm[4].x - lm[9].x)
        if dist < 0.06:
            return "ج"

    # ح — نفس ج بدون الإبهام في المنتصف (اليد مفتوحة قليلاً)
    if f == [0, 0, 0, 0, 0] and lm[8].y > lm[6].y and lm[12].y > lm[10].y:
        dist = abs(lm[4].x - lm[9].x)
        if dist >= 0.06:
            return "ح"

    # خ — نفس ح مع السبابة فوق اليد (سبابة مرفوعة)
    if f == [0, 1, 0, 0, 0] and lm[8].y > lm[6].y and lm[12].y > lm[10].y:
        return "خ"

    # د — سبابة وإبهام يشكلان زاوية (مثل حرف د)
    if f == [1, 1, 0, 0, 0] and abs(lm[8].y - lm[4].y) < 0.05 and lm[12].y > lm[10].y:
        return "د"

    # ذ — نفس د مع رفع الوسطى للأعلى كنقطة
    if f == [1, 1, 1, 0, 0] and abs(lm[8].y - lm[4].y) < 0.05 and lm[8].y < lm[5].y:
        return "ذ"

    # ر — حني السبابة قليلاً (سبابة منحنية، باقي الأصابع مغلقة)
    if f == [0, 0, 0, 0, 0] and lm[8].y < lm[6].y and lm[8].y > lm[5].y:
        return "ر"

    # ز — نفس ر مع وسطى فوق السبابة
    if f == [0, 1, 0, 0, 0] and lm[8].y > lm[5].y and lm[12].y > lm[10].y:
        return "ز"

    # س — ثلاثة أصابع للأعلى (سبابة، وسطى، بنصر) متفرقة كأسنان
    if f == [0, 1, 1, 1, 0] and lm[8].y < lm[5].y - 0.05:
        return "س"

    # ش — نفس س مع تحريك الأصابع (إبهام مرفوع إضافي)
    if f == [1, 1, 1, 1, 0] and lm[8].y < lm[5].y - 0.05:
        return "ش"

    # ص — قبضة مع الإبهام أفقي أمام القبضة
    if f == [1, 0, 0, 0, 0] and lm[4].y > lm[3].y - 0.02 and lm[4].y < lm[3].y + 0.02:
        return "ص"

    # ض — نفس ص مع رفع السبابة للأعلى
    if f == [1, 1, 0, 0, 0] and lm[4].y > lm[3].y - 0.02 and lm[8].y < lm[6].y:
        return "ض"

    # ط — فرد الأصابع أفقياً مع رفع الإبهام للأعلى
    if f == [1, 1, 1, 1, 1] and lm[4].y < lm[3].y and lm[8].y > lm[5].y - 0.04:
        return "ط"

    # ظ — نفس ط مع رفع السبابة أكثر للأعلى
    if f == [1, 1, 1, 1, 1] and lm[4].y < lm[3].y and lm[8].y < lm[5].y - 0.04:
        return "ظ"

    # ع — سبابة ووسطى بشكل V مقلوب
    if f == [0, 1, 1, 0, 0] and abs(lm[8].x - lm[12].x) > 0.05:
        return "ع"

    # غ — نفس ع مع الإبهام فوقهما
    if f == [1, 1, 1, 0, 0] and abs(lm[8].x - lm[12].x) > 0.05:
        return "غ"

    # ف — طرف السبابة مع الإبهام دائرة صغيرة + باقي الأصابع مرفوعة
    if f == [0, 1, 1, 1, 1] and abs(lm[8].x - lm[4].x) < 0.04 and abs(lm[8].y - lm[4].y) < 0.04:
        return "ف"

    # ق — دائرة بالسبابة والإبهام مع ثني باقي الأصابع للأسفل
    if f == [0, 0, 0, 0, 0] and abs(lm[8].x - lm[4].x) < 0.04 and abs(lm[8].y - lm[4].y) < 0.04:
        return "ق"

    # ك — إبهام وسبابة بزاوية قائمة + باقي الأصابع منحنية للداخل
    if f == [1, 1, 0, 0, 0] and abs(lm[8].x - lm[4].x) > 0.07 and lm[12].y > lm[10].y:
        return "ك"

    # ل — إبهام وسبابة بزاوية L واسعة
    if f == [1, 1, 0, 0, 0] and lm[8].y < lm[5].y - 0.03 and lm[4].y > lm[3].y + 0.04:
        return "ل"

    # م — إغلاق قبضة اليد بالكامل
    if f == [1, 0, 0, 0, 0] and lm[4].y < lm[2].y and lm[8].y > lm[5].y + 0.05:
        return "م"

    # ن — فتح كف مع الإبهام نحو منتصف الكف
    if f == [0, 1, 1, 1, 1] and lm[4].y > lm[3].y + 0.03 and lm[4].x > lm[3].x:
        return "ن"

    # هـ — ضم جميع رؤوس الأصابع مع الإبهام بشكل دائري
    if f == [0, 0, 0, 0, 0] and lm[4].y > lm[2].y:
        dist = abs(lm[8].x - lm[4].x) + abs(lm[12].x - lm[4].x) + abs(lm[16].x - lm[4].x)
        if dist < 0.10:
            return "هـ"

    # و — ثني الأصابع بشكل دائري يشبه دائرة الواو
    if f == [0, 0, 0, 0, 0] and lm[8].y > lm[5].y and lm[4].y > lm[2].y:
        return "و"

    # ي — مد الخنصر فقط للأسفل
    if f == [0, 0, 0, 0, 1] and lm[20].y > lm[18].y + 0.03:
        return "ي"

    return ""


def draw_arabic_text(frame, text, position, font, color=(0, 0, 0)):
    if text.strip() == "":
        return frame
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def gen_frames():
    global sentence, temp_letter, count
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_engine.process(rgb)

        if res.multi_hand_landmarks:
            for hlms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)
                res_sign = get_sign(hlms.landmark)

                if res_sign != "":
                    if res_sign == temp_letter:
                        pass
                    else:
                        count += 1
                        if count >= 15:
                            if res_sign == "مسافة":
                                sentence += " "
                            elif res_sign == "حذف_حرف":
                                sentence = sentence[:-1]
                            elif res_sign == "مسح_الكل":
                                sentence = ""
                            else:
                                sentence += res_sign
                            temp_letter = res_sign
                            count = 0
                else:
                    temp_letter = ""
                    count = 0

        cv2.rectangle(frame, (0, 400), (640, 480), (255, 255, 255), -1)
        display_text = sentence if sentence else "..."
        frame = draw_arabic_text(frame, display_text, (20, 415), font, color=(0, 0, 0))

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_sentence')
def get_sentence():
    return jsonify({'sentence': sentence})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True, use_reloader=False)