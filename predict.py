import cv2
import numpy as np
import mediapipe as mp
import joblib
import pyttsx3
import time

model = joblib.load("model.pkl")

engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands()

draw = mp.solutions.drawing_utils

sentence = ""

last_letter = ""
last_added_time = 0

while True:

    success, frame = cap.read()
    black = np.zeros_like(frame)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            features = []

            for lm in hand.landmark:
                draw.draw_landmarks(
                    black,
                    hand,
                    mp.solutions.hands.HAND_CONNECTIONS
                )


                features.append(lm.x)
                features.append(lm.y)

            prediction = model.predict(
                [features]
            )[0]

            probabilities = model.predict_proba(
                [features]
            )

            confidence = (
                max(probabilities[0]) * 100
            )

            current_time = time.time()

            if (
                prediction != last_letter
                and confidence > 90
                and current_time - last_added_time > 1
            ):

                sentence += prediction

                last_letter = prediction

                last_added_time = current_time
            cv2.putText(
                black,
                "SIGN LANGUAGE DETECTOR",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                black,
                f"Prediction: {prediction}",
                (20,55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.putText(
                 black,
                f"Confidence: {confidence:.1f}%",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2
            )
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (300, 120),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.5,
            frame,
            0.5,
            0,
            frame
        )
    cv2.putText(
        black,
        f"Text: {sentence}",
        (20,105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Sign Language Detector",
        black
    )

    key = cv2.waitKey(1)

    if key == ord("s"):

        if sentence:

            engine.say(sentence)
            engine.runAndWait()

    elif key == ord("c"):

        sentence = ""

    elif key == ord("q"):

        break

cap.release()
cv2.destroyAllWindows()