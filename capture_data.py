import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands()

draw = mp.solutions.drawing_utils

while True:

    success, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            draw.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS
            )

            for id, lm in enumerate(hand.landmark):
                print(id, lm.x, lm.y)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()