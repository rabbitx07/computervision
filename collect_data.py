import cv2
import mediapipe as mp
import csv

label = input("Enter sign label: ")

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands()

with open("dataset.csv", "a", newline="") as f:

    writer = csv.writer(f)

    while True:

        success, frame = cap.read()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            for hand in result.multi_hand_landmarks:

                row = []

                for lm in hand.landmark:
                    row.append(lm.x)
                    row.append(lm.y)


                row.append(label)

                writer.writerow(row)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()