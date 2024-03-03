import cv2
import face_recognition


def detect_age(frame):
    # Find faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Load pre-trained age estimation model
    age_model = cv2.dnn.readNetFromCaffe(
        "deploy_age.prototxt",
        "age_net.caffemodel"
    )

    ages = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Crop face from the frame
        face = frame[top:bottom, left:right]

        # Convert the face to a blob for the age model
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)

        # Make predictions using the age model
        age_model.setInput(blob)
        age_preds = age_model.forward()

        # Find the age with the maximum probability
        age = int(age_preds[0].argmax())

        ages.append(age)

        # Draw rectangle and display the age on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {age} years', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


if __name__ == "__main__":
    # Open the video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frames from the camera
        ret, frame = cap.read()

        # Detect age and display the result on the frame
        result_frame = detect_age(frame)

        # Display the result frame
        cv2.imshow('Age Detection', result_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
