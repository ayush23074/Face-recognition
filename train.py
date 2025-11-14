import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time

# ----------------- CONFIG -----------------
# Set this to your project root folder
base_dir = r"D:\STUDENT-ATTENDENCE-USING-FACE-RECOGNITION-master"

training_dir = os.path.join(base_dir, "Project files", "TrainingImage")
student_details_file = os.path.join(base_dir, "Project files", "StudentDetails", "StudentDetails.csv")
haarcascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
recognizer_file = os.path.join(base_dir, "face-trainner.yml")
images_unknown_dir = os.path.join(base_dir, "ImagesUnknown")
attendance_dir = os.path.join(base_dir, "Attendance")

# ensure directories exist
os.makedirs(training_dir, exist_ok=True)
os.makedirs(os.path.dirname(student_details_file), exist_ok=True)
os.makedirs(images_unknown_dir, exist_ok=True)
os.makedirs(attendance_dir, exist_ok=True)

# ----------------- GUI -----------------
window = tk.Tk()
window.title("Face_Recognition_for_Attendance")
window.configure(background='black')
window.attributes('-fullscreen', True)

# background image (optional — will fail if path wrong)
try:
    canvas = tk.Canvas(window, width=1500, height=900)
    canvas.pack()
    bg_img = ImageTk.PhotoImage(Image.open(os.path.join(base_dir, "p1.jpg")))
    canvas.create_image(0, 0, anchor='nw', image=bg_img)
except Exception:
    # if image not found just continue
    pass

border = tk.Label(window,
                  text="WELCOME TO SMART ATTENDANCE SYSTEM :: ENTER YOUR DETAILS AND CAPTURE PICTURES AND PRESS 'TRAIN IMAGE' :: EXISTING USER: Click on 'TRACK IMAGE' after tracking Press 'Q'",
                  width=200, height=1, bg='Red', fg="yellow", font=('Helvetica', 10, ' bold '), relief="raised")
border.place(x=0, y=0)

title_label = tk.Label(window, text="PU TECH - FACE RECOGNITION SYSTEM", bg="white", fg="black", width=50, height=3,
                       font=('times', 35, ' bold'))
title_label.place(x=70, y=22)

lbl_id = tk.Label(window, text="Enter ID", width=20, height=2, fg="black", bg="white", font=('times', 15, ' bold '))
lbl_id.place(x=400, y=200)
entry_id = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
entry_id.place(x=700, y=215)

lbl_name = tk.Label(window, text="Enter Name", width=20, fg="black", bg="white", height=2, font=('times', 15, ' bold '))
lbl_name.place(x=400, y=300)
entry_name = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
entry_name.place(x=700, y=315)

lbl_notify = tk.Label(window, text="Notification : ", width=20, fg="black", bg="white", height=2,
                      font=('times', 15, ' bold underline '))
lbl_notify.place(x=400, y=400)
notification_label = tk.Label(window, text="", bg="white", fg="black", width=30, height=2,
                              activebackground="yellow", font=('times', 15, ' bold '))
notification_label.place(x=700, y=400)

lbl_att = tk.Label(window, text="Attendance : ", width=20, fg="white", bg="Blue", height=2,
                   font=('times', 15, ' bold  underline'))
lbl_att.place(x=400, y=720)
attendance_label = tk.Label(window, text="", fg="white", bg="Blue", width=30, height=2,
                            activeforeground="green", font=('times', 15, ' bold '))
attendance_label.place(x=700, y=720)

# ----------------- Helpers -----------------
def clear_id():
    entry_id.delete(0, 'end')
    notification_label.configure(text="")

def clear_name():
    entry_name.delete(0, 'end')
    notification_label.configure(text="")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# ----------------- Core functions -----------------
def TakeImages():
    id_text = entry_id.get().strip()
    name = entry_name.get().strip()

    if not id_text or not name:
        notification_label.configure(text="Please enter both ID and Name")
        return

    if not is_number(id_text):
        notification_label.configure(text="Enter numeric ID")
        return

    if not name.isalpha():
        notification_label.configure(text="Enter alphabetical Name (no spaces/special chars)")
        return

    Id = int(id_text)
    name = name.title()  # normalize name
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        notification_label.configure(text="Unable to open camera")
        return

    detector = cv2.CascadeClassifier(haarcascade_file)
    sampleNum = 0
    max_samples = 60

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1
            filename = os.path.join(training_dir, f"{name}.{Id}.{sampleNum}.jpg")
            cv2.imwrite(filename, gray[y:y + h, x:x + w])
            cv2.imshow('frame', img)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        if sampleNum >= max_samples:
            break

    cam.release()
    cv2.destroyAllWindows()

    # append student details (avoid duplicate header)
    row = [Id, name]
    try:
        with open(student_details_file, 'a', newline='', encoding='utf-8') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
    except Exception as e:
        notification_label.configure(text=f"Saved images but failed to write CSV: {e}")
        return

    notification_label.configure(text=f"Images saved for ID: {Id} Name: {name}")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')  # grayscale
        imageNp = np.array(pilImage, 'uint8')
        # filename format: name.Id.sample.jpg -> Id is second component
        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
        except Exception:
            # skip files that don't match naming pattern
            continue
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def TrainImages():
    # create recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        notification_label.configure(text="cv2.face not available in this OpenCV build")
        return

    faces, Ids = getImagesAndLabels(training_dir)
    if len(faces) == 0:
        notification_label.configure(text="No training images found")
        return
    recognizer.train(faces, np.array(Ids))
    recognizer.write(recognizer_file)  # save trained model
    notification_label.configure(text="Image training complete and saved")

def TrackImages():
    # check recognizer file
    if not os.path.exists(recognizer_file):
        messagebox.showerror("Error", "Recognizer file not found. Please Train Images first.")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(recognizer_file)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load recognizer: {e}")
        return

    faceCascade = cv2.CascadeClassifier(haarcascade_file)

    # read students csv (if missing, create empty df)
    if os.path.exists(student_details_file):
        df = pd.read_csv(student_details_file, header=None, names=['Id', 'Name'], dtype={'Id': int, 'Name': str})
    else:
        df = pd.DataFrame(columns=['Id', 'Name'])

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        notification_label.configure(text="Unable to open camera")
        return

    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id_pred, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                # look up name
                name_arr = df.loc[df['Id'] == Id_pred]['Name'].values
                name = name_arr[0] if len(name_arr) > 0 else "Unknown"
                attendance.loc[len(attendance)] = [Id_pred, name, date, timeStamp]
                display_text = f"{Id_pred} - {name}"
            else:
                display_text = "Mismatched"

            # save unknown faces with higher confidence (worse match)
            if conf > 75:
                noOfFile = len([f for f in os.listdir(images_unknown_dir) if f.lower().endswith(('.jpg', '.png'))]) + 1
                cv2.imwrite(os.path.join(images_unknown_dir, f"Image{noOfFile}.jpg"), im[y:y + h, x:x + w])

            cv2.putText(im, display_text, (x, y + h if y + h < im.shape[0] else im.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break

    # save attendance
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')
    fileName = os.path.join(attendance_dir, f"Attendance_{date}_{timeStamp}.csv")
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()

    attendance_label.configure(text=f"Saved: {os.path.basename(fileName)}")
    notification_label.configure(text=f"{len(attendance)} attendance records saved")

# ----------------- Buttons -----------------
btn_clear_id = tk.Button(window, text="Clear", command=clear_id, fg="black", bg="white", width=20, height=2,
                         activebackground="Red", font=('times', 15, ' bold '))
btn_clear_id.place(x=950, y=200)

btn_clear_name = tk.Button(window, text="Clear", command=clear_name, fg="black", bg="white", width=20, height=2,
                           activebackground="Red", font=('times', 15, ' bold '))
btn_clear_name.place(x=950, y=300)

btn_take_images = tk.Button(window, text="Take Images", command=TakeImages, fg="black", bg="white", width=20, height=3,
                            activebackground="Red", font=('times', 15, ' bold '))
btn_take_images.place(x=200, y=500)

btn_student_data = tk.Button(window, text="Student Data (Open CSV)", fg="black", bg="white", width=20, height=3,
                             activebackground="Red", font=('times', 15, ' bold '),
                             command=lambda: os.startfile(os.path.dirname(student_details_file)))
btn_student_data.place(x=200, y=600)

btn_train = tk.Button(window, text="Train Images", command=TrainImages, fg="black", bg="white", width=20, height=3,
                      activebackground="Red", font=('times', 15, ' bold '))
btn_train.place(x=500, y=500)

btn_attendance_files = tk.Button(window, text="Attendance Files (Open Dir)", fg="black", bg="white", width=20, height=3,
                                 activebackground="Red", font=('times', 15, ' bold '),
                                 command=lambda: os.startfile(attendance_dir))
btn_attendance_files.place(x=500, y=600)

btn_track = tk.Button(window, text="Track Images", command=TrackImages, fg="black", bg="white", width=20, height=3,
                      activebackground="Red", font=('times', 15, ' bold '))
btn_track.place(x=800, y=500)

btn_analytics = tk.Button(window, text="Analytics (Open Attendance Dir)", fg="black", bg="white", width=20, height=3,
                          activebackground="Red", font=('times', 15, ' bold '),
                          command=lambda: os.startfile(attendance_dir))
btn_analytics.place(x=800, y=600)

copyWrite = tk.Label(window, text="Created By \n (Ayush)", bg="Red", fg="black", width=100, height=2,
                     activebackground="Blue", font=('times', 12, ' bold '))
copyWrite.place(x=300, y=800)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=20, height=3,
                       activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)

window.mainloop()
