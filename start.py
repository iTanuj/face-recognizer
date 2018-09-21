import tkinter
import os


def train():
    os.system("python E:\\Documents\\Projects\\College\\FaceRecognition\\faceTrainer.py")

def recog_photos():
    os.system("python E:\\Documents\\Projects\\College\\FaceRecognition\\faceRecognizer.py")

def recog_cam():
    os.system("python E:\\Documents\\Projects\\College\\FaceRecognition\\faceRecogLive.py")

window = tkinter.Tk()

image1 = tkinter.PhotoImage(file="E:\Documents\Projects\College\FaceRecognition\\back.png")
window.image = image1
w = image1.width()
h = image1.height()


panel1 = tkinter.Label(window, image=image1, bg = 'white')
panel1.pack(side='top', fill='both', expand='yes')
window.geometry("%dx%d+250+250" % (w, h))
window.wm_attributes("-topmost", True)
window.wm_attributes("-transparentcolor", "white")
window.configure(background="#40C4FF")
window.title("Face Recognizer")
window.wm_iconbitmap(r'E:\Documents\Projects\College\FaceRecognition\icon.ico')

lblInst = tkinter.Label(panel1,text="Please choose your option:", fg="#e835f2", bg="#485d90",  font=("Roboto", 16))
#lblInst.grid(row=0,column=0, padx=(20,20),pady=(20,20))
lblInst.pack(padx=50,pady=(0,85))

btn = tkinter.Button(panel1, text="Train", fg="#e835f2", bg="#485d90",command=train)
#btn.grid(row=1,column=0,padx=(10,5))
btn.pack(padx=32,pady=(80,0),side='left')

btn = tkinter.Button(panel1, text="Recognize from images", fg="#e835f2", bg="#485d90",command=recog_photos)
#btn.grid(row=1,column=0,padx=(5,5))
btn.pack(padx=32,pady=(80,0),side='left')

btn = tkinter.Button(panel1, text="Recognize using cam", fg="#e835f2", bg="#485d90",command=recog_cam)
#btn.grid(row=1,column=0,padx=(5,10))
btn.pack(padx=32,pady=(80,0),side='left')
#panel1.image = image1
panel1.mainloop()
