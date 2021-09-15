from tkinter import *
import time
from tkinter import filedialog
import cv2
from PIL import ImageTk,Image
from TestOneSample.Counting import Count
from TestOneSample.MaskDetection import *

def WebcamCapture(root):
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            print(check)  # prints true as long as the webcam is running
            print(frame)  # prints matrix values of each framecd
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(filename='data/SH_partA/test/images/IMG_1.jpg', img=frame)
                webcam.release()
                img_new = cv2.imread('data/SH_partA/test/images/IMG_1.jpg', cv2.IMREAD_GRAYSCALE)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                img = Image.open('data/SH_partA/test/images/IMG_1.jpg')
                img = img.resize((450, 450), Image.ANTIALIAS)

                # PhotoImage class is used to add image to widgets, icons etc
                img = ImageTk.PhotoImage(img)

                # create a label
                panel = Label(root, image=img)

                # set the image as img
                panel.image = img
                Label(root, text="").grid(row=2)
                panel.grid(row=3)

                print("Processing image...")
                print("Image saved!")


                return img

            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            break


def show_finalimage(x):
  

    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down s3ampling filter
    img = img.resize((450, 450), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    Label(root, text="").grid(row=2)

    panel.grid(row=3, column=2)


def open_img():
    # Select the Imagename from a folder
    x = openfilename()

    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((450, 450), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    Label(root, text="").grid(row=2)

    panel.grid(row=3)
    return x


def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title='"pen')
    return filename


root = Tk()

# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
root.geometry("930x550")

# Allow Window to be resizable
root.resizable(width=True, height=True)

Label(root, text="Choose 1 from 2 Options to Upload The Image ").grid(row=1)
# Create a button and place it into the window using grid layout
btn = Button(root, text='open image ', command=lambda m="open": which_button(m)).grid(row=1, column=2)

btn2 = Button(root, text='Capture image', command=lambda m="capture": which_button(m)).grid(row=1, column=3)


def which_button(button_press):
    if button_press == "open":
        open_img()
        Count()
        maskde()
        show_finalimage("TestOneSample/result.png")
    else:
        WebcamCapture(root)
        Count()
        maskde()
        show_finalimage("TestOneSample/result.png")


root.mainloop()
