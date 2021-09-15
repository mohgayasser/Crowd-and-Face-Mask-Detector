from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
from DetectMask import RunModel

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
    picture=img;
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

Label(root, text="Choose Image from your files ").grid(row=1)
# Create a button and place it into the window using grid layout
btn = Button(root, text='open image ', command=lambda m="open": which_button(m)).grid(row=1, column=2)


def which_button(button_press):
    if button_press == "open":
        InputImage=open_img()

        Output=RunModel(InputImage)
        fileName="Output.jpg";
        cv2.imwrite(fileName, Output)
        show_finalimage(fileName)

root.mainloop()
