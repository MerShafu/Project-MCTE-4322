
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
# from tkinter import font

from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

from NeuralPredictor import HeartAttackPredictor, PredictorData
#from nnpredictor import Predictor, PredictorData


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

predictor = HeartAttackPredictor()
#predictor = Predictor()

window = Tk()
window.title("Heart Disease Detector Normal NN")

window.geometry("700x500")
window.configure(bg = "#FFFFFF")

def getdata() :
    try:
        cholestrol = entry_choles.get("1.0",'end-1c').strip()
        bps = entry_bp.get("1.0",'end-1c').strip()
        physical_activity = entry_pa.get("1.0",'end-1c').strip()
        age = entry_age.get("1.0",'end-1c').strip()
        bmi = entry_bmi.get("1.0",'end-1c').strip()
        smoke = entry_smoke.get("1.0",'end-1c').strip()
        dia = entry_dia.get("1.0",'end-1c').strip()
        
        if not (cholestrol and bps and physical_activity and age and bmi and smoke and dia):
            raise ValueError("Please fill in all the required fields.")
        
        try:
            cholestrol = int(cholestrol)
            bps = float(bps)
            physical_activity = int(physical_activity)
            age = int(age)
            bmi = float(bmi)
        except ValueError:
            raise ValueError("\tInvalid Input")
        
        smoke = smoke.lower()
        dia = dia.lower()
        
        if smoke == "yes" or smoke == "no":
            smoking = (smoke == "yes")
        else:
            raise ValueError("          Invalid input for Smoking")

        if dia == "yes" or dia == "no":
            diabetes = (dia == "yes")
        else:
            raise ValueError("          Invalid input for Diabetes")

        message_label.config(text="")

        print("Cholesterol:", cholestrol, "BP:", bps, "PA:", physical_activity,
              "Age:", age, "BMI:", bmi, "Smoke:", smoke, "-", smoking, "Diab:", dia, "-", diabetes)
        
        # an object oriented approach to pass the data to the predictor
        predictor_data = PredictorData(cholestrol,bps,physical_activity,age,bmi,smoking,diabetes)
        print(predictor)
        predictor.load_predictor_data(predictor_data)

        result = predictor.predict()
        entry_result.config(state=NORMAL)
        entry_result.delete(0, END)
        entry_result.insert(END, str(result[0]))
        entry_result.config(state=DISABLED)

        round_value = round(result[1],3)
        prediction_value.config(text=round_value, fg="black")
        range.config(text="<.50: Not Likely | [.50,.75]: Possibly |>.75: Most Likely", fg="black")

    except ValueError as e:
        print(str(e))
        message_label.config(text=str(e), fg="red")

window.bind("<Return>", lambda event: getdata())

#The gui for the tkinter 

canvas = Canvas(
        window,
        bg = "#FFFFFF",
        height = 500,
        width = 700,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    700.0,
    121.0,
    fill="#385399",
    outline="")

canvas.create_text(
    139.0,
    164.0,
    anchor="nw",
    text="Cholesterol (mg/dl)",
    fill="#000000",
    font=("MontserratItalic ExtraBold", 12*-1 )
)

canvas.create_text(
    411.0,
    164.0,
    anchor="nw",
    text="BMI (kg/m" + "\u00b2)",
    fill="#000000",
    font=("MontserratItalic ExtraBold", 12 * -1)
)

canvas.create_text(
    411.0,
    224.0,
    anchor="nw",
    text="Smoking (Yes / No)",
    fill="#000000",
    font=("MontserratItalic ExtraBold", 12 * -1)
)

canvas.create_text(
    411.0,
    284.0,
    anchor="nw",
    text="Diabetes (Yes / No)",
    fill="#000000",
    font=("MontserratItalic ExtraBold", 12 * -1)
)

canvas.create_text(
    139.0,
    224.0,
    anchor="nw",
    text="Blood pressure (Hg-mm)",
    fill="#000000",
    font=("MontserratItalic ExtraBold", 12 * -1)
)

canvas.create_text(
    139.0,
    284.0,
    anchor="nw",
    text="Physical Activity ( Per Week) \n",
    fill="#000000",
    font=("MontserratItalic ExtraBold", 12 * -1)
)

canvas.create_text(
    139.0,
    344.0,
    anchor="nw",
    text="Age\n",
    fill="#000000",
    font=("MontserratItalic ExtraBold", 12 * -1)
)

canvas.create_text(
    324.0,
    408.0,
    anchor="nw",
    text="RESULT\n",
    fill="#000000",
    font=("MontserratItalic ExtraBold", 12 * -1)
)

canvas.create_text(
    139.0,
    37.0,
    anchor="nw",
    text="HEART DISEASE ",
    fill="#CCCCCC",
    font=("MontserratRoman ", 24*-1,"bold")
)

canvas.create_text(
    139.0,
    64.0,
    anchor="nw",
    text="DETECTOR",
    fill="#CCCCCC",
    font=("MontserratRoman ", 24*-1,"bold")
)

prediction_value = Label(
    window,
    text="",
    fg="red",
    bg="#FFFFFF",
    font=("MontserratItalic ExtraBold", 7)
)
prediction_value.place(x=335, y=465)

range = Label(
    window,
    text="",
    fg="red",
    bg="#FFFFFF",
    font=("MontserratItalic ExtraBold", 7)
)
range.place(x=230 ,y=480)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    448.0,
    69.0,
    image=image_image_1
)

message_label = Label(
    window,
    text="",
    fg="red",
    bg="#FFFFFF",
    font=("MontserratItalic ExtraBold", 12)
)
message_label.place(x=400, y=400)

def focus_next_widget(event):
    event.widget.tk_focusNext().focus()
    return "break"

def prevent_newline(event):
    event.widget.delete("end-1c")  # Delete the last character (newline)
    window.focus_set()
    getdata()
    return "break"  # Prevent the default behavior

entry_image_choles = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_choles = canvas.create_image(
    219.0,
    202.0,
    image=entry_image_choles
)
entry_choles = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_choles.place(
    x=154.0,
    y=192.0,
    width=130.0,
    height=23.0
)
entry_choles.bind("<Tab>", focus_next_widget)
entry_choles.bind("<Return>", prevent_newline)

entry_image_bp = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_bp = canvas.create_image(
    219.0,
    262.0,
    image=entry_image_bp
)
entry_bp = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_bp.place(
    x=154.0,
    y=252.0,
    width=130.0,
    height=23.0
)
entry_bp.bind("<Tab>", focus_next_widget)
entry_bp.bind("<Return>", prevent_newline)

entry_image_pa = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_pa = canvas.create_image(
    219.0,
    322.0,
    image=entry_image_pa
)
entry_pa = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_pa.place(
    x=154.0,
    y=312.0,
    width=130.0,
    height=23.0
)
entry_pa.bind("<Tab>", focus_next_widget)
entry_pa.bind("<Return>", prevent_newline)

entry_image_result = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_result = canvas.create_image(
    350.0,
    446.0,
    image=entry_image_result
)
entry_result = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    justify= "center",
    state = DISABLED
)
entry_result.place(
    x=285.0,
    y=434.0,
    width=130.0,
    height=23.0
)

entry_image_age = PhotoImage(
    file=relative_to_assets("entry_5.png"))
entry_bg_age = canvas.create_image(
    219.0,
    382.0,
    image=entry_image_age
)
entry_age = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_age.place(
    x=154.0,
    y=372.0,
    width=130.0,
    height=23.0
)
entry_age.bind("<Tab>", focus_next_widget)
entry_age.bind("<Return>", prevent_newline)

entry_image_bmi = PhotoImage(
    file=relative_to_assets("entry_6.png"))
entry_bg_bmi = canvas.create_image(
    491.0,
    202.0,
    image=entry_image_bmi
)
entry_bmi = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_bmi.place(
    x=426.0,
    y=192.0,
    width=130.0,
    height=23.0
)
entry_bmi.bind("<Tab>", focus_next_widget)
entry_bmi.bind("<Return>", prevent_newline)

entry_image_smoke = PhotoImage(
    file=relative_to_assets("entry_7.png"))
entry_bg_smoke = canvas.create_image(
    491.0,
    262.0,
    image=entry_image_smoke
)
entry_smoke = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_smoke.place(
    x=426.0,
    y=252.0,
    width=130.0,
    height=23.0
)
entry_smoke.bind("<Tab>", focus_next_widget)
entry_smoke.bind("<Return>", prevent_newline)

entry_image_dia = PhotoImage(
    file=relative_to_assets("entry_8.png"))
entry_bg_dia = canvas.create_image(
    491.0,
    322.0,
    image=entry_image_dia
)
entry_dia = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_dia.place(
    x=426.0,
    y=312.0,
    width=130.0,
    height=23.0
)
entry_dia.bind("<Tab>", focus_next_widget)
entry_dia.bind("<Return>", prevent_newline)

#submit button
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=getdata,
    relief="flat"
)
button_1.place(
    x=468.0,
    y=367.0,
    width=103.0,
    height=37.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=640.0,
    y=446.0,
    width=39.0,
    height=48.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_3 clicked"),
    relief="flat"
)
button_3.place(
    x=559.0,
    y=444.0,
    width=57.64706039428711,
    height=49.0
)

window.resizable(False, False)
window.mainloop()

