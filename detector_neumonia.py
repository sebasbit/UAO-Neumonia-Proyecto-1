import csv
import os
import time
from tkinter import Tk, StringVar, Text, END
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING

import tkcap
from PIL import ImageTk, Image

from src import integrator


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10, state="disabled")

        #   GET ID
        self.ID_content = self.text1.get()

        #   TRACE para validar cuando se ingresa cédula
        self.ID.trace_add("write", self.validate_patient_id)

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", state="disabled", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", state="disabled", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", state="disabled", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID - Comentado porque el campo inicia deshabilitado
        # self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            # Llamada al módulo refactorizado
            self.array, img2show = integrator.load_and_prepare_image(filepath)

            # Procesamiento para la interfaz
            self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)

            # Habilitar campo de cédula y dar foco
            self.text1["state"] = "normal"
            self.text1.focus_set()

    def validate_patient_id(self, *args):
        # Habilitar botón "Predecir" solo si hay cédula ingresada y hay imagen cargada
        patient_id = self.ID.get().strip()
        if patient_id and self.array is not None:
            self.button1["state"] = "normal"
        else:
            self.button1["state"] = "disabled"

    def run_model(self):
        # Limpiar resultados anteriores
        self.text_img2.delete("1.0", END)
        self.text2.delete("1.0", END)
        self.text3.delete("1.0", END)

        # Generar nuevos resultados
        self.label, probability, self.heatmap = integrator.predict_pneumonia(self.array)
        self.proba = probability * 100
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

        # Habilitar botones de acciones post-predicción
        self.button6["state"] = "normal"  # Guardar
        self.button4["state"] = "normal"  # PDF
        self.button3["state"] = "normal"  # Borrar

    def save_results_csv(self):
        csv_path = os.path.abspath("historial.csv")
        with open(csv_path, "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(
                title="Guardar",
                message=f"Los datos se guardaron con éxito.\n\nRuta: {csv_path}"
            )

    def create_pdf(self):
        patient_id = self.text1.get()
        if not patient_id:
            patient_id = "Desconocido"
        timestamp = int(time.time())
        img_filename = f"Reporte_{patient_id}_{timestamp}.jpg"
        pdf_filename = f"Reporte_{patient_id}_{timestamp}.pdf"

        try:
            cap = tkcap.CAP(self.root)
            cap.capture(img_filename)

            # Verificar que la captura se realizó
            if not os.path.exists(img_filename):
                raise FileNotFoundError("La captura de pantalla falló. Esto puede ocurrir en macOS debido a permisos.")

            img = Image.open(img_filename)
            img = img.convert("RGB")
            img.save(pdf_filename)

            pdf_path = os.path.abspath(pdf_filename)
            showinfo(
                title="PDF",
                message=f"El PDF fue generado con éxito.\n\nRuta: {pdf_path}"
            )
        except Exception as e:
            showinfo(
                title="Error",
                message=f"Ocurrió un error al generar PDF:\n{str(e)}\n\nNota: En macOS, puede necesitar dar permisos de captura de pantalla a Python/Terminal en Configuración > Privacidad y Seguridad."
            )
        finally:
            if os.path.exists(img_filename):
                os.remove(img_filename)

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            # Limpiar campos de texto
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete("1.0", "end")
            self.text_img2.delete("1.0", "end")

            # Resetear estado de la aplicación
            self.array = None
            self.label = None
            self.proba = None
            self.heatmap = None

            # Resetear flujo: deshabilitar todo excepto "Cargar Imagen"
            self.text1["state"] = "disabled"
            self.button1["state"] = "disabled"  # Predecir
            self.button3["state"] = "disabled"  # Borrar
            self.button4["state"] = "disabled"  # PDF
            self.button6["state"] = "disabled"  # Guardar

            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
