import gradio as gr
from tkinter import Tk, filedialog
import pandas as pd
from pathlib import Path
import openpyxl
from image_processing import process_images


class Selections():
    FILES = 'Files'
    DIRECTORY = 'Directory'

RESULTS_PATH = Path(__file__).parents[1] / 'results'
RESULTS_PATH.mkdir(exist_ok=True)

def on_browse(data_type):
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    if data_type == Selections.FILES:
        filenames = filedialog.askopenfilenames(
            filetypes=[("Image-files", "*.jpg;*.jpeg;*.png;*.tiff;*.tif")]
        )

        if len(filenames) > 0:
            root.destroy()
            return [Path(file) for file in filenames]
        else:
            root.destroy()
            return "No files selected."

    elif data_type == Selections.DIRECTORY:
        filename = filedialog.askdirectory()
        if filename:
            directory_path = Path(filename)
            if directory_path.is_dir():
                root.destroy()
                return directory_path
        else:
            root.destroy()
            return "No directory selected."

def get_app():

    footer_text = """
    <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                width: 100%; text-align: center;'>
        <b>This application is developed at KIT - Karlsruhe Institute of Technology.</b>
    </div>
    """

    with gr.Blocks() as app:
        with gr.Tab("Single Image"):
            gr.Interface(
                fn=process_single_image,
                inputs=gr.Image(type="filepath"),
                outputs=["text"],
                description="Perform genus/sex classification and outlier detection on a single image by dragging it into the designated box.",
                title="Diapriidae/Ismaridae Image Genus Classification and Outlier Detection",
                article=footer_text,
                allow_flagging='never'
            )
        with gr.Tab("Image Batch"):
            data_type = gr.Radio(choices=[Selections.DIRECTORY, Selections.FILES], value=Selections.DIRECTORY, label="Image data type")
            output_txt = gr.Textbox(label="", scale=5, interactive=False)
            image_browse_btn = gr.Button("Browse", min_width=1)
            image_browse_btn.click(process_image_batch, inputs=data_type, outputs=output_txt)
            gr.HTML(footer_text)

    return app

def save_as_excel(data: pd.DataFrame, decimals=0):
    i = 0
    while True:
        result_dir = RESULTS_PATH / f'predictions_{i}.xlsx'
        if not result_dir.exists():
            SAVE_PATH = result_dir
            break
        i += 1

    df = data
    df = df.round(decimals) if decimals else df

    excel_writer = pd.ExcelWriter(SAVE_PATH, engine='openpyxl')
    df.to_excel(excel_writer, sheet_name='Predictions', index=False)

    worksheet = excel_writer.sheets['Predictions']

    # red cell color if outlier detected
    for index, row in df.iterrows():
        is_outlier = row['Outlier']
        # color green or red
        color = 'FF0000' if is_outlier else '00FF00'
        idx = df.columns.get_loc('Outlier')
        cell = worksheet.cell(row=index + 2, column=idx + 1)
        cell.fill = openpyxl.styles.PatternFill(start_color=color, end_color=color, fill_type="solid")

    excel_writer.close()

    return SAVE_PATH

def process_single_image(image):
    image = Path(image)
    df = process_images([image])

    pred_genus = df['Genus'].tolist()[0]
    genus_score = df['Genus Score'].tolist()[0]
    pred_sex = df['Sex'].tolist()[0]
    sex_score = df['Sex Score'].tolist()[0]
    is_outlier = df['Outlier'].tolist()[0]

    if is_outlier:
        text_result = f"""Prediction from Classification:
Non-Hymenoptera.
Outlier detected."""
    else:
        text_result = f"""Prediction from Classification:
{pred_genus}: {round(genus_score, 2)}
{pred_sex}: {round(sex_score, 2)}

Prediction from Outlier Detection:
No Outlier detected."""

    return text_result

def process_image_batch(data_type):
    browsed = on_browse(data_type)
    if isinstance(browsed, str):
        # returned string means error; exit
        return browsed
    elif isinstance(browsed, Path):
        image_types = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        img_paths = [path for path in browsed.rglob('*') if path.suffix.lower() in image_types]
        if not len(img_paths):
            return f'No images found in {browsed}.'
    else:
        img_paths = browsed

    df = process_images(img_paths)

    file_path = save_as_excel(df, decimals=3)
    
    return f"Results saved to {file_path}"

if __name__ == '__main__':
    app = get_app()
    app.launch(inbrowser=True)
