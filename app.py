"""
Integrates all modules and runs the Gradio interface.
"""

import gradio as gr
from data_loader import preprocess_image
from model import (
    load_recognition_model,
    recognize_face,
    generate_and_save_embeddings,
    load_embeddings,
)
import attendance  # Import the entire attendance module
import os
from arcface_recognition import recognize_student_arcface
from card_attendance import mark_attendance_from_card

# --- Configuration ----
dataset_path = "dataset/"
# ----------------------

# Load model and processor
print("Loading model...")
model, processor = load_recognition_model()

# Load or generate embeddings
known_embeddings = load_embeddings()
if known_embeddings is None:
    print("Embeddings not found. Generating from dataset...")
    # Ensure the dataset directory exists before generating embeddings
    if not os.path.exists(dataset_path):
        print(
            f"Error: Dataset directory '{dataset_path}' not found. Cannot generate embeddings."
        )
        # Exit or handle error appropriately if dataset is missing
    else:
        generate_and_save_embeddings(dataset_path, model, processor)
        known_embeddings = load_embeddings()  # Load again after generation

# Extract student IDs from known embeddings for recognition function
student_ids = list(known_embeddings.keys()) if known_embeddings else []


def load_and_display_attendance():
    """
    Loads attendance records and returns them as a pandas DataFrame.
    """
    print("Loading attendance records for display...")
    df = attendance.load_attendance_records()
    # Optionally, format the date/time columns for better display
    return df


def attendance_system(image):
    """
    Processes the input image, performs face recognition, and records attendance.

    Args:
        image (PIL.Image.Image): The input image from Gradio.

    Returns:
        str: The attendance status message.
    """
    # Use ArcFace-based recognition
    predicted_student_id, confidence = recognize_student_arcface(image)

    if predicted_student_id:
        message = attendance.record_attendance(predicted_student_id)
        message += f" (Confidence: {confidence:.4f})"
    elif confidence is not None:
        message = f"Face not recognized. (Highest Confidence: {confidence:.4f})"
    else:
        message = "Face not recognized."

    return message


# Set up the Gradio interface.
if __name__ == "__main__":
    if (
        model is not None
        and processor is not None
        and known_embeddings is not None
        and student_ids
    ):
        # Set up the Gradio interface with theme and title
        with gr.Blocks(theme=gr.themes.Default(), title="Face Recognition System") as demo:
            gr.Label("Face Recognition Attendance System") # Main label for the app
            gr.Markdown("Upload an image to record your attendance (Face or Card).")

            with gr.Tab("Face Recognition Attendance"):
                with gr.Row():
                    with gr.Column(scale=1):
                        output_message = gr.Textbox(label="Status")
                        gr.Markdown("## Attendance Records")
                        with gr.Row():
                            load_button = gr.Button("Refresh Attendance Records")
                        attendance_table = gr.DataFrame(label="Attendance Records")
                    with gr.Column(scale=1):
                        image_input = gr.Image(type="pil", label="Upload Face Image")
                        with gr.Row():
                            clear_button = gr.Button("Clear", scale=0)
                            submit_button = gr.Button("Submit", scale=0, variant="primary")
                submit_button.click(
                    fn=attendance_system,
                    inputs=image_input,
                    outputs=output_message
                )
                clear_button.click(
                    fn=lambda: [None, ""],
                    inputs=None,
                    outputs=[image_input, output_message]
                )
                load_button.click(
                    fn=load_and_display_attendance,
                    inputs=None,
                    outputs=attendance_table
                )

            with gr.Tab("Card-Based Attendance"):
                with gr.Row():
                    with gr.Column(scale=1):
                        card_output_message = gr.Textbox(label="Status (Card)")
                    with gr.Column(scale=1):
                        card_image_input = gr.Image(type="pil", label="Upload Card Image")
                        with gr.Row():
                            card_clear_button = gr.Button("Clear Card", scale=0)
                            card_submit_button = gr.Button("Submit Card", scale=0, variant="primary")
                card_submit_button.click(
                    fn=mark_attendance_from_card,
                    inputs=card_image_input,
                    outputs=card_output_message
                )
                card_clear_button.click(
                    fn=lambda: [None, ""],
                    inputs=None,
                    outputs=[card_image_input, card_output_message]
                )

        print("Launching Gradio interface...")
        demo.launch()
    else:
        print("Cannot launch Gradio interface due to initialization errors.")
