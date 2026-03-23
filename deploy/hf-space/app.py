"""Gradio app for skin lesion classification using Defused IBR model."""

from __future__ import annotations

import gradio as gr

from inference import predict


def run_inference(image):
    try:
        top_class, top_confidence, score_map = predict(image)
        return top_class, top_confidence, score_map
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="Skin Lesion Classifier (Defused IBR)") as demo:
    gr.Markdown("# Skin Lesion Classifier (Defused IBR5 + IBR6)")
    gr.Markdown(
        "Upload a dermoscopic image to get the predicted class and confidence scores. "
        "This demo is for research/education only and not for medical diagnosis."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Input Image")
            predict_button = gr.Button("Predict", variant="primary")
        with gr.Column(scale=1):
            class_output = gr.Textbox(label="Predicted Class")
            confidence_output = gr.Number(label="Confidence")
            scores_output = gr.Label(label="Class Probabilities", num_top_classes=7)

    predict_button.click(
        fn=run_inference,
        inputs=[image_input],
        outputs=[class_output, confidence_output, scores_output],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch()
