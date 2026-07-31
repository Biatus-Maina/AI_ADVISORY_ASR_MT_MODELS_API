import os
import gradio as gr
import requests

# Retrieve API endpoint from environment configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

def process_audio(audio_path, language_choice="Auto-detect"):
    if not audio_path:
        return "Tafadhali weka sauti (Please provide audio).", ""
    
    try:
        url = f"{API_URL}/process-audio"
        print(f"🎙️ Routing audio payload to API server: {url} (Hint: {language_choice})")
        
        filename = os.path.basename(audio_path)
        # Determine basic mime type or default to audio/wav
        mime_type = "audio/wav"
        if filename.endswith(".mp3"):
            mime_type = "audio/mpeg"
        elif filename.endswith(".m4a"):
            mime_type = "audio/mp4"
        elif filename.endswith(".ogg"):
            mime_type = "audio/ogg"

        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, mime_type)}
            data = {"language_choice": language_choice}
            response = requests.post(url, files=files, data=data, timeout=300)
            
        if response.status_code != 200:
            err_msg = f"Kosa kutoka kwa API (HTTP {response.status_code}): {response.text}"
            print(f"❌ Backend error: {err_msg}")
            return err_msg, ""
            
        result = response.json()
        return result.get("transcript", ""), result.get("translation", "")
        
    except Exception as e:
        print(f"❌ Network/Connection error: {str(e)}")
        return f"Kosa la muunganisho (Connection error): {str(e)}. Hakikisha kuwa huduma ya API inafanya kazi kwenye {API_URL}.", ""


# Build Premium Gradio Interface (matches original design)
theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

css_styles = """
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #10b981, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem !important;
    margin-bottom: 2px;
}
.sub-header {
    text-align: center;
    color: #64748b;
    font-size: 1.2rem;
    margin-bottom: 30px;
    font-weight: 500;
}
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    padding: 30px !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1) !important;
}
"""

with gr.Blocks(title="Swahili Agric Advisory") as demo:
    gr.HTML("<h1 class='main-header'>Swahili AgriTranslate</h1>")
    gr.HTML("<p class='sub-header'>Swahili Speech Transcription & English Advisory Translation</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Sema au Weka Sauti (Record or Upload Audio)"
            )
            language_choice = gr.Dropdown(
                choices=["Auto-detect", "Swahili", "English"],
                value="Auto-detect",
                label="Target Language Hint"
            )
            with gr.Row():
                submit_btn = gr.Button("Tafsiri (Translate)", variant="primary")
                clear_btn = gr.Button("Futa (Clear)", variant="secondary")
            
        with gr.Column(scale=1):
            transcript_output = gr.Textbox(
                label="Maandishi ya Sauti (Audio Transcript)",
                placeholder="Maandishi yataonekana hapa...",
                interactive=False,
                buttons=["copy"]
            )
            translation_output = gr.Textbox(
                label="Tafsiri (Translation)",
                placeholder="Tafsiri itaonekana hapa...",
                interactive=False,
                buttons=["copy"]
            )
            
    # Interactive wiring
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, language_choice],
        outputs=[transcript_output, translation_output]
    )
    clear_btn.click(
        fn=lambda: (None, "Auto-detect", "", ""),
        inputs=None,
        outputs=[audio_input, language_choice, transcript_output, translation_output]
    )

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Launching frontend Gradio client on http://{host}:{port} targeting API at {API_URL}...")
    demo.launch(server_name=host, server_port=port, theme=theme, css=css_styles)

