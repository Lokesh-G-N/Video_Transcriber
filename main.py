import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from PIL import Image
import torch
import os
import tempfile
import shutil
import cv2
from collections import Counter

st.set_page_config(page_title="Video Captioning with SceneDetect", layout="centered")

@st.cache_resource(show_spinner=False)
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.eval()
    return processor, model

@st.cache_resource(show_spinner=False)
def load_flan_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    model.eval()
    return tokenizer, model

processor, blip_model = load_blip_model()
flan_tokenizer, flan_model = load_flan_model()

def detect_scenes(video_path, output_dir, max_keyframes=20):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=5.0, min_scene_len=5))  # Adaptive threshold
    base_timecode = video_manager.get_base_timecode()

    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)

        if scene_list:
            save_images(scene_list, video_manager, output_dir=output_dir)
            all_frames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")])
            if len(all_frames) > max_keyframes:
                step = max(1, len(all_frames) // max_keyframes)
                selected = all_frames[::step][:max_keyframes]
                for f in all_frames:
                    if f not in selected:
                        os.remove(f)
                return selected
            return all_frames

        # If no scenes found
        return sample_frames(video_path, output_dir, max_keyframes)

    finally:
        video_manager.release()

def sample_frames(video_path, output_dir, max_frames=12):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    count, saved = 0, 0
    paths = []
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            path = os.path.join(output_dir, f"frame_{saved}.jpg")
            cv2.imwrite(path, frame)
            paths.append(path)
            saved += 1
        count += 1
    cap.release()
    return paths

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def generate_summary_frequent(captions):
    # Count and group by frequency
    caption_counts = Counter(captions)
    common = caption_counts.most_common()
    frequent_captions = [cap for cap, count in common if count > 1 or len(common) <= 3]

    prompt = (
        "Write a clear, coherent paragraph describing the video based on the following frequent visual elements:\n\n"
        + "\n".join(f"- {c}" for c in frequent_captions)
        + "\n\nVideo description:"
    )
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = flan_model.generate(
            **inputs,
            max_length=150,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("🎬 AI Video Description from Scene Keyframes")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file:
    temp_video_path = os.path.join(tempfile.gettempdir(), "video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.video(temp_video_path)

    st.info("🔍 Detecting scenes and extracting up to 20 keyframes...")
    keyframe_dir = os.path.join(tempfile.gettempdir(), "keyframes")
    os.makedirs(keyframe_dir, exist_ok=True)
    keyframes = detect_scenes(temp_video_path, keyframe_dir, max_keyframes=20)

    if not keyframes:
        st.error("No keyframes could be extracted.")
    else:
        st.success(f"{len(keyframes)} keyframes extracted.")
        captions = []
        for i, frame_path in enumerate(sorted(keyframes)):
            caption = generate_caption(frame_path)
            captions.append(caption)
            st.image(frame_path, caption=f"Keyframe {i+1}: {caption}", use_column_width=True)

        st.info("📝 Generating final summary from keyframe descriptions...")
        summary = generate_summary_frequent(captions)

        st.markdown("---")
        st.subheader("📄 Natural Video Description")
        st.write(summary)

        # Cleanup
        shutil.rmtree(keyframe_dir)
