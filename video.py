import os
from moviepy import ImageSequenceClip

def images_to_video(image_folder, output_video, seconds_per_image=0.5):

    image_extensions = (".png", ".jpg", ".jpeg")

    images = sorted([
        os.path.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.lower().endswith(image_extensions)
    ])

    if not images:
        print(f"No images found in {image_folder}")
        return

    # Duration of each image
    durations = [seconds_per_image] * len(images)

    # Create video clip
    clip = ImageSequenceClip(images, durations=durations)

    # Export MP4 video
    clip.write_videofile(output_video, codec="libx264", fps=24)

    print(f"Saved video: {output_video}")


if __name__ == "__main__":

    # Desktop path
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")

    # Main folder
    base_folder = os.path.join(desktop, "heatflux_video3")

    # Subfolders
    absorbed_folder = os.path.join(base_folder, "absorbed")
    incident_folder = os.path.join(base_folder, "incident")

    # Output videos
    absorbed_video = os.path.join(base_folder, "absorbed.mp4")
    incident_video = os.path.join(base_folder, "incident.mp4")

    # Create videos
    images_to_video(
        absorbed_folder,
        absorbed_video,
        seconds_per_image=0.5
    )

    images_to_video(
        incident_folder,
        incident_video,
        seconds_per_image=0.5
    )
