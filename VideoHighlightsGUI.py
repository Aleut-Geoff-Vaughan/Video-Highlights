"""
Video Highlights Generator - GUI Frontend
-----------------------------------------
A graphical user interface for the Video Highlights generator.
Allows easy video selection, trimming, and configuration of all options.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import subprocess
from pathlib import Path


class VideoHighlightsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Highlights Generator")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # Variables
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="./highlights_output")
        self.trim_start = tk.StringVar()
        self.trim_end = tk.StringVar()
        self.pre_seconds = tk.StringVar(value="2.0")
        self.post_seconds = tk.StringVar(value="6.0")
        self.min_clip = tk.StringVar(value="4.0")
        self.select_player = tk.BooleanVar(value=False)
        self.overlay = tk.BooleanVar(value=False)
        self.no_audio = tk.BooleanVar(value=False)
        self.processing = False

        self.setup_ui()

    def setup_ui(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # Title
        title = ttk.Label(main_frame, text="Soccer Highlight Generator",
                         font=('Helvetica', 16, 'bold'))
        title.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1

        # Video file selection
        ttk.Label(main_frame, text="Video File:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_video).grid(row=row, column=2, padx=(5, 0), pady=5)
        row += 1

        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_output).grid(row=row, column=2, padx=(5, 0), pady=5)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1

        # Trim section
        trim_label = ttk.Label(main_frame, text="Video Trimming (Optional)", font=('Helvetica', 11, 'bold'))
        trim_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1

        # Trim start
        ttk.Label(main_frame, text="Trim Start:").grid(row=row, column=0, sticky=tk.W, pady=5)
        trim_start_entry = ttk.Entry(main_frame, textvariable=self.trim_start, width=20)
        trim_start_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(main_frame, text="(MM:SS or HH:MM:SS)", foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=(5, 0))
        row += 1

        # Trim end
        ttk.Label(main_frame, text="Trim End:").grid(row=row, column=0, sticky=tk.W, pady=5)
        trim_end_entry = ttk.Entry(main_frame, textvariable=self.trim_end, width=20)
        trim_end_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(main_frame, text="(MM:SS or HH:MM:SS)", foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=(5, 0))
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1

        # Detection parameters
        param_label = ttk.Label(main_frame, text="Detection Parameters", font=('Helvetica', 11, 'bold'))
        param_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1

        # Pre-event buffer
        ttk.Label(main_frame, text="Pre-event buffer (seconds):").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.pre_seconds, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Post-event buffer
        ttk.Label(main_frame, text="Post-event buffer (seconds):").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.post_seconds, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Minimum clip duration
        ttk.Label(main_frame, text="Minimum clip duration (seconds):").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.min_clip, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1

        # Options
        options_label = ttk.Label(main_frame, text="Options", font=('Helvetica', 11, 'bold'))
        options_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1

        # Checkboxes
        ttk.Checkbutton(main_frame, text="Manually select player on first frame",
                       variable=self.select_player).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=3)
        row += 1

        ttk.Checkbutton(main_frame, text="Create spotlight overlay clips (slower)",
                       variable=self.overlay).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=3)
        row += 1

        ttk.Checkbutton(main_frame, text="Disable audio-based detection",
                       variable=self.no_audio).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=3)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1

        # Run button
        self.run_button = ttk.Button(main_frame, text="Generate Highlights",
                                     command=self.run_processing, style='Accent.TButton')
        self.run_button.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # Progress/Output section
        output_label = ttk.Label(main_frame, text="Output:", font=('Helvetica', 10, 'bold'))
        output_label.grid(row=row, column=0, sticky=tk.W, pady=(10, 5))
        row += 1

        # Console output
        self.console = scrolledtext.ScrolledText(main_frame, height=12, width=80,
                                                 state='disabled', wrap=tk.WORD,
                                                 background='#1e1e1e', foreground='#d4d4d4',
                                                 font=('Courier', 9))
        self.console.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(row, weight=1)
        row += 1

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # Configure style for accent button
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Helvetica', 11, 'bold'))

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
            # Auto-set output directory based on video location
            video_dir = os.path.dirname(filename)
            video_name = os.path.splitext(os.path.basename(filename))[0]
            default_output = os.path.join(video_dir, f"{video_name}_highlights")
            self.output_dir.set(default_output)

    def browse_output(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)

    def log(self, message):
        """Add message to console"""
        self.console.configure(state='normal')
        self.console.insert(tk.END, message + "\n")
        self.console.configure(state='disabled')
        self.console.see(tk.END)
        self.root.update()

    def validate_inputs(self):
        """Validate user inputs"""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return False

        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", f"Video file not found: {self.video_path.get()}")
            return False

        if not self.output_dir.get():
            messagebox.showerror("Error", "Please specify an output directory")
            return False

        # Validate numeric inputs
        try:
            float(self.pre_seconds.get())
            float(self.post_seconds.get())
            float(self.min_clip.get())
        except ValueError:
            messagebox.showerror("Error", "Detection parameters must be valid numbers")
            return False

        return True

    def build_command(self):
        """Build the command line for VideoHighlights.py"""
        cmd = [sys.executable, "VideoHighlights.py"]

        cmd.extend(["--video", self.video_path.get()])
        cmd.extend(["--out", self.output_dir.get()])
        cmd.extend(["--pre", self.pre_seconds.get()])
        cmd.extend(["--post", self.post_seconds.get()])
        cmd.extend(["--min-clip", self.min_clip.get()])

        if self.trim_start.get():
            cmd.extend(["--trim-start", self.trim_start.get()])
        if self.trim_end.get():
            cmd.extend(["--trim-end", self.trim_end.get()])

        if self.select_player.get():
            cmd.append("--select")
        if self.overlay.get():
            cmd.append("--overlay")
        if self.no_audio.get():
            cmd.append("--no-audio")

        return cmd

    def run_processing(self):
        """Run the video processing in a separate thread"""
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already running")
            return

        if not self.validate_inputs():
            return

        # Clear console
        self.console.configure(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.configure(state='disabled')

        # Build command
        cmd = self.build_command()
        self.log(f"Running command: {' '.join(cmd)}\n")
        self.log("=" * 80)

        # Disable run button
        self.run_button.configure(state='disabled')
        self.processing = True
        self.status_var.set("Processing...")

        # Run in thread
        thread = threading.Thread(target=self.execute_command, args=(cmd,))
        thread.daemon = True
        thread.start()

    def execute_command(self, cmd):
        """Execute the command and capture output"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Read output line by line
            for line in process.stdout:
                self.log(line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.log("\n" + "=" * 80)
                self.log("✓ Processing completed successfully!")
                self.status_var.set("Completed successfully")
                messagebox.showinfo("Success",
                                  f"Highlights generated successfully!\n\nOutput saved to:\n{self.output_dir.get()}")
            else:
                self.log("\n" + "=" * 80)
                self.log(f"✗ Processing failed with return code {process.returncode}")
                self.status_var.set("Failed")
                messagebox.showerror("Error", "Processing failed. Check the output for details.")

        except Exception as e:
            self.log(f"\n✗ Error: {str(e)}")
            self.status_var.set("Error")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

        finally:
            # Re-enable run button
            self.processing = False
            self.run_button.configure(state='normal')
            if self.status_var.get() == "Processing...":
                self.status_var.set("Ready")


def main():
    root = tk.Tk()
    app = VideoHighlightsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
