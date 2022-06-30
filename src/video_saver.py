import numpy as np
import imageio


class VideoSaver:
    """
    Class that generates mp4 videos from matplotlib figure information.
    Usage:
        - Create new video_saver = VideoSaver()
        - [... your code that repeatedly updates/creates matplotlib Figure object(s)
            - after every update (i.e. at each frame you want to save in your video): video_saver.add_frame(fig)
                if you dont have a reference to your figure, consider using `plt.gcf()` to get the current figure
        - Once all images are saved: video_saver.generate_video("my_video.mp4")

    Note: Re-sizing the figure between calls to add_frame causes all previous frames to be deleted. This is intended
    behaviour, as imageio.mimsave can not create a single video of images of different sizes.
        ! This includes changing your matplotlib window into fullscreen mode !

    This implementation is built around using plt.draw() / plt.pause(...) as a pseudo-animation system.

    If you get an error about a missing renderer, you probably don't call fig.canvas.draw() somewhere. This might
    indicate that you plotting structure uses plt.show(), in which case you should try to change to the plt.draw()/pause
    based approach. If you want to stick to the plt.show() based approach (which is fine too), you need to call .draw()
    BEFORE plt.show(). In my testing, doing <plotting> -> fig.canvas.draw() -> .add_frame(fig) -> plt.show() works.
    """
    def __init__(self, verbose=False):
        """
        Create a new VideoSaver
        :param verbose: Bool: Whether to be verbose about performed actions, printing information to console
        """
        self._verbose = verbose
        self._frames = []
        self._last_shape = None

    def clear_frame_list(self):
        self._frames = []
        self._last_shape = None

    def get_frames(self):
        return self._frames

    def add_frame(self, fig):
        """
        Add a snapshot of the current figure to the internal list of _frames
        :param fig: Matplotlib figure to extract from
        """
        # extract
        plot_output = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot_output = plot_output.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # ensure shape stayed the same, else clear before continuing
        if self._last_shape is not None and self._last_shape != plot_output.shape:
            if self._verbose:
                print("Saved frame dimension mismatch to previous frames, clearing {} previous frames and starting "
                      "from scratch.".format(len(self._frames)))
            self.clear_frame_list()

        # save
        self._last_shape = plot_output.shape
        self._frames.append(plot_output)

    def generate_video(self, filename, fps=60, repeat_frame_count=20):
        """
        Generate a video and save it to file.

        The default fps + repeat_frame_count parameters produce smooth where each saved image is shown for about 0.3s

        If your saved videos lag/stop/have other artifacts, consider increasing fps and repeat_frame_count parameters
        accordingly. This helps mimsave in my experience.
        :param filename: Name/Location of the file, passed to imageio.mimsave. E.g. "video.mp4"
        :param fps: FPS of the video. Typical values are 30 or 60.
        :param repeat_frame_count: How often to include each frame. If you want high FPS but a longer video, you can
        use this to show every frame multiple times. A value of 1 indicates each frame should be used once.
        """
        repeat_frame_count = int(repeat_frame_count)
        if repeat_frame_count > 1:
            frame_list = []
            for frame in self._frames:
                for i in range(repeat_frame_count):
                    frame_list.append(frame)
        else:
            frame_list = self._frames
        imageio.mimsave(filename, frame_list, fps=fps)
        if self._verbose:
            print("Successfully created video {} (fps={}, repeat_frame_count={}).".format(filename, fps, repeat_frame_count))
