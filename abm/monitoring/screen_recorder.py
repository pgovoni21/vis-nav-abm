import cv2
import pygame


class ScreenRecorder:
    """
        This class is used to record a PyGame surface and save it to a video file.
        https://github.com/tdrmk/pygame_recorder
    """
    def __init__(self, width, height, fps, out_file='output.avi'):
        """
        Initialize the recorder with parameters of the surface.
        :param width: Width of the surface to capture
        :param height: Height of the surface to capture
        :param fps: Frames per second
        :param out_file: Output file to save the recording
        """
        print(f'Initializing ScreenRecorder with parameters width:{width} height:{height} fps:{fps}.')
        print(f'Output of the screen recording saved to {out_file}.')

        # define the codec and create a video writer object
        four_cc = cv2.VideoWriter_fourcc('m','p','4','v')
        self.video = cv2.VideoWriter(out_file, four_cc, float(fps), (width, height))

    def capture_frame(self, surf):
        """
         Call this method every frame, pass in the pygame surface to capture.
        :param surf: pygame surface to capture
        :return: None
        """
        """
           
            Note: surface must have the dimensions specified in the constructor.
        """
        # transform the pixels to the format used by open-cv
        pixels = cv2.rotate(pygame.surfarray.pixels3d(surf), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # write the frame
        self.video.write(pixels)

    def end_recording(self):
        """
        Call this method to stop recording.
        :return: None
        """
        # stop recording
        self.video.release()