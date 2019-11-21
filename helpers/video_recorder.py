import os.path as osp

import gym
from gym.wrappers.monitoring import video_recorder

from helpers import logger


class VideoRecorder(gym.Wrapper):

    def __init__(self, env, save_dir, record_video_trigger, video_length=200, prefix='vid'):
        """Wrap environment to record rendered image as mp4 video.

        Args:
            env: Env to wrap
            save_dir: Where to save video recordings
            record_video_trigger: Function that defines when to start recording.
                                  The function takes the current number of step,
                                  and returns whether we should start recording or not.
            video_length: Length of recorded video in frames
        """
        # Inherit attributes from parent class
        gym.Wrapper.__init__(self, env)
        # Define new attributes
        self.save_dir = save_dir
        self.record_video_trigger = record_video_trigger
        self.video_length = video_length
        self.video_recorder = None
        self.prefix = prefix
        self.step_id = 0
        self.recording = False
        self.num_recorded_frames = 0

    def start_video_recorder(self):
        """Start video recorder"""
        # Close the recorder if already currently recording
        self.close_video_recorder()
        # Define video recording's path
        vid = "{}_step{}_to_step{}".format(self.prefix,
                                           self.step_id,
                                           self.step_id + self.video_length)
        path = osp.join(self.save_dir, vid)
        # Define video recorder
        self.video_recorder = video_recorder.VideoRecorder(env=self.env,
                                                           base_path=path,
                                                           metadata={'step_id': self.step_id})
        # Render and add a frame to the video
        self.video_recorder.capture_frame()
        # Update running statistics
        self.num_recorded_frames = 1
        self.recording = True

    def reset(self):
        obs = self.env.reset()
        self.start_video_recorder()
        return obs

    def _video_enabled(self):
        """Answer whether we record a frame at the current step or not"""
        is_enabled = self.record_video_trigger(self.step_id)
        assert isinstance(is_enabled, bool)
        return is_enabled

    def step(self, action):
        """Do one step, render and record the frame"""
        # Perform step in environment
        obs, reward, done, info = self.env.step(action)
        # Render and record the associated frame
        self.step_id += 1
        if self.recording:
            # Render and add a frame to the video
            self.video_recorder.capture_frame()
            self.num_recorded_frames += 1
            if self.num_recorded_frames > self.video_length:
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, reward, done, info

    def close_video_recorder(self):
        """Close video recorder"""
        if self.recording:
            logger.info("saving video to:\n  {}".format(self.video_recorder.path))
            #  If recording, close the recorder
            self.video_recorder.close()
        # Reset running statistics
        self.recording = False
        self.num_recorded_frames = 1

    def close(self):
        """Close the environment and close video recorder"""
        gym.Wrapper.close(self)
        self.close_video_recorder()

    def __del__(self):
        """Close the envionment at garbage collection"""
        self.close()
