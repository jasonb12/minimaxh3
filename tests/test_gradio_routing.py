import unittest
from unittest.mock import patch

class GradioRoutingTest(unittest.TestCase):
    def test_ui_submits_without_touching_pipeline_or_local_queue(self):
        import app
        result = {'jobId':'queue-id','status':'queued','queueUrl':'https://brightify/queue'}
        before = len(app._jobs)
        with patch('gradio_queue.BrightifyQueue') as client, patch.object(app, '_run_generation', side_effect=AssertionError('UI invoked renderer')), patch.object(app, '_get_pipe', side_effect=AssertionError('UI loaded models')):
            client.return_value.submit.return_value = result
            video, info, job_id = app.generate(app.MODE_FL2VA, 'Test', None, None, [], [], [],
                '832×480 landscape (Spark fast)', 5.2, 7, 21, True, 1.0)
        self.assertIsNone(video)
        self.assertEqual(job_id, 'queue-id')
        self.assertIn('Brightify job queue-id', info)
        self.assertEqual(len(app._jobs), before)

    def test_worker_api_accepts_mixed_media_and_rejects_mixed_modes(self):
        import app
        request = app.GenerateRequest(prompt='Test', mode='ref2va', reference_video_urls=['https://v/test.mp4'], reference_audio_urls=['https://a/test.wav'])
        self.assertEqual(app._job_task(request), 'ref2va')
        with self.assertRaises(ValueError):
            app.GenerateRequest(prompt='Test', mode='fl2va', reference_video_urls=['https://v/test.mp4'])
        with self.assertRaises(ValueError):
            app.GenerateRequest(prompt='Test', reference_audio_urls=['https://a/test.wav'])
