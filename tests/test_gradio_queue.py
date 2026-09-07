import os
import unittest
from unittest.mock import patch, Mock
from gradio_queue import BrightifyQueue


class GradioQueueTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'BRIGHTIFY_QUEUE_URL': 'https://queue.test', 'BRIGHTIFY_QUEUE_TOKEN': 'test'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_unconfigured_fails_closed(self):
        with patch.dict(os.environ, {'BRIGHTIFY_QUEUE_TOKEN': ''}):
            with self.assertRaisesRegex(ValueError, 'No local render'):
                BrightifyQueue()

    def test_submission_preserves_options_and_reference_order(self):
        client = BrightifyQueue()
        client.upload = Mock(side_effect=lambda p,k,s: {'path':p,'kind':k})
        client.request = Mock(return_value={'jobId':'queued'})
        result = client.submit('ref2va',' Scene ',None,None,[('b.png','caption'),'a.png'],['v.mp4'],['s.wav'],
                               (832,480),8.3,7,21,True,1.15)
        self.assertEqual(result, {'jobId':'queued'})
        body = client.request.call_args.args[2]
        self.assertEqual([r['path'] for r in body['refs']], ['b.png','a.png','v.mp4','s.wav'])
        self.assertEqual(body['durationSeconds'],8.3)
        self.assertEqual(body['seed'],21)
        self.assertEqual(body['turboStrength'],1.15)
        self.assertTrue(body['includeAudio'])
        self.assertEqual(body['width'],832)

    def test_network_retry_reuses_idempotency_key(self):
        import requests
        response = Mock(ok=True, status_code=202)
        response.json.return_value = {'jobId':'one'}
        with patch('gradio_queue.requests.request', side_effect=[requests.Timeout(),response]) as send, patch('gradio_queue.time.sleep'):
            self.assertEqual(BrightifyQueue().request('POST','/jobs',{},key='fixed'), {'jobId':'one'})
        self.assertEqual([c.kwargs['headers']['Idempotency-Key'] for c in send.call_args_list], ['fixed','fixed'])

    def test_invalid_reference_set_is_rejected_before_upload(self):
        client = BrightifyQueue()
        client.upload = Mock()
        with self.assertRaisesRegex(ValueError, 'image or video'):
            client.submit('ref2va','Scene',None,None,[],[],['s.wav'],None,8,7,1,True,1)
        client.upload.assert_not_called()
