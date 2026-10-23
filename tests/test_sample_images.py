import tempfile
import unittest
from pathlib import Path
from PIL import Image
from ui.sample_images import sample_image
from processing.events import build_event
from simulation.generate_data import generate_trip
from ui.maps import map_html

class SampleImageTests(unittest.TestCase):
    def test_category_and_stable_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder,color in [('pothole_images','red'),('speedbreaker_images','blue')]:
                (root/folder).mkdir()
                Image.new('RGB',(40,30),color).save(root/folder/'sample.png')
            for category,folder in [('pothole','pothole_images'),('speed_breaker','speedbreaker_images')]:
                event={'id':'example','prediction':category}
                first=sample_image(event,root)
                self.assertTrue(first['key'].startswith(folder+'/'))
                self.assertTrue(first['url'].startswith('data:image/jpeg;base64,'))
                self.assertEqual(first,sample_image(event,root))
            self.assertIsNone(sample_image({'id':'x','prediction':'normal'},root))
            (root/'pothole_images/sample.png').write_text('broken')
            self.assertIsNone(sample_image({'id':'x','prediction':'pothole'},root))

    def test_speed_breaker_event_and_actual_folder_images(self):
        frame=generate_trip(0,42,1)
        for category in ['pothole','speed_breaker']:
            event=build_event(frame,dict(label=category,confidence=.8,method='motion'),'image-test')
            self.assertEqual(event['prediction'],category)
            self.assertEqual(event['evidence'],[])
            sample=sample_image(event)
            self.assertIsNotNone(sample)
            event['status']='Inspection required'
            html=map_html([event],[],tiles=False)
            self.assertIn('data:image/jpeg;base64,',html)
            self.assertIn('not event evidence',html)
        self.assertIsNone(build_event(frame,dict(label='rough_road'),'test'))
