import tempfile
import unittest
from pathlib import Path
import pandas as pd
from simulation.generate_data import generate_trip
from processing.features import extract
from processing.detector import Detector
from processing.events import build_event
from processing.preprocess import validate
from storage.database import save_event, list_events, update_status

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.trip=generate_trip(99,2026,4)
        self.window=self.trip[self.trip.window_id==1].copy()

    def test_deterministic_generator(self):
        pd.testing.assert_frame_equal(self.trip,generate_trip(99,2026,4))

    def test_rejects_invalid_samples(self):
        invalid=self.window.copy()
        invalid.loc[invalid.index[0],'az']=float('nan')
        with self.assertRaises(ValueError): validate(invalid)
        invalid=self.window.copy()
        invalid['timestamp']=0
        with self.assertRaises(ValueError): validate(invalid)

    def test_predictions_do_not_read_ground_truth(self):
        for method in ['threshold','motion','speed']:
            detector=Detector(method)
            expected=detector.predict(self.window)
            changed=self.window.copy()
            changed['label']='not_a_real_class'
            self.assertEqual(expected,detector.predict(changed))
            self.assertEqual(expected,detector.predict(changed.drop(columns=['label'])))

    def test_feature_contract(self):
        from config import FEATURES
        self.assertEqual(set(extract(self.window)),set(FEATURES+['speed_kmh']))

    def test_event_and_persistent_review(self):
        prediction={'label':'pothole','confidence':.9,'method':'speed'}
        event=build_event(self.window,prediction,'run1')
        impact=self.window.loc[self.window.az.abs().idxmax()]
        self.assertEqual(event['timestamp'],impact.timestamp)
        self.assertEqual(event['latitude'],impact.latitude)
        with tempfile.TemporaryDirectory() as d:
            db=Path(d)/'test.sqlite3'
            save_event(event,db)
            update_status(event['id'],'Repaired',db)
            save_event(event,db)
            self.assertEqual(len(list_events(db)),1)
            self.assertEqual(list_events(db)[0]['status'],'Repaired')
            with self.assertRaises(ValueError): update_status(event['id'],'invalid',db)
        self.assertIsNone(build_event(self.window,{'label':'normal'},'run1'))
        self.assertNotEqual(event['id'],build_event(self.window,prediction,'run2')['id'])

    def test_split_separation(self):
        import json
        from config import DATA
        splits=json.loads((DATA/'manifest.json').read_text())['splits']
        groups=[set(v) for v in splits.values()]
        for i,a in enumerate(groups):
            for b in groups[i+1:]: self.assertFalse(a & b)

if __name__=='__main__': unittest.main()
