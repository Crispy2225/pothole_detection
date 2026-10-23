import unittest
import numpy as np
from simulation.generate_data import generate_trip
from simulation.import_csv import import_csv
from processing.events import build_event
from processing.detector import Detector
from ui.insights import replay_summary
from ui.maps import valid_location

class ImportTests(unittest.TestCase):
    def setUp(self):self.frame=generate_trip(0,44,4)
    def raw(self,frame=None):return (self.frame if frame is None else frame).to_csv(index=False).encode()

    def test_recorded_provenance_and_unknown_labels(self):
        frame=self.frame.drop(columns=['label','speed_kmh','latitude','longitude'])
        recording=import_csv(self.raw(frame))
        self.assertEqual(recording.metadata['source'],'imported')
        self.assertEqual(recording.metadata['labelled_windows'],0)
        self.assertFalse(recording.metadata['has_speed'])
        prediction=Detector('motion').predict(recording.windows[0])
        self.assertIn(prediction['label'],['pothole','normal','rough_road','speed_breaker'])
        with self.assertRaises(ValueError):Detector('speed').predict(recording.windows[0])
        event=build_event(recording.windows[0],dict(label='pothole',confidence=.8,method='motion'),'run',recording.metadata)
        self.assertEqual(event['source'],'imported')
        self.assertEqual(event['evidence'],[])
        self.assertIsNone(event['latitude'])
        self.assertFalse(valid_location(event))
        self.assertIsNone(replay_summary([{'Ground truth':None,'Prediction':'pothole'}])['precision'])

    def test_unit_conversion_and_identity(self):
        a=import_csv(self.raw())
        frame=self.frame.copy();frame['timestamp']*=1000
        frame[['ax','ay','az']]/=9.80665;frame[['gx','gy','gz']]*=180/np.pi
        b=import_csv(self.raw(frame),time_unit='milliseconds',acceleration_unit='g',gyro_unit='deg/s')
        np.testing.assert_allclose(a.frame.az,b.frame.az,atol=1e-10)
        np.testing.assert_allclose(a.frame.gx,b.frame.gx,atol=1e-10)
        self.assertNotEqual(a.metadata['dataset_id'],import_csv(self.raw(),source='synthetic').metadata['dataset_id'])

    def test_rejects_bad_inputs(self):
        cases=[]
        f=self.frame.copy();f['timestamp']*=2;cases.append(f)
        f=self.frame.copy();f.loc[2,'timestamp']=f.loc[1,'timestamp'];cases.append(f)
        f=self.frame.copy();f.loc[2,'az']=np.nan;cases.append(f)
        f=self.frame.copy();f.loc[2,'latitude']=95;cases.append(f)
        f=self.frame.copy();f.loc[2,'label']='mystery';cases.append(f)
        f=self.frame.copy();f.loc[2,'trip_id']=55;cases.append(f)
        f=self.frame.copy();f.loc[20:,'timestamp']+=1;cases.append(f)
        for frame in cases:
            with self.subTest(columns=frame.columns.tolist()):
                with self.assertRaises(ValueError):import_csv(self.raw(frame))
        with self.assertRaises(ValueError):import_csv(b'timestamp,ax,ax,ay,az,gx,gy,gz\n0,1,1,1,1,1,1,1')
        with self.assertRaises(ValueError):import_csv(b'')
        with self.assertRaises(ValueError):import_csv(b'x'* (10*1024*1024+1))

    def test_partial_tail_and_mixed_label(self):
        f=self.frame.iloc[:250].copy();f['label']='normal';f.loc[2,'label']='pothole'
        recording=import_csv(self.raw(f))
        self.assertEqual(len(recording.windows),2)
        self.assertEqual(recording.metadata['labelled_windows'],1)
        self.assertTrue(recording.windows[0].label.isna().all())
        self.assertEqual(recording.metadata['usable_samples'],200)
        self.assertTrue(recording.metadata['warnings'])

    def test_generator_varies_and_speed_is_continuous(self):
        self.assertLess(np.max(np.abs(np.diff(self.frame.speed_kmh))),.2)
        self.assertGreater(generate_trip(0,2026,60).label.value_counts()['normal'],2000)
        self.assertGreater(self.frame.ax.std(),.08)

if __name__=='__main__':unittest.main()
