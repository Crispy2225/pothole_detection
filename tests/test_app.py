import tempfile
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
import storage.database as db
from config import ROOT

class AppTests(unittest.TestCase):
    def test_workspace_navigation_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'app.sqlite3'
            original_save,original_list,original_update=db.save_event,db.list_events,db.update_status
            db.save_event=lambda event:original_save(event,path)
            db.list_events=lambda:original_list(path)
            db.update_status=lambda event_id,status:original_update(event_id,status,path)
            try:
                app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
                self.assertFalse(app.exception)
                next(b for b in app.button if b.label=='Next window').click().run()
                self.assertEqual(app.session_state['cursor'],1)
                next(b for b in app.button if b.label=='Analyze full trip').click().run()
                self.assertEqual(app.session_state['cursor'],60)
                self.assertFalse(app.exception)
                self.assertTrue(original_list(path))
                app.radio(key='page').set_value('Map & reports').run()
                self.assertFalse(app.exception)
                event_id=original_list(path)[0]['id']
                next(s for s in app.selectbox if s.label=='Inspect report').select(event_id).run()
                next(s for s in app.selectbox if s.label=='Review status').select('Repaired').run()
                next(b for b in app.button if b.label=='Save review').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(original_list(path)[0]['status'],'Repaired')
                for page in ['Investigation','Model validation','Import recording','System notes']:
                    app.radio(key='page').set_value(page).run()
                    self.assertFalse(app.exception,page)
                app.radio(key='page').set_value('Trip workspace').run()
                next(b for b in app.button if b.label=='New analysis').click().run()
                self.assertEqual(app.session_state['cursor'],0)
                self.assertTrue(original_list(path))
            finally:db.save_event,db.list_events,db.update_status=original_save,original_list,original_update

if __name__=='__main__':unittest.main()
