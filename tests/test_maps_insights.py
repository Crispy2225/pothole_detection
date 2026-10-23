import unittest
from ui.maps import valid_location, geojson, map_html
from ui.insights import replay_summary, report_summary

class MapInsightsTests(unittest.TestCase):
    def event(self, **changes):
        return dict(dict(id='test', run_id='run', latitude=13.35, longitude=74.79, status='Inspection required', confidence=None, timestamp=2, speed_kmh=20, method='threshold',source='synthetic'), **changes)

    def test_invalid_coordinates(self):
        self.assertTrue(valid_location(self.event(latitude=0,longitude=0)))
        for value in [float('nan'),float('inf'),91,None,'bad']:
            self.assertFalse(valid_location(self.event(latitude=value)))
        self.assertFalse(valid_location(self.event(longitude=181)))

    def test_geojson_longitude_first_and_exclusion(self):
        result=geojson([self.event(),self.event(latitude=100)])
        self.assertEqual(len(result['features']),1)
        self.assertEqual(result['features'][0]['geometry']['coordinates'],[74.79,13.35])

    def test_popup_script_escape(self):
        html=map_html([self.event(id='</script><script>alert(1)</script>')],[],tiles=False)
        self.assertNotIn('</script><script>alert(1)',html)
        self.assertIn(r'\u003c/script>',html)
        self.assertIn('textContent',html)

    def test_binary_outcomes_and_empty_denominators(self):
        self.assertIsNone(replay_summary([])['precision'])
        self.assertIsNone(replay_summary([])['recall'])
        rows=[{'Ground truth (synthetic)':a,'Prediction':b} for a,b in [('pothole','pothole'),('normal','pothole'),('pothole','non_pothole'),('speed_breaker','non_pothole')]]
        result=replay_summary(rows)
        self.assertEqual([result[k] for k in ['true_positives','false_positives','false_negatives','true_negatives']],[1,1,1,1])
        self.assertEqual(result['precision'],.5)
        self.assertEqual(result['recall'],.5)

    def test_repeated_reports_are_not_unique_locations(self):
        result=report_summary([self.event(),self.event(id='second',run_id='second',confidence=.8)])
        self.assertEqual(result['reports'],2)
        self.assertEqual(result['locations'],1)
        self.assertEqual(result['runs'],2)
        self.assertEqual(result['probability_count'],1)
        self.assertEqual(result['mean_probability'],.8)

if __name__=='__main__': unittest.main()
