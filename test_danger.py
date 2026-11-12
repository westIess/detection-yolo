import unittest

from videores import (DANGER_CONFIRM_SECONDS, DANGER_MIN_OBSERVATIONS, PPE_CONF,
                      SafetyState, plausible_person)

STEP = 0.04  


def hidden():
   
    return {name: False for name in PPE_CONF}


def observe(state, pid, now, danger, visible=None, seen=()):
    _, events = state.update(pid, now, set(seen), set(), visible or hidden(), danger)
    return events


class DangerZoneTests(unittest.TestCase):
    def test_short_appearance_emits_nothing(self):
      
        state = SafetyState()
        events = []
        for i in range(5):
            events += observe(state, 1, i * STEP, True)
        self.assertNotIn('danger_zone', [e['type'] for e in events])

        
        state = SafetyState()
        events = []
        for now in (0.0, 0.27, 0.53, 0.80):
            events += observe(state, 1, now, True)
        self.assertNotIn('danger_zone', [e['type'] for e in events])

    def test_confirmed_entry_and_repeat_interval(self):
        state = SafetyState()
        events = []
        for i in range(13):  
            events += observe(state, 1, i * STEP, True)
        self.assertNotIn('danger_zone', [e['type'] for e in events])

        events = observe(state, 1, 13 * STEP, True)  
        self.assertEqual([e['type'] for e in events], ['danger_zone'])

        repeat = []
        for i in range(14, 146): 
            repeat += observe(state, 1, i * STEP, True)
        repeat = [e for e in repeat if e['type'] == 'danger_zone']
        self.assertEqual(len(repeat), 1)
        self.assertGreaterEqual(repeat[0]['time_s'], 5.52 - 0.05)

    def test_exit_resets_confirmation(self):
        state = SafetyState()
        events = []
        for i in range(10):  
            events += observe(state, 1, i * STEP, True)
        events += observe(state, 1, 10 * STEP, False)  
        for i in range(11, 21): 
            events += observe(state, 1, i * STEP, True)
        self.assertNotIn('danger_zone', [e['type'] for e in events])

        for i in range(21, 25):  
            events += observe(state, 1, i * STEP, True)
        self.assertEqual([e['type'] for e in events].count('danger_zone'), 1)

    def test_tracking_gap_resets_confirmation(self):
        state = SafetyState()
        events = []
        for i in range(11):  
            events += observe(state, 1, i * STEP, True)
        self.assertNotIn('danger_zone', [e['type'] for e in events])

        events += observe(state, 1, 1.00, True)  
        for t in (1.04, 1.08, 1.12, 1.16, 1.20, 1.24, 1.28, 1.32, 1.36, 1.40, 1.44, 1.48):
            events += observe(state, 1, t, True)
        self.assertNotIn('danger_zone', [e['type'] for e in events])

        events += observe(state, 1, 1.52, True)
        self.assertEqual([e['type'] for e in events].count('danger_zone'), 1)

    def test_ppe_checks_are_unchanged(self):
        
        state = SafetyState()
        visible = {name: True for name in PPE_CONF}
        events = []
        for i in range(21):  
            events += observe(state, 1, i * 0.1, False, visible=visible)
        kinds = [e['type'] for e in events]
        self.assertIn('no_helmet', kinds)
        self.assertIn('no_safety-vest', kinds)
        self.assertNotIn('danger_zone', kinds)


class PlausiblePersonTests(unittest.TestCase):
    def test_tiny_false_box_is_rejected(self):
        
        self.assertFalse(plausible_person((1204, 226, 1223, 274), 1280, 720))

    def test_real_people_are_accepted(self):
        self.assertTrue(plausible_person((810, 224, 913, 344), 1280, 720))
        self.assertTrue(plausible_person((260, 43, 350, 337), 1280, 720))

    def test_tiny_box_inside_zone_never_confirms(self):
        box = (1204, 226, 1223, 274)
        inside = plausible_person(box, 1280, 720)
        state = SafetyState()
        events = []
        for i in range(20):  
            events += observe(state, 24, i * STEP, inside)
        self.assertFalse(inside)
        self.assertEqual(events, [])
        


if __name__ == '__main__':
    unittest.main()
