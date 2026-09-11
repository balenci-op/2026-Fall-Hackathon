r"""Durability/exposure checks using real caches and disposable human records only.

Run: .venv-ground\Scripts\python.exe -m unittest test_review_app -v
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import review_app
import review_batch


def review_hashes():
    root=review_app.ROOT/'outputs/objects'
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for parent in (root/'reviews',root/'reference_set/reviews')
            for p in parent.glob('*.json')}


class ReviewPersistenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_path=review_app.read_json(review_app.ROOT/'outputs/objects/review_batches/active_batch.json')['batch_path']

    def test_restart_history_exposure_exports_and_preservation(self):
        before=review_hashes()
        with tempfile.TemporaryDirectory(prefix='ai4infra_review_test_') as temporary:
            # The only fabricated labels in this check live in this disposable directory.
            data=Path(temporary).resolve()
            self.assertTrue(data.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            store=review_app.ReviewStore(self.batch_path,data)
            first,second=store.candidates[:2]
            self.assertEqual(store.progress(),{'reviewed':0,'total':20})
            store.position(second['id'],'temporary tester')
            store=review_app.ReviewStore(self.batch_path,data)
            self.assertEqual(store.state()['current_id'],second['id'])
            payload={'candidate_id':first['id'],'human_class':'TEMPORARY TEST CLASS',
                     'crop_quality':'incomplete object','notes':'disposable validation',
                     'reviewer_name':'temporary tester','expected_revision':0,'advance':True}
            saved=store.save(payload)
            self.assertEqual(saved['current_id'],second['id'])
            first_save=store.latest(first['id'])
            self.assertFalse(first_save['suggestions_revealed'])
            self.assertEqual(first_save['model_and_feature_references']['models']['candidate']['checkpoint'],
                             'openai/clip-vit-base-patch32')
            store=review_app.ReviewStore(self.batch_path,data)
            self.assertEqual(store.latest(first['id']),first_save)
            revealed=store.reveal(first['id'])
            self.assertTrue(revealed['clip']['candidate']['ranking'])
            self.assertTrue(revealed['clip']['context']['ranking'])
            self.assertTrue(revealed['geometry'])
            # Restart does not reset the conservative exposure flag.
            store=review_app.ReviewStore(self.batch_path,data)
            store.save({**payload,'human_class':None,'crop_quality':'unclear','expected_revision':1})
            history=store.document(first['id'])['history']
            self.assertEqual(len(history),2)
            self.assertEqual(history[0],first_save)
            self.assertIsNone(history[1]['human']['human_class'])
            self.assertTrue(history[1]['suggestions_revealed'])
            with self.assertRaisesRegex(ValueError,'changed in another window'):
                store.save(payload)
            # Additions never inherit parent guesses/features or human classes.
            addition=next(c for c in store.candidates if not c['is_existing_parent'])
            self.assertFalse(store.reveal(addition['id'])['available'])
            self.assertIsNone(store.latest(addition['id']))
            rows=store.reference_rows()
            self.assertEqual(len({r['evaluation_group'] for r in rows}),3)
            csv_text,manifest=store.export()
            self.assertEqual(manifest['reviewed_count'],1)
            self.assertFalse(any(r['eligible_for_later_classification'] for r in rows))
            self.assertIn('context_prediction_reference',csv_text)
            self.assertTrue((data/'exports/manifest.json').is_file())
        self.assertFalse(data.exists())
        self.assertEqual(before,review_hashes())

    def test_real_membership_and_display_separation(self):
        batch=review_batch.load_batch(self.batch_path)
        candidate=batch['candidates'][0]
        arrays=review_batch.load_candidate_arrays(batch,candidate['id'])
        self.assertEqual(len(arrays['xyz']),candidate['point_count'])
        keep=~np.isin(arrays['context_raw_record_indices'],arrays['raw_record_indices'])
        self.assertEqual(int(keep.sum()),candidate['crop_point_count']-candidate['point_count'])
        self.assertEqual(len(np.intersect1d(arrays['context_raw_record_indices'][keep],arrays['raw_record_indices'])),0)
        figure=review_app.figure_json(self.batch_path,candidate['id'])
        self.assertEqual(len(figure['data']),3)
        self.assertEqual(figure['data'][2]['name'],'Candidate bounds')


if __name__=='__main__':unittest.main()
